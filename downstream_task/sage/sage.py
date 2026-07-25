import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, SAGEConv
from synthetic_dataset.network_creation import read_edg_file
from tqdm import tqdm

torch.manual_seed(42)


def create_mappers(features_dict):
    """
    Pytorch requires nodes to have IDs that start from 0. If it's not the case in the dataset, this function creates
    a mapper from the original IDs to progressive IDs suitable with Pytorch. The function also creates an inverse
    mapper, which is the same as the mapper but with keys and values are switched
    :return:
    """
    mapper = {}
    for i, k in enumerate(features_dict):
        mapper[i] = k
    inv_map = {v: k for k, v in mapper.items()}
    return mapper, inv_map


def create_graph(inv_map, features, edg_dir, df, field_name_id, field_name_label=None, edgelist=None, inference=False):
    """
    Function to create a graph starting from the features, the edge list, and the node labels.
    :param: df: Dataframe containing the users. It is used to retrieve the node labels
    :param: id_field: Name of the id field in the dataframe. Default 'id'
    :param: label_field: Name of the label field in the dataframe. Default 'label'
    :param: edgelist: List of edges. If set to none, the function will create it. It is not set to none only when
            providing prediction for a specific set of users
    :return: (graph, node_ids). node_ids[i] is the account_id corresponding to row i of graph.x/graph.y, and of
            any predictions later computed on graph - use it to map predictions back to account_ids.
    """
    node_ids = list(inv_map.keys())
    feats = [features[int(k)] for k in node_ids]
    x = torch.Tensor(np.array(feats))
    if edgelist is None:
        edgelist = read_edg_file(edg_dir, type_pairs="tuple", mapper=inv_map)
    edges = np.array(list(zip(*edgelist)))
    if len(edges.shape) == 1:
        edges = np.zeros(shape=(2, 1))
    edge_index = torch.tensor(edges, dtype=torch.long)
    if not inference:
        labels = df.drop_duplicates(subset=field_name_id).set_index(field_name_id)[field_name_label]
        y = torch.tensor(labels.loc[node_ids].to_numpy(), dtype=torch.long)
    else:
        y = None
    graph = Data(x=x, y=y, edge_index=edge_index)
    return graph, node_ids


class SAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, loss, n_classes):
        super(SAGE, self).__init__()
        self.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.loss = loss
        self.n_classes = n_classes
        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr="mean", normalize=True))
        for _ in range(num_layers-1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean", normalize=True))
        self.output = SAGEConv(hidden_dim, n_classes, aggr="mean", normalize=True)

    def forward(self, batch, inference=False, inference_for_embedding=False):
        x = batch.x
        for i in range(len(self.convs)):
            x = self.convs[i](x, batch.edge_index)
            x = F.relu(x)

        if inference_for_embedding:
            return x
        x = self.output(x, batch.edge_index)
        x = torch.log_softmax(x, dim=-1)
        return x

    def train_sage(self, train_loader, optimizer, weights):
        self.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(self.device)
            out = self(batch)
            seed_out = out[:batch.batch_size]
            seed_y = batch.y[:batch.batch_size]
            if self.loss == "weighted":
                loss = F.nll_loss(seed_out, target=seed_y, weight=weights)
            else:
                raise ValueError("loss must be 'weighted'")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss/len(train_loader)

    @torch.no_grad()
    def test(self, data, mask):
        data = data.to(self.device)
        mask = mask.to(self.device)
        self.eval()
        out = self(data)
        val_out = out[mask]
        val_y = data.y[mask]
        loss = F.nll_loss(val_out, target=val_y).item()
        preds = val_out.argmax(dim=-1)
        acc = (preds == val_y).float().mean().item()

        y_true = val_y.cpu().numpy()
        y_pred = preds.cpu().numpy()
        labels = np.arange(self.n_classes)
        macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        return loss, acc, macro_f1, per_class_f1
