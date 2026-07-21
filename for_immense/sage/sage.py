import numpy as np
import torch
import torch.nn.functional as F
from kornia.losses import binary_focal_loss_with_logits
from sklearn.metrics import roc_auc_score
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


def create_graph(inv_map, features, edg_dir, df, field_name_id, field_name_label, edgelist=None, inference=False):
    """
    Function to create a graph starting from the features, the edge list, and the node labels.
    :param: df: Dataframe containing the users. It is used to retrieve the node labels
    :param: id_field: Name of the id field in the dataframe. Default 'id'
    :param: label_field: Name of the label field in the dataframe. Default 'label'
    :param: edgelist: List of edges. If set to none, the function will create it. It is not set to none only when
            providing prediction for a specific set of users
    """
    inv_mapper_list = list(inv_map.keys())
    feats = []
    # Create the graph
    for i in inv_mapper_list:
        feats.append(features[int(i)])
    x = torch.Tensor(np.array(feats))
    y = []
    if edgelist is None:
        edgelist = read_edg_file(edg_dir, type_pairs="tuple", mapper=inv_map)
    #print(len(edgelist))
    edges = np.array(list(zip(*edgelist)))
    if len(edges.shape) == 1:
        edges = np.zeros(shape=(2, 1))
    edge_index = torch.tensor(edges, dtype=torch.long)
    if not inference:
        for k in features:
            if k in df[field_name_id].unique():
                y.append(df[df[field_name_id] == k][field_name_label].values[0])
        y = torch.tensor(np.array(y), dtype=torch.long)
    else:
        y = None
    graph = Data(x=x, y=y, edge_index=edge_index)
    return graph


class SAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, directed, loss, n_classes):
        super(SAGE, self).__init__()
        self.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.directed = directed
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
        if self.loss != "focal":
            x = torch.log_softmax(x, dim=-1)

        if inference and self.loss == "focal":
            x = F.log_softmax(x, dim=-1)
        return x

    def train_sage(self, train_loader, optimizer, weights):
        self.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(self.device)
            out = self(batch)
            if self.loss == "weighted":
                loss = F.nll_loss(out, target=batch.y, weight=weights)
            elif self.loss == "focal":
                target = torch.tensor(np.eye(2, dtype='uint8')[batch.y.cpu()], dtype=torch.float)
                loss = binary_focal_loss_with_logits(out, target=target.to(self.device), reduction="mean")
            elif self.loss == "none":
                loss = F.nll_loss(out, target=batch.y, weight=None)
            else:
                raise ValueError("loss must either be 'weighted', 'focal' or 'none'")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss/len(train_loader)

    @torch.no_grad()
    def test(self, data):
        data = data.to(self.device)
        self.eval()
        out_val = self(data)
        out_val_src = out_val[data.edge_label_index[0]]
        out_val_dst = out_val[data.edge_label_index[1]]
        link_pred = (out_val_src * out_val_dst).sum(-1)
        return roc_auc_score(data.edge_label.cpu().numpy(), link_pred.cpu().numpy())
