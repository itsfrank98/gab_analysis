from os.path import join, exists
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import NeighborLoader
import yaml

from sage import SAGE, create_mappers, create_graph
from utils import save_to_pickle, load_from_pickle

def get_model(model_dir, ne_dim, df, we_dim, batch_size, lr, edge_path, epochs, features_dict, sizes,
              aggregation, training_weights, field_name_id, field_name_label, loss, mod):
    """
    This function applies one of the node dimensionality reduction techniques and generate the feature vectors for
    training the decision tree.
    Args:
        :param model_dir: Directory where the models will be saved.
        :param ne_dim: Dimension of the embeddings to create.
        :param df: Dataframe with the training data. The IDs will be used.
        :param batch_size: Batch size to use during training.
        :param edge_path: Path to file containing the list of edges. The file shall contain one row per each edge,
        and the row shall contain the ids of the nodes being connected and, in case of the spatial network, their
        distance. Example 1:
        12\t15
        13\t17
        means that user 12 follows user 15 and user 13 follows user 17. Example 2:
        12\t5\t0.52
        means that the users with id 12 and 5 have spatial distance equal to 0.52
        :param epochs: Epochs for training the node embedding model.
        :param features_dict: (graphsage) Dictionary having as keys the IDs of the users and as values the sum of the
        embeddings of their posts.
        :param sizes: Array containing the number of neighbors to sample for each node.
        :param training_weights: tensor of shape (num_classes,) containing the weight to give to each class while
        training the graphsage model. If None, no weights will be used (only valid when loss=="none").
        :param loss: Training loss
    Returns:
        predictions (n, num_classes): Predictions made by the node embedding model for the nodes. For each node, its
        prediction is the computed probability of the node to belong to each of the class
        node_ids (n,): account_id corresponding to each row of predictions, in the same order.
    """
    weights_path = join(model_dir, f"graphsage_{aggregation}_{ne_dim}_{we_dim}_{loss}.h5")
    model_path = join(model_dir, f"graphsage_{aggregation}_{ne_dim}_{we_dim}_{loss}.pkl")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if mod=="train":
        first_key = list(features_dict.keys())[0]
        in_channels = len(features_dict[first_key])

        mapper_train, inv_map_train = create_mappers(features_dict)
        graph, node_ids = create_graph(inv_map=inv_map_train, features=features_dict, edg_dir=edge_path,
                                       df=df, field_name_id=field_name_id, field_name_label=field_name_label)
        save_to_pickle(f"graph_{we_dim}.pkl", graph)

        # Node-level train/val split (not an edge split): validation must be on held-out
        # node labels so val_loss actually reflects classification performance.
        num_nodes = graph.num_nodes
        perm = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(42))
        num_val = max(1, int(0.1 * num_nodes))
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[num_val:]] = True
        val_mask[perm[:num_val]] = True
        graph.train_mask = train_mask
        graph.val_mask = val_mask

        graph = graph.to(device)
        sage = SAGE(in_dim=in_channels, hidden_dim=ne_dim, num_layers=len(sizes), loss=loss, n_classes=6)
        sage = sage.to(device)
        if training_weights is not None:
            training_weights = training_weights.to(device)

        train_loader = NeighborLoader(graph, num_neighbors=sizes, input_nodes=graph.train_mask, batch_size=batch_size)

        print("Training node embedding model\n")
        optimizer = torch.optim.Adam(lr=lr, params=sage.parameters(), weight_decay=1e-4)
        best_val_loss = float("inf")
        for i in range(epochs):
            train_loss = sage.train_sage(train_loader, optimizer=optimizer, weights=training_weights)
            val_loss, val_acc, val_macro_f1, val_per_class_f1 = sage.test(graph, mask=graph.val_mask)
            per_class_str = np.array2string(val_per_class_f1, precision=2)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("New best model found at epoch {}. Train loss: {}, val_loss: {}, val_acc: {}, "
                      "val_macro_f1: {}, val_per_class_f1: {}".format(
                          i, train_loss, val_loss, val_acc, val_macro_f1, per_class_str))
                torch.save(sage.state_dict(), weights_path)
            if i % 5 == 0:
                print("Epoch {}: train loss {}, val_loss: {}, val_acc: {}, val_macro_f1: {}, val_per_class_f1: {}"
                      .format(i, train_loss, val_loss, val_acc, val_macro_f1, per_class_str))
        sage.load_state_dict(torch.load(weights_path))
        save_to_pickle(model_path, sage)
    else:
        sage = load_from_pickle(model_path)
        sage.load_state_dict(torch.load(weights_path))
    return sage


def get_predictions(model, ids_to_predict, full_features_dict, full_network_path, field_name_id, df, sizes,
                    batch_size=128, device="cuda"):
    mapper, inv_map = create_mappers(full_features_dict)
    graph, node_ids = create_graph(inv_map=inv_map, features=full_features_dict, edg_dir=full_network_path,
                                   df=df, field_name_id=field_name_id, inference=True)
    graph = graph.to(device)

    # Only seed the loader with the test nodes: NeighborLoader samples the k-hop neighborhood
    # needed to compute their embeddings (which can include train nodes), without wasting
    # compute on every other train node in the full network.
    test_idx = torch.tensor(
        [i for i, nid in enumerate(node_ids) if nid in ids_to_predict],
        dtype=torch.long,
    )
    loader = NeighborLoader(graph, num_neighbors=sizes, input_nodes=test_idx, batch_size=batch_size)

    model.eval()
    pred_d = {}
    with torch.no_grad():
        for batch in loader:
            out = model(batch, inference=True).cpu().numpy()
            seed_preds = np.argmax(out[:batch.batch_size], axis=1)
            seed_node_ids = batch.n_id[:batch.batch_size].cpu().numpy()
            for pred, gidx in zip(seed_preds, seed_node_ids):
                pred_d[node_ids[gidx]] = pred
    return pred_d


if __name__ == "__main__":
    with open("parameters.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset_params = config["dataset_params"]
    mode = dataset_params["mode"]
    features_path = dataset_params["features_path"]
    df_path = dataset_params["df_path"]
    train_network_path = dataset_params["train_social_net"]
    full_network_path = dataset_params["full_social_net"]
    field_label = dataset_params["field_label"]
    field_id = dataset_params["field_id"]

    model_params = config["model_params"]
    dir_models = model_params["dir_models"]
    epochs = model_params["epochs"]
    batch_size = model_params["batch_size"]
    lr = float(model_params["lr"])
    ne_dim = model_params["ne_dim"]

    features = torch.load(features_path.format(mode))
    full_df = pd.read_csv(df_path, sep="\t")
    features_dict = {}
    for k, v in zip(features[field_id], features["embeddings"]):
        features_dict[k] = v

    df = full_df[full_df[field_id].isin(features_dict.keys())].reset_index(drop=True)   # will either contain only the training nodes or only the test nodes

    n_classes = df[field_label].nunique()
    training_weights = torch.tensor(
        compute_class_weight(class_weight="balanced", classes=np.arange(n_classes), y=df["exact_level_found"]),
        #np.ones(6),
        dtype=torch.float,
    )

    sizes = [2, 3]
    model = get_model(df=df, model_dir=dir_models, ne_dim=ne_dim, we_dim=768, batch_size=batch_size,
                      edge_path=train_network_path, epochs=epochs, features_dict=features_dict,
                      sizes=sizes, aggregation="xlmt_attm_pooled", field_name_id=field_id,
                      field_name_label=field_label, loss="weighted", training_weights=training_weights, lr=lr, mod=mode)
    features_test = torch.load(features_path.format("test"))
    ftdict = features_test.copy()
    ids_to_predict = list(ftdict[field_id])
    test_features_dict = {}
    for k, v in zip(features_test[field_id], features_test["embeddings"]):
        test_features_dict[k] = v
    features_dict.update(test_features_dict)
    preds = get_predictions(model, ids_to_predict=ids_to_predict, full_features_dict=features_dict,
                            full_network_path=full_network_path, field_name_id=field_id, df=df,
                            sizes=sizes, batch_size=batch_size)

    y_pred = [preds[i] for i in ids_to_predict]

    df = full_df.set_index("account_id").loc[ids_to_predict]
    y_true = df[field_label].tolist()
    print(classification_report(y_pred=y_pred, y_true=y_true))
    if mode == "test":
        ftdict = features_dict.copy()
        ids_to_predict = ftdict.keys()
        train_features = torch.load(features_path.format("train"))
        #features = features_dict
        train_features_dict = {}
        for k, v in zip(train_features[field_id], train_features["embeddings"]):
            train_features_dict[k] = v
        features_dict.update(train_features_dict)
        #full_network_path=train_network_path
        preds = get_predictions(model, ids_to_predict=ids_to_predict, full_features_dict=features_dict,
                                full_network_path=full_network_path, field_name_id=field_id, df=df,
                                sizes=sizes, batch_size=batch_size)
        y_pred = [preds[i] for i in ids_to_predict]

        df = df.set_index("account_id").loc[ids_to_predict]
        y_true = df[field_label].tolist()
        print(classification_report(y_pred=y_pred, y_true=y_true))
