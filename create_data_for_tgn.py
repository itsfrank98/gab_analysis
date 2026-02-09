from utils import load_from_pickle, save_to_pickle
from synthetic_dataset.network_creation import read_edg_file
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import torch

def create_mapping(dst_mapping, dst_inv_mapping):
    df_real = pd.read_csv("dataset/snapshots/Test/posts_incremental.csv")
    batch_synthetic_1 = pd.read_csv("dataset/snapshots/batch1_synthetic.csv")
    batch_synthetic_2 = pd.read_csv("dataset/snapshots/batch2_synthetic.csv")
    batch_synthetic_3 = pd.read_csv("dataset/snapshots/batch3_synthetic.csv")
    real_users = list(set(df_real["account_id"].tolist()))
    batch_1_users = list(set(batch_synthetic_1["user_id"].tolist()))
    batch_2_users = list(set(batch_synthetic_2["user_id"].tolist()))
    batch_3_users = list(set(batch_synthetic_3["user_id"].tolist()))

    users = real_users + batch_1_users + batch_2_users + batch_3_users
    mapping = {users[v]: v for v in range(len(users))}
    inv_mapping = {mapping[k]: k for k in list(mapping.keys())}
    save_to_pickle(dst_mapping, mapping)
    save_to_pickle(dst_inv_mapping, inv_mapping)

def tensor_creation(post_features, mapping, dfs):
    tensor = np.zeros((len(dfs), len(mapping), 768))
    for i, df in tqdm(enumerate(dfs)):
        if i <= 5:
            features = post_features[0]
        else:
            features = post_features[1]
        users_so_far = list(set(df["account_id"].tolist()))
        for u in users_so_far:
            posts_by_user = df[df["account_id"] == u]["id"].tolist()
            posts_features = [features[i] for i in posts_by_user]
            avg = torch.stack(posts_features).mean(dim=0).numpy()
            tensor[i:, mapping[u], :] = avg
    return tensor

def interaction_dataset_creation(src: str, features, full_users_set):
    ld = []
    edge_feat_dim = 172
    node_feat_dim = features[list(features.keys())[0]].shape[0]
    dirs = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
    for i, d in tqdm(enumerate(dirs)):
        df_incremental = pd.read_csv(os.path.join(src, d, "posts_incremental.csv"))
        network = read_edg_file(os.path.join(src, d, "social_network.edg"))
        users_so_far = list(set(df_incremental["account_id"].tolist()))
        for u in tqdm(users_so_far):
            posts_by_user = df_incremental[df_incremental["account_id"] == u]["id"].tolist()
            posts_features = [features[i] for i in posts_by_user]
            avg = torch.stack(posts_features).mean(dim=0).numpy()
            ld.append({"follower_id": u, "followed_id": u, "timestamp": i, "state_label": 0,
                           "comma_separated_list_of_features": ",".join(str(e) for e in avg), "type_interaction": 1})
        if i == 0:
            users_not_in_first_snap = list(full_users_set - set(users_so_far))
            for u in users_not_in_first_snap:
                ld.append({"follower_id": u, "followed_id": u, "timestamp": 0, "state_label": 0,
                           "comma_separated_list_of_features": ",".join(str(e) for e in np.zeros(node_feat_dim)), "type_interaction": 1})
        for edge in network:
            ld.append({"follower_id": edge[0], "followed_id": edge[1], "timestamp": i, "state_label": 0,
                           "comma_separated_list_of_features": ",".join(str(e) for e in np.zeros(edge_feat_dim)), "type_interaction": 0})
        df = pd.DataFrame(ld)
        df.to_csv("dataset/snapshots_new/provisory.csv", index=False)
    return pd.DataFrame(ld)



CREATE_TENSOR = False
CREATE_INTERACTION_DF = True

src = "dataset/snapshots_new"

if CREATE_INTERACTION_DF:
    features = load_from_pickle("dataset/snapshots_new/bert_features_real_posts.pkl")
    full_df = pd.read_csv("dataset/posts_processed.csv")
    users = set(full_df["account_id"].tolist())
    df = interaction_dataset_creation(src=src, features=features, full_users_set=users)
    df.to_csv(os.path.join(src, "gab1.csv"))
if CREATE_TENSOR:
    if not os.path.exists(os.path.join(src, "inv_map.pkl")):
        create_mapping(dst_inv_mapping=os.path.join(src, "inv_map.pkl"), dst_mapping=os.path.join(src, "mapping.pkl"))
    mapping = load_from_pickle(os.path.join(src, "inv_map.pkl"))
    dfs_l = []
    for d in os.listdir(src):
        if d == "synthetic":
            dfs_l.append(pd.read_csv(os.path.join(src, d, "batch1_synthetic.csv")))
            dfs_l.append(pd.read_csv(os.path.join(src, d, "batch2_synthetic.csv")))
            dfs_l.append(pd.read_csv(os.path.join(src, d, "batch3_synthetic.csv")))
        elif os.path.isdir(os.path.join(src, d)):
            dfs_l.append(pd.read_csv(os.path.join(src, d, "posts_incremental.csv")))
    real_post_features = load_from_pickle(os.path.join(src, "bert_features_real_posts.pkl"))
    synthetic_post_features = load_from_pickle(os.path.join(src, "bert_features_synthetic.pkl"))
    tensor = tensor_creation([real_post_features, synthetic_post_features], mapping, dfs_l)
    np.save("dataset/tensor.npy", tensor)