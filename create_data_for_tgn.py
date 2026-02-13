from utils import load_from_pickle, save_to_pickle
from synthetic_dataset.network_creation import read_edg_file
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import torch

def create_mapping(dst_mapping, dst_inv_mapping, src, consider_synthetic=True):
    df_real = pd.read_csv(src)
    users = list(set(df_real["account_id"].tolist()))
    if consider_synthetic:
        synthetic_batches = [pd.read_csv(f"dataset/snapshots/batch{i}_synthetic.csv") for i in [1, 2, 3]]
        user_batches = [b["account_id"].drop_duplicates().tolist() for b in synthetic_batches]
        users += user_batches[0] + user_batches[1] + user_batches[2]

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


def interaction_dataset_creation(src: str, features, full_users_set, mapping):
    ld = []
    network_so_far = set()
    edge_feat_dim = 172
    node_feat_dim = features[list(features.keys())[0]].shape[0]
    dirs = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
    for i, d in tqdm(enumerate(dirs)):
        df_incremental = pd.read_csv(os.path.join(src, d, "posts_incremental.csv"))
        network = set(read_edg_file(os.path.join(src, d, "social_network.edg"), type_pairs="tuple"))
        edges_to_add = network - network_so_far
        users_so_far = list(set(df_incremental["account_id"].tolist()))
        for u in tqdm(users_so_far):
            posts_by_user = df_incremental[df_incremental["account_id"] == u]["id"].tolist()
            posts_features = [features[i] for i in posts_by_user]
            avg = torch.stack(posts_features).mean(dim=0).numpy()
            ld.append({"follower_id": mapping[u], "followed_id": mapping[u], "timestamp": i, "state_label": 0,
                       "comma_separated_list_of_features": ",".join(str(e) for e in avg), "type_interaction": 1})
        if i == 0:
            users_not_in_first_snap = list(full_users_set - set(users_so_far))
            for u in users_not_in_first_snap:
                ld.append({"follower_id": mapping[u], "followed_id": mapping[u], "timestamp": 0, "state_label": 0,
                           "comma_separated_list_of_features": ",".join(str(e) for e in np.zeros(node_feat_dim)), "type_interaction": 1})
        for edge in edges_to_add:
            ld.append({"follower_id": mapping[int(edge[0])], "followed_id": mapping[int(edge[1])], "timestamp": i, "state_label": 0,
                       "comma_separated_list_of_features": ",".join(str(e) for e in np.zeros(edge_feat_dim)), "type_interaction": 0})
        network_so_far = network
    return pd.DataFrame(ld)


CREATE_TENSOR = False
CREATE_INTERACTION_DF = False
CONSIDER_SYNTHETIC = True
SRC = "dataset/snapshots"

if not os.path.exists(os.path.join(SRC, "inv_map.pkl")):
    create_mapping(dst_inv_mapping=os.path.join(SRC, "inv_map.pkl"), dst_mapping=os.path.join(SRC, "mapping.pkl"),
                   src="dataset/posts_processed.csv", consider_synthetic=CONSIDER_SYNTHETIC)
inv_mapping = load_from_pickle(os.path.join(SRC, "inv_map.pkl"))
mapping = load_from_pickle(os.path.join(SRC, "mapping.pkl"))


if CREATE_INTERACTION_DF:
    features = load_from_pickle(os.path.join(SRC, "bert_features_real_posts.pkl"))
    full_df = pd.read_csv("dataset/posts_processed.csv")
    users = set(full_df["account_id"].tolist())
    df = interaction_dataset_creation(src=SRC, features=features, full_users_set=users, mapping=mapping)
    df.to_csv(os.path.join(SRC, "gab.csv"))
if CREATE_TENSOR:
    dfs_l = []
    for d in os.listdir(SRC):
        if d == "synthetic":
            dfs_l.append(pd.read_csv(os.path.join(SRC, d, "batch1_synthetic.csv")))
            dfs_l.append(pd.read_csv(os.path.join(SRC, d, "batch2_synthetic.csv")))
            dfs_l.append(pd.read_csv(os.path.join(SRC, d, "batch3_synthetic.csv")))
        elif os.path.isdir(os.path.join(SRC, d)):
            dfs_l.append(pd.read_csv(os.path.join(SRC, d, "posts_incremental.csv")))
    real_post_features = load_from_pickle(os.path.join(SRC, "bert_features_real_posts.pkl"))
    synthetic_post_features = load_from_pickle(os.path.join(SRC, "bert_features_synthetic.pkl"))
    tensor = tensor_creation([real_post_features, synthetic_post_features], mapping, dfs_l)
    np.save("dataset/tensor.npy", tensor)
if CONSIDER_SYNTHETIC:
    df = pd.read_csv(os.path.join(SRC, "gab.csv"))
    df = df.drop(columns=[c for c in df.columns if c.__contains__("Unnamed")])
    synthetic_features = load_from_pickle(os.path.join(SRC, "bert_features_synthetic.pkl"))
    ld = []
    node_feat_dim = synthetic_features[list(synthetic_features.keys())[0]].shape[0]
    last_ts = sorted(df["timestamp"].tolist())[-1]+1
    synthetic_df_list = [os.path.join(SRC, f"batch{i}_synthetic.csv") for i in [1,2,3]]
    for d in synthetic_df_list:
        synthetic_df = pd.read_csv(d)
        users = synthetic_df["account_id"].drop_duplicates().tolist()
        for u in tqdm(users):
            posts_by_user = synthetic_df[synthetic_df["account_id"] == u]["id"].tolist()
            posts_features = [synthetic_features[i] for i in posts_by_user]
            avg = torch.stack(posts_features).mean(dim=0).numpy()
            ld.append({"follower_id": mapping[u], "followed_id": mapping[u], "timestamp": last_ts, "state_label": 0,
                       "comma_separated_list_of_features": ",".join(str(e) for e in avg), "type_interaction": 1})
            ld.append({"follower_id": mapping[u], "followed_id": mapping[u], "timestamp": 0, "state_label": 0,
                       "comma_separated_list_of_features": ",".join(str(e) for e in np.zeros(node_feat_dim)), "type_interaction": 1})
        last_ts += 1

    df_synth = pd.DataFrame(ld)
    final_df = pd.concat([df, df_synth]).sort_values(by=["timestamp"])
    final_df.to_csv(os.path.join(SRC, "gab_with_synthetic.csv"))