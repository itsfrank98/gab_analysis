import numpy as np
import os
from utils import load_from_pickle, save_to_pickle
import pandas as pd
from tqdm import tqdm
import torch

def create_mapping():
    df_real = pd.read_csv("dataset/snapshots/Jul 25/posts_incremental.csv")
    batch_synthetic_1 = pd.read_csv("dataset/snapshots/batch1_synthetic.csv")
    batch_synthetic_2 = pd.read_csv("dataset/snapshots/batch2_synthetic.csv")
    batch_synthetic_3 = pd.read_csv("dataset/snapshots/batch3_synthetic.csv")
    real_users = list(set(df_real["account_id"].tolist()))
    batch_1_users = list(set(batch_synthetic_1["user_id"].tolist()))
    batch_2_users = list(set(batch_synthetic_2["user_id"].tolist()))
    batch_3_users = list(set(batch_synthetic_3["user_id"].tolist()))

    users = real_users + batch_1_users + batch_2_users + batch_3_users
    mapping = {users[v]: v for v in range(len(users))}
    save_to_pickle("dataset/snapshots/mapping.pkl", mapping)

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


#create_mapping()

src = "dataset/snapshots"
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