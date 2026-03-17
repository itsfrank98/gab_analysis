import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from synthetic_dataset.network_creation import read_edg_file, write_edg_file
import os

posts_src = "../dataset/posts_processed.csv"
network_src = "../dataset/social_network.edg"

posts_df = pd.read_csv(posts_src).drop(columns=["Unnamed: 0"])
accounts = posts_df["account_id"].tolist()
edge_list = read_edg_file(network_src, type_pairs=tuple)

def split_edges(
    edges: list[tuple],
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    random_state: int = 42,
) -> dict:
    edges = np.array(edges)
    train_pos, temp_pos = train_test_split(
        edges, test_size=val_ratio + test_ratio, random_state=random_state
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_pos, test_pos = train_test_split(
        temp_pos, test_size=relative_test, random_state=random_state
    )

    result = dict(train=train_pos, val=val_pos, test=test_pos)

    return result

def create_snapshots(edges, posts_df, dst, timestamp):
    nodes = [e[0] for e in edges]
    nodes += [e[1] for e in edges]
    nodes = list(set(nodes))
    posts_of_snapshot = posts_df[posts_df["account_id"].astype(str).isin(nodes)]
    posts_of_snapshot = posts_of_snapshot.assign(timestamp=[timestamp] * len(posts_of_snapshot))
    write_edg_file(edges, os.path.join(dst, "social_network.edg"))
    posts_of_snapshot.to_csv(os.path.join(dst, "posts_current_snapshot.csv"))


edges = split_edges(edge_list, val_ratio=.15, test_ratio=.15)
for d in ["1_train", "2_val", "3_test"]:
    os.makedirs(f"../dataset/baseline/{d}", exist_ok=True)

create_snapshots(edges["train"], posts_df, "../dataset/baseline/1_train", timestamp=1)
create_snapshots(edges["val"], posts_df, "../dataset/baseline/2_val", timestamp=2)
create_snapshots(edges["test"], posts_df, "../dataset/baseline/3_test", timestamp=3)

