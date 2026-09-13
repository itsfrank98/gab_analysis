import os
import pandas as pd
from tqdm import tqdm

from synthetic_dataset.network_creation import read_edg_file, write_edg_file

mode = "rs"
real_network_src = "original_networks/real_social_network.edg"
synthetic_network_src = "original_networks/synthetic_social_network.edg"
src_files_dir = f"percented_train_{mode}"


real_edges = read_edg_file(real_network_src, type_ids=str)
synthetic_edges = read_edg_file(synthetic_network_src, type_ids=str)
if mode == "rs":
    network = real_edges + synthetic_edges
else:
    network = real_edges
for perc in tqdm([10, 20, 30, 40, 50, 60]):
    for fold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        train_ids = set(pd.read_csv(os.path.join(src_files_dir, f"{perc}_perc", "cross_validation", f"fold_{fold}", "train_ids.tsv"), sep="\t")["account_id"].tolist())
        test_ids = set(pd.read_csv(os.path.join(src_files_dir, f"{perc}_perc", "cross_validation", f"fold_{fold}", "test_ids.tsv"), sep="\t")["account_id"].tolist())
        edges_to_keep_train = []
        edges_to_keep_test = []
        for ed in network:
            if ed[0] in train_ids and ed[1] in train_ids:
                edges_to_keep_train.append(ed)
            if ed[0] in test_ids and ed[1] in test_ids:
                edges_to_keep_test.append(ed)

        write_edg_file(edges_to_keep_train, os.path.join(src_files_dir, f"{perc}_perc", "cross_validation", f"fold_{fold}", "train_network.edg"))
        write_edg_file(edges_to_keep_test, os.path.join(src_files_dir, f"{perc}_perc", "cross_validation", f"fold_{fold}", "test_network.edg"))



