"""
In questo file creiamo i csv necessari per lanciare tgn. Al momento le interazioni gestite sono follow, authorship, comment, user evolution. Se vuoi considerare un altro
tipo di interazione devi prima creare i file negli snapshot, e poi implementare qui la gestione di quell'interazione

IMPORTANTE:
type_interaction values:
- 0 indica il follow
- 1 indica l'evoluzione degli utenti
"""

from utils import load_from_pickle, save_to_pickle
from synthetic_dataset.network_creation import read_edg_file
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import random
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

def interaction_authorship_dataset_creation(src, content_features, type_interaction, df_name: str):
    """
    This method manages the 'author' relation between a user and some content. The possible authorship types are 2:
     1) user writes a comment: source is the user ID, target is the ID of the post that the user is commenting;
     2) user writes a post: source is the user ID, target is the ID of the post that the user is writing
    type_interaction: 2 for the user posting a comment, 3 for the user publishing a post
    """
    ld = []     # list of dictionaries that will be later converted in the output dataframe
    if type_interaction == 2:
        target_field = "in_reply_to_id"
    elif type_interaction == 3:
        target_field = "id"
    else:
        raise ValueError("Select a valid value for the type interaction")

    dirs = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
    for i, d in enumerate(dirs):
        df_comments = pd.read_csv(os.path.join(src, d, df_name))
        print(len(df_comments))
        for j, row in tqdm(df_comments.iterrows()):
            source = row["account_id"]
            target = row[target_field]
            features = content_features[row["id"]]
            ld.append({"source": mapping[source], "target": target, "timestamp": i, "state_label": 0,
                       "comma_separated_list_of_features": ",".join(str(e) for e in features), "type_interaction": type_interaction})
    return pd.DataFrame(ld)

def interaction_follow_dataset_creation(src, features, full_users_set, mapping, df_name="posts_incremental.csv",
                                        shuffle_until=None, reset_network=False):
    """
    reset_network: set to true when you are in the baseline case, where we don't consider the temporal correlation of the data
    """
    ld = []
    network_so_far = set()
    edge_feat_dim = 172
    node_feat_dim = features[list(features.keys())[0]].shape[0]
    dirs = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
    if shuffle_until:
        incremental_df_name = "posts_incremental_shuffled.csv"
        to_shuffle = dirs[:shuffle_until]
        random.shuffle(to_shuffle)
        dirs = to_shuffle + dirs[shuffle_until:]
        df_snaps = []
        for d in dirs:
            df_snaps.append(pd.read_csv(os.path.join(src, d, "posts_current_snapshot.csv")).drop(columns=["Unnamed: 0"]))
            df_shuffled_inc = pd.concat(df_snaps)
            df_shuffled_inc.to_csv(os.path.join(src, d, incremental_df_name), index=False)
            print(len(df_shuffled_inc))

    for i, d in tqdm(enumerate(dirs)):
        df_incremental = pd.read_csv(os.path.join(src, d, df_name))
        network = set(read_edg_file(os.path.join(src, d, "social_network.edg"), type_pairs=tuple))
        if reset_network:
            edges_to_add = network
        else:
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
            try:
                ld.append({"follower_id": mapping[int(edge[0])], "followed_id": mapping[int(edge[1])], "timestamp": i, "state_label": 0,
                       "comma_separated_list_of_features": ",".join(str(e) for e in np.zeros(edge_feat_dim)), "type_interaction": 0})
            except KeyError:
                pass
        network_so_far = network
    return pd.DataFrame(ld)


if __name__ == "__main__":
    CREATE_TENSOR = False
    CONSIDER_SYNTHETIC = False
    CREATE_INTERACTION_DF = True
    RESET_NETWORK = False
    SHUFFLE_UNTIL = None  # -3 Se settato a none non fa lo shuffling temporale (snapshot ordinati in maniera casuale)
    CREATE_SYNTHETIC_SNAPSHOTS = 5      # setta questa variabile al unmero di snapshot che vuoi creare, oppure settala a 0
    BASE = "dataset"

    # BERT FEATURES PATHS
    BERT_FEATURES_REAL_POSTS_SRC = os.path.join(BASE, "features_bert/bert_features_real_posts.pkl")
    BERT_FEATURES_REAL_COMMENTS_SRC = os.path.join(BASE, "features_bert/bert_features_real_comments.pkl")
    BERT_FEATURES_SYNTHETIC_SRC = os.path.join(BASE, "features_bert/bert_features_few_shot_enriched.pkl")


    df_names = ["comments_current_snapshot.csv", "posts_current_snapshot.csv"]
    processed_contents_src = ["comments_only_posting_users.csv", "posts_processed.csv"]

    df_name = df_names[1]

    SNAPSHOT_SRC = os.path.join(BASE, "files_for_tgn/snapshots")        # "baseline"
    CONTENT_PROCESSED_SRC = os.path.join(BASE, processed_contents_src[1])
    OUTPUT_GAB_DF_NAME = "gab_posts"

    synthetic_posts_src = os.path.join(BASE, "synthetic_posts_irix_fewshot_enriched.csv")

    MAP_SRC = "files_for_tgn/mapping.pkl"    # "mapping_baseline.pkl"
    INV_MAP_SRC ="files_for_tgn/inv_mapping.pkl"  # "inv_mapping_baseline.pkl"

    if not os.path.exists(os.path.join(BASE, INV_MAP_SRC)):
        create_mapping(dst_inv_mapping=os.path.join(BASE, INV_MAP_SRC), dst_mapping=os.path.join(BASE, MAP_SRC),
                       src=os.path.join(BASE, "posts_processed.csv"), consider_synthetic=CONSIDER_SYNTHETIC)
    inv_mapping = load_from_pickle(os.path.join(BASE, INV_MAP_SRC))
    mapping = load_from_pickle(os.path.join(BASE, MAP_SRC))

    if CREATE_SYNTHETIC_SNAPSHOTS > 0:
        synthetic_posts = pd.read_csv(synthetic_posts_src)
        shuffled = synthetic_posts.sample(frac=1).reset_index(drop=True)
        chunk_size = len(synthetic_posts)//CREATE_SYNTHETIC_SNAPSHOTS
        dfs = [shuffled.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(CREATE_SYNTHETIC_SNAPSHOTS)]
        dfs.append(shuffled.iloc[CREATE_SYNTHETIC_SNAPSHOTS * chunk_size:])
        for i, df in enumerate(dfs):
            df.to_csv(os.path.join(SNAPSHOT_SRC, f"batch{i}_synthetic.csv"), index=False)

    # branch that manages the interactions in the real dataset
    if CREATE_INTERACTION_DF:
        full_df = pd.read_csv(CONTENT_PROCESSED_SRC)
        users = set(full_df["account_id"].tolist())

        interaction_type = 0
        if df_name.__contains__("comment"):
            interaction_type = 2
            features = load_from_pickle(BERT_FEATURES_REAL_COMMENTS_SRC)
        elif df_name.__contains__("post"):
            interaction_type = 3
            features = load_from_pickle(BERT_FEATURES_REAL_POSTS_SRC)

        # gestisce il follow e l'evoluzione degli utenti da uno snapshot all'altro
        df = interaction_follow_dataset_creation(src=SNAPSHOT_SRC, features=features,
                                                 full_users_set=users,
                                                 mapping=mapping, df_name=df_name, shuffle_until=SHUFFLE_UNTIL,
                                                 reset_network=RESET_NETWORK)

        # TODO la riga commentata sotto è per l'estensione, in cui consideriamo i commenti e le relazioni utente --writes--> post
        #df = interaction_authorship_dataset_creation(src=SNAPSHOT_SRC, content_features=features, df_name=df_name, type_interaction=interaction_type)

        df.to_csv(os.path.join(SNAPSHOT_SRC, OUTPUT_GAB_DF_NAME + ".csv"))
    # branch that manages the synthetic dataset
    if CONSIDER_SYNTHETIC:
        df = pd.read_csv(os.path.join(SNAPSHOT_SRC, OUTPUT_GAB_DF_NAME + ".csv"))
        df = df.drop(columns=[c for c in df.columns if c.__contains__("Unnamed")])
        synthetic_features = load_from_pickle(BERT_FEATURES_SYNTHETIC_SRC)
        ld = []
        node_feat_dim = synthetic_features[list(synthetic_features.keys())[0]].shape[0]
        last_ts = sorted(df["timestamp"].tolist())[-1]+1
        synthetic_df_list = [f"batch{i}_synthetic.csv" for i in [1, 2, 3]]
        already_initialized_users = []
        #synthetic_df_list.append(os.path.join(BASE, "totti_vangogh.csv"))
        for d in synthetic_df_list:
            synthetic_df = pd.read_csv(d)
            users = synthetic_df["account_id"].drop_duplicates().tolist()
            for u in tqdm(users):
                posts_by_user = synthetic_df[synthetic_df["account_id"] == u]["id"].tolist()
                posts_features = [synthetic_features[i] for i in posts_by_user]
                avg = torch.stack(posts_features).mean(dim=0).numpy()
                ld.append({"follower_id": mapping[u], "followed_id": mapping[u], "timestamp": last_ts, "state_label": 0,        # Evoluzione dell'utente nello snapshot corrente
                           "comma_separated_list_of_features": ",".join(str(e) for e in avg), "type_interaction": 1})
                if u not in already_initialized_users:
                    ld.append({"follower_id": mapping[u], "followed_id": mapping[u], "timestamp": 0, "state_label": 0,              # Aggiungo l'utente con features = 0 nello snapshot 0
                               "comma_separated_list_of_features": ",".join(str(e) for e in np.zeros(node_feat_dim)), "type_interaction": 1})
                    already_initialized_users.append(u)
            last_ts += 1

        df_synth = pd.DataFrame(ld)
        final_df = pd.concat([df, df_synth]).sort_values(by=["timestamp"])
        final_df.to_csv(os.path.join(BASE, OUTPUT_GAB_DF_NAME + "_with_synthetic.csv"))

    if CREATE_TENSOR:
        dfs_l = []
        for d in os.listdir(BASE):
            if d == "synthetic":
                dfs_l.append(pd.read_csv(os.path.join(BASE, d, "batch1_synthetic.csv")))
                dfs_l.append(pd.read_csv(os.path.join(BASE, d, "batch2_synthetic.csv")))
                dfs_l.append(pd.read_csv(os.path.join(BASE, d, "batch3_synthetic.csv")))
            elif os.path.isdir(os.path.join(BASE, d)):
                dfs_l.append(pd.read_csv(os.path.join(BASE, d, "posts_incremental.csv")))
        real_post_features = load_from_pickle(os.path.join(BASE, "bert_features_real_posts.pkl"))
        synthetic_post_features = load_from_pickle(os.path.join(BASE, "bert_features_synthetic.pkl"))
        tensor = tensor_creation([real_post_features, synthetic_post_features], mapping, dfs_l)
        np.save("dataset/tensor.npy", tensor)