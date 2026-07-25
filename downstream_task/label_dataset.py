import pickle
import os
import pandas as pd
from tqdm import tqdm
from synthetic_dataset.network_creation import read_edg_file

def load_df(path):
    sep = "\t" if path.endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep)
    df["exact_level_found"] = df["exact_level_found"].astype(int)
    return df


def label_users(dataframe, binary_label_name, post_separator, concatenate_posts):
    """For each account_id, assign the highest post level found and
    concatenate all of that user's posts."""
    # setta a true se vuoi fare in modo che la label binaria dipenda da quella multi class. Se è false, la label binaria è 1 se una certa percentuale
    # di post ha livello >= 2. Quindi se un utente ha un solo post di livello 5, la sua label binaria potrebbe essere 1, mentre quella multiclass sarebbe 5.
    # settando questo flag a True, quello stesso utente avrebbe label 1

    if binary_label_name is not None:
        grouped = dataframe.groupby("account_id").agg(
            exact_level_found=("exact_level_found", "max"),
            posts=("content", lambda posts: post_separator.join(posts.astype(str))),
        ).reset_index()
        grouped[binary_label_name] = (grouped["exact_level_found"] >= 2).astype(int)
        return grouped[["account_id", "exact_level_found", "posts", binary_label_name]]
    else:
        dataframe = dataframe.copy()
        if concatenate_posts:
            grouped = dataframe.groupby("account_id").agg(
                exact_level_found=("exact_level_found", "max"),
                binary_label=(binary_label_name, lambda levels: int(levels.mean() > 0.3)),
                posts=("content", lambda posts: post_separator.join(posts.astype(str))),
            ).reset_index()
            return grouped[["account_id", "exact_level_found", "posts"]]
        else:
            grouped = dataframe.groupby("account_id").agg(
                exact_level_found=("exact_level_found", "max"),
            ).reset_index()
            return grouped



def label_by_connections(dataframe, edges, binary_label_name="binary_label"):
    conn_dict = {}
    df_labels = dataframe.copy()
    df_labels.index = df_labels["account_id"]
    df_labels = df_labels.drop(columns=["account_id"])
    user_labels_dict = df_labels[binary_label_name].to_dict()
    for ed in tqdm(edges):
        ed[0] = int(ed[0])
        ed[1] = int(ed[1])
        if ed[1] in user_labels_dict:
            if ed[0] in conn_dict:
                conn_dict[int(ed[0])].append(user_labels_dict[int(ed[1])])
            else:
                conn_dict[int(ed[0])] = [user_labels_dict[int(ed[1])]]
    for index, row in dataframe.iterrows():
        if row[binary_label_name] == 0:
            id = row["account_id"]
            if id in conn_dict:
                flws = conn_dict[id]
                risky_neighborhood_percentage = sum(flws) / len(flws) * 100
                if risky_neighborhood_percentage > 25:
                    #print("goddayum")
                    dataframe.at[index, binary_label_name] = 1
    return dataframe

def build_network_lookup(edges):
    d={}
    for ed in tqdm(edges, "First level..."):
        if ed[0] in d:
            d[ed[0]]["first_level"].append(ed[1])
        else:
            d[ed[0]] = {"first_level": [ed[1]], "second_level": set()}

    for node in tqdm(d, "Second level..."):
        d[node]["second_level"] = set()
        keyslist = set(d.keys())
        for node_1 in d[node]["first_level"]:
            if node_1 in keyslist:
                for n2 in d[node_1]["first_level"]:
                    d[node]["second_level"].add(n2)
    """for node in tqdm(d, "Cleaning..."):
        d[node]["first_level"] = set(d[node]["first_level"])
        if node in d[node]["first_level"]:
            d[node]["first_level"].remove(node)
        if node in d[node]["second_level"]:
            d[node]["second_level"].remove(node)
        d[node]["second_level"] -= d[node]["first_level"]"""
    return d

def neighbor_based_labeling(df, network):
    pass


if __name__ == "__main__":
    real_posts = "real_posts.tsv"
    synthetic_posts = "synthetic_posts.tsv"
    binary_label_name = "binary_label"
    post_separator = "  "
    synthetic_social_network = "dataset/synthetic_social_network.edg"
    real_social_network = "dataset/real_social_network.edg"

    # REAL DATASET
    real = load_df(real_posts)
    real_edges = read_edg_file(real_social_network, type_pairs="list")
    lt_path = os.path.join("dataset", "lookup_table.pkl")
    if not os.path.exists(lt_path):
        lt = build_network_lookup(real_edges)
        with open("dataset/lookup_table.pkl", "wb") as f:
            pickle.dump(lt, f)
    else:
        with open("dataset/lookup_table.pkl", "rb") as f:
            lt = pickle.load(f)

    posts_df = pd.read_csv(real_posts, sep="\t")
    levels_df = posts_df.groupby("account_id")["exact_level_found"].agg(list).reset_index().set_index("account_id")
    d_labels = {}
    for i, row in tqdm(levels_df.iterrows()):
        d_labels[i] = sum([(v**2)/5 for v in row["exact_level_found"]])/len(row["exact_level_found"])
   #TODO ARROTONDARE I VALORI


    """real_labeled = label_users(real, binary_label_name, post_separator=post_separator)
    real_labeled = label_by_connections(real_labeled, real_edges)
    real_labeled.to_csv("dataset/real_users.tsv", sep="\t", index=False)"""

    '''
    # SYNTHETIC DATASET
    synthetic = load_df(synthetic_posts)
    synthetic = synthetic.drop(columns=[c for c in synthetic.columns if c not in ["account_id", "content", "exact_level_found"]])
    #synthetic[binary_label_name] = (synthetic['exact_level_found'] >= 2).astype(int)
    synthetic_labeled = label_users(synthetic, binary_label_name=None, post_separator=post_separator, concatenate_posts=False)
    synthetic_edges = read_edg_file(synthetic_social_network, type_pairs="list")
    #print(synthetic_labeled["binary_label"].value_counts())
    synthetic_labeled.to_csv("sage/synthetic_users_labels.tsv", sep="\t", index=False)
    #synthetic_labeled = label_by_connections(synthetic_labeled, synthetic_edges)
    #print(synthetic_labeled["binary_label"].value_counts())
    #synthetic_labeled.to_csv("dataset/synthetic_users.tsv", sep="\t", index=False)
    '''

