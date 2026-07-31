import pickle
import os
import pandas as pd
from tqdm import tqdm
from synthetic_dataset.network_creation import read_edg_file
from numpy import round, mean


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


def label_by_connections_threshold(dataframe, edges, binary_label_name="binary_label"):
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
                    dataframe.at[index, binary_label_name] = 1
    return dataframe


def label_by_network(edges, posts_df):
    lt = {}

    # Step 1: Write a dictionary with the nodes and their one-hop and two hops neighbors
    for ed in tqdm(edges, "First level..."):
        if ed[0] in lt:
            lt[ed[0]]["first_level"].append(ed[1])
        else:
            lt[ed[0]] = {"first_level": [ed[1]], "second_level": set()}

    for node in tqdm(lt, "Second level..."):
        lt[node]["second_level"] = set()
        keyslist = set(lt.keys())
        for node_1 in lt[node]["first_level"]:
            if node_1 in keyslist:
                for n2 in lt[node_1]["first_level"]:
                    lt[node]["second_level"].add(n2)

    for node in tqdm(lt, "Cleaning..."):
        lt[node]["first_level"] = set(lt[node]["first_level"])
        if node in lt[node]["first_level"]:
            lt[node]["first_level"].remove(node)
        if node in lt[node]["second_level"]:
            lt[node]["second_level"].remove(node)
        lt[node]["second_level"] -= lt[node]["first_level"]

    # Step 2: for each user, calculate its label with this formula:
    # l(u) = 1/N * sum_{p \in posts_u} ((l_p**2)/5)
    # where N is the number of posts from that user

    levels_df = posts_df.groupby("account_id")["exact_level_found"].agg(list).reset_index().set_index("account_id")

    # Step 3: for each user in the lookup table, and its first and second level neighbors, convert their IDs with their label
    weights = {
        0: .1, 1: .1, 2: .2, 3: .5, 4: .75, 5: 1
    }
    users_content_based_labels = {}
    for i, row in tqdm(levels_df.iterrows()):
        users_content_based_labels[i] = sum([v*weights[v] for v in row["exact_level_found"]]) / sum(weights[v] for v in row["exact_level_found"])

    labels_dict = {}
    for j in tqdm(list(lt.keys())):
        if j in users_content_based_labels:
            labels_dict[j] = {"zero_level": users_content_based_labels[j], "first_level": [], "second_level": []}
            labels_1_neighbor = []
            labels_2_neighbor = []
            for node in lt[j]["first_level"]:
                if node in users_content_based_labels:
                    labels_1_neighbor.append(users_content_based_labels[node])
            for node in lt[j]["second_level"]:
                if node in users_content_based_labels:
                    labels_2_neighbor.append(users_content_based_labels[node])
            labels_dict[j]["first_level"] = labels_1_neighbor
            labels_dict[j]["second_level"] = labels_2_neighbor

    # Step 4: Recompute each user's score
    final_labels_dict = {}
    q = 2
    for node in tqdm(list(labels_dict.keys())):
        if int(round(labels_dict[node]["zero_level"])) != 5:
            n_1_score = sum([v / q for v in labels_dict[node]["first_level"]])
            n_2_score = sum([v / q**2 for v in labels_dict[node]["second_level"]])
            if n_2_score == 0:
                n2_value = 0
            else:
                n2_value = n_2_score / len(labels_dict[node]["second_level"])
            final_labels_dict[node] = int(round(labels_dict[node]["zero_level"] + n_1_score / len(labels_dict[node]["first_level"]) + n2_value))
        else:
            final_labels_dict[node] = 5
    for k in list(users_content_based_labels.keys()):
        if k not in final_labels_dict:
            final_labels_dict[k] = int(round(users_content_based_labels[k]))
    print(len(final_labels_dict))

    print("\nDistribution before considering the relations")
    a = list(labels_dict.values())
    a = [int(round(e["zero_level"])) for e in a]
    for v in [0,1,2,3,4,5]:
        print(a.count(v)/len(a)*100)

    print("Distribution after considering the relations")
    a = list(final_labels_dict.values())
    for v in [0, 1, 2, 3, 4, 5]:
        print(a.count(v) / len(a) * 100)

    df = pd.DataFrame(final_labels_dict.items(), columns=["account_id", "exact_level_found"])
    return df



def pagerank_based(d_labels):
    di = {}
    alpha = 0.7
    for node in d_labels:
        di[node] = alpha * d_labels[node]["zero_level"] + (1 - alpha) * mean(d_labels[node]["first_level"])

if __name__ == "__main__":
    real_posts = "real_posts_capped_64.tsv"
    synthetic_posts = "synthetic_posts.tsv"
    binary_label_name = "binary_label"
    post_separator = "  "
    synthetic_social_network = "dataset/synthetic_social_networkx4.edg"
    real_social_network = "dataset/real_social_network.edg"

    real_or_synthetic = "synthetic"
    labeling_procedure = "by_network"

    # REAL DATASET
    real_posts_df = load_df(real_posts)
    real_edges = read_edg_file(real_social_network, type_ids="str")

    # SYNTHETIC DATASET
    synthetic_posts_df = load_df(synthetic_posts)
    synthetic_edges = read_edg_file(synthetic_social_network, type_ids="str")

    if real_or_synthetic == "real":
        edges = real_edges
        posts = real_posts_df
    else:
        edges = synthetic_edges
        posts = synthetic_posts_df
    if labeling_procedure == "by_network":
        labeled = label_by_network(edges=edges, posts_df=posts)
    elif labeling_procedure == "by_max":
        labeled = label_users(real_posts_df, binary_label_name, post_separator=post_separator, concatenate_posts=None)

    labeled = labeled.rename(columns={"exact_level_found": "label"})
    labeled["label"] = labeled["label"].replace(6, 5)
    #labeled.to_csv(f"dataset/labeling/{labeling_procedure}/{real_or_synthetic}_users.tsv", sep="\t", index=False)
    print(labeled["label"].value_counts()*100/len(labeled))

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

