from openai import OpenAI
import pandas as pd
from tqdm import tqdm

def read_edg_file(path):
    edges = []
    with open(path, "r") as f:
        for line in f.readlines():
            edges.append(line.split())
    return edges

def write_edg_file(edges, path):
    with open(path, "w") as f:
        for ed in edges:
            f.write(f"{ed[0]}\t{ed[1]}\n")

def pick_n_posts_per_user(df, idlist, n):
    ld = []
    for id in tqdm(idlist):
        sorted_posts = df[df["account_id"] == id].sort_values(by="created_at", ascending=False)
        ld.append(sorted_posts[:n])

    concatenated = pd.concat(ld)
    return concatenated

def create_files_for_custom_network_dimension(n_users, n_posts, all_users_path="dataset/preprocessed/profiles_users_all.csv",
          all_posts_path="all_posts_all_users_nona.csv", network_path="dataset/network/social_network.edg"):
    """This function creates a sub-dataset of the original dataset. the user can specify how many users he awants in the
    dataset, and how many posts per user. The function will take th n_users with most published posts, and keep for
    each of them the n_posts most recently published posts. Then the function will also extract, from the original
    social network, the network containing only the sampled users"""

    df_users = pd.read_csv(all_users_path)
    posts = pd.read_csv(all_posts_path)
    edges = read_edg_file(network_path)
    df_counts = posts.groupby("account_id").size().reset_index(name="count").sort_values(by=["count"],
                                                                                            ascending=False)[:n_users]
    users_with_top_posts_id = df_counts["account_id"].tolist()
    final_posts = pick_n_posts_per_user(posts, users_with_top_posts_id, n_posts)

    users_list = final_posts["account_id"].astype(str).tolist()
    final_users = df_users[df_users.id.astype(str).isin(users_list)]

    edges_to_keep = []
    for ed in edges:
        if str(ed[0]) in users_list and str(ed[1]) in users_list:
            edges_to_keep.append(ed)

    return final_posts, final_users, edges_to_keep


if __name__ == "__main__":
    fields_to_consider = ["bio", "content"]
    posts_path = "../dataset/preprocessed/df_for_network/posts_5.csv"
    network_path = "../dataset/network/social_network_1k.edg"
    users_path = "../dataset/preprocessed/df_for_network/users_1k.csv"

    df_posts = pd.read_csv(posts_path)
    df_users = pd.read_csv(users_path)


    for i, r in df_users.iterrows():
        id = r["account_id"]

