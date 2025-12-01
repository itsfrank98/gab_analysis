import numpy as np
import json
import os
import pandas as pd
import time
from openai import OpenAI
from tqdm import tqdm
from llama_cpp import Llama

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
        sorted_posts = df[df["account_id"].astype(str) == str(id)].sort_values(by="created_at", ascending=False)
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

def count_tokens(messages):
    llm = Llama(
        model_path="../../../.lmstudio/models/MaziyarPanahi/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct.Q3_K_S.gguf"
    )
    tokens = llm.tokenize(
        llm.create_chat_completion_openai_v1(messages=messages)["choices"][0]["message"]["content"].encode()
    )

def create_unique_edgelist(src_dir, dst_dir):
    l = []
    for fname in os.listdir(src_dir):
        id = fname.split("_")[0]
        with open(os.path.join(src_dir, fname), "r") as f:
            friends = json.loads(f)
            friends = friends.split("\n")
            for friend in friends:
                l.append((id, friend))
    print(len(l))
    with open(os.path.join(dst_dir, "edgelist.edg"), "w") as f:
        for e in l:
            f.write(f"{e[0]}\t{e[1]}\n")


if __name__ == "__main__":
    base_url = "http://127.0.0.1:1234/v1"
    client = OpenAI(
        base_url=base_url,
        api_key="OPENAI_API_KEY",
    )

    n_posts_per_user = 5
    n_batches = 10
    fields_to_consider = ["bio", "content"]
    posts_path = f"../dataset/preprocessed/df_for_network/posts_{n_posts_per_user}.csv"
    network_path = "../dataset/network/social_network_1k.edg"
    users_path = "../dataset/preprocessed/df_for_network/users_1k.csv"

    df_posts = pd.read_csv(posts_path)
    df_users = pd.read_csv(users_path)

    splitted_users = np.array_split(df_users, n_batches)

    if "Unnamed: 0" in df_posts.columns:
        df_posts = df_posts.drop("Unnamed: 0", axis=1)
    df_users = df_users.drop(columns=[c for c in df_users.columns if c not in ["id", "note"]], axis=1)

    for i, r in tqdm(df_users.iterrows()):
        user_id, user_bio = r["id"], r["note"]
        user_posts = df_posts[df_posts["account_id"] == user_id]["content"].tolist()
        if len(user_posts) == 0:
            user_post = ""
        else:
            user_post = user_posts[0]
        system_prompt = (f"You have a profile on a social network website. This is the bio of your profile: \"{user_bio}\". "
                 f"This is the most recent post you published: {user_post}. You will be provided a "
                 "list of people in the network, where each person is described by these fields: "
                 "ID (string), bio (string), post: (string). Which of these people will you start following? Provide a list "
                 "of the accounts *YOU* will follow in the format ID, ID, ID, etc. Do not include any other text in your response. "
                 "Do not include any people who are not listed below.")


        for b, batch_df in enumerate(splitted_users):
            user_prompt = ""
            if not os.path.exists(f"gab_network_llm/{user_id}_{b}.json"):
                for j, r2 in batch_df.iterrows():
                    id2 = r2["id"]
                    if id2 != user_id:
                        if pd.isna(r2["note"]):
                            user_bio = ""
                        else:
                            user_bio = r2["note"].replace("\"", "'")
                        user_posts = df_posts[df_posts["account_id"] == id2]["content"].tolist()
                        if len(user_posts) == 0:
                            user_posts = ""
                        else:
                            for k in range(len(user_posts)):
                                user_posts[k] = user_posts[k].replace("\"", "'")
                            user_posts = user_posts[0]
                        user_prompt += "ID: {}, bio: {}, post: {}\n".format(id2, user_bio, user_posts)

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                #count_tokens(messages)

                now = time.time()
                resp = client.chat.completions.create(
                    model="local-model",
                    messages=messages,
                    temperature=0.8
                )
                print(time.time() - now)
                followees = resp.choices[0].message.content
                with open(f"gab_network_llm/{user_id}_{b}.json", "w") as f:
                    json.dump(followees, f)

