"""
Requires python 3.10.6
"""

from laser_encoders import LaserEncoderPipeline
import argparse
import pandas as pd
from numpy import arange
import pickle
import numpy as np
from tqdm import tqdm

def aggregate_embeddings(embs_dict, df, user_ids_set, user_id_field_name, post_id_field_name, dst, pooling="mean"):
    new_dict = {}
    for user_id in tqdm(user_ids_set):
        user_posts_ids = df[df[user_id_field_name]==user_id][post_id_field_name].astype(int).tolist()
        arrays = np.array([embs_dict[post_id] for post_id in user_posts_ids])
        if pooling == "mean":
            new_dict[user_id] = arrays.mean(axis=0)
        elif pooling == "sum":
            new_dict[user_id] = arrays.sum(axis=0)

    with open(dst, "wb") as f:
        pickle.dump(new_dict, f)

def main(df, content_field_name, features_dst, post_id_field_name):
    encoder = LaserEncoderPipeline(lang="eng_Latn")
    keys = df[post_id_field_name].tolist()
    texts = df[content_field_name].tolist()

    embeddings = encoder.encode_sentences(texts)
    dct = dict(zip(keys, embeddings))

    with open(features_dst, "wb") as f:
        pickle.dump(dct, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_src", type=str, default=None, required=False)  # "all_posts_2800_3000.csv"
    parser.add_argument("--content_field_name", type=str, default="posts", required=False)
    parser.add_argument("--features_dst", type=str, help="path where the features dictionary is saved", required=False)     # , default=ft_dst
    parser.add_argument("--user_id_field_name", type=str, default="account_id", required=False)
    parser.add_argument("--post_id_field_name", type=str, default="id", required=False)
    parser.add_argument("--non_aggregated_embs_src", type=str, help="path where the non-aggregated embeddings are saved", required=False, default=None)     #"../dataset/features/sonar_features/sonar_features.pkl"
    parser.add_argument("--aggregated_features_dst", type=str, help="path where the aggregated features are saved", required=False, default="../dataset/features/sonar_features_aggregated.pkl")
    parser.add_argument("--add_post_id_flag", action="store_true", default=False)
    parser.add_argument("--pooling", type=str, required=False)

    args = parser.parse_args()
    df_src = args.df_src
    content_field_name = args.content_field_name
    features_dst = args.features_dst
    user_id_field_name = args.user_id_field_name
    post_id_field_name = args.post_id_field_name
    aggregated_features_dst = args.aggregated_features_dst
    non_aggregated_embs_src = args.non_aggregated_embs_src
    add_id = args.add_post_id_flag

    if df_src.endswith(".tsv"):
        df = pd.read_csv(df_src, sep="\t")
    elif df_src.endswith(".csv"):
        df = pd.read_csv(df_src)
    print(len(df))

    if add_id:       # quello che sta in questo if serve a settare degli id per i post sintetici. andrebbe spostato nel file in cui vengono creati i post
        df = df.drop(columns=[post_id_field_name]) if post_id_field_name in df.columns else df
        post_ids = arange(len(df))
        df[post_id_field_name] = post_ids
        new_df_name = df_src.split(".")[0]+"_id.csv"
        df.to_csv(new_df_name, index=False)
    if features_dst:
        if not features_dst.endswith(".pkl"):
            features_dst += ".pkl"
        main(df, content_field_name, features_dst, post_id_field_name)

    if aggregated_features_dst and non_aggregated_embs_src:
        with open(non_aggregated_embs_src, "rb") as f:
            feats_dict = pickle.load(f)
        user_ids_set = set(df[user_id_field_name].tolist())
        aggregate_embeddings(embs_dict=feats_dict, df=df, user_id_field_name=user_id_field_name, post_id_field_name=post_id_field_name,
                             dst=aggregated_features_dst, user_ids_set=user_ids_set)

    """
    df_src = "../dataset/posts_processed.csv"
    content_field_name = "content"
    features_dst = "bert_features_posts.pkl"
    """
