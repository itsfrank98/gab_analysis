from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import pickle
import pandas as pd
import torch
import argparse


def main(df, content_field_name, features_dst, post_id_field_name):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    model = DistilBertModel.from_pretrained("distilbert-base-cased")
    dct = {}

    for index, row in tqdm(df.iterrows()):
        with torch.no_grad():
            post_text = row[content_field_name]
            encoded_input = tokenizer(post_text, return_tensors='pt', truncation=True)
            output = model(**encoded_input)
            output = output.last_hidden_state.mean(dim=1).squeeze()
            dct[row[post_id_field_name]] = output
        if index % 100 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    with open(features_dst+".pkl", "wb") as f:
        pickle.dump(dct, f)

def aggregate_embeddings(embs_dict, df, user_ids_set, user_id_field_name, post_id_field_name, dst):
    new_dict = {}
    for user_id in tqdm(user_ids_set):
        user_posts_ids = df[df[user_id_field_name]==user_id][post_id_field_name].tolist()
        tensors = [embs_dict[post_id] for post_id in user_posts_ids]
        new_dict[user_id] = torch.stack(tensors).mean(dim=0)

    with open(dst+".pkl", "wb") as f:
        pickle.dump(new_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_src", type=str, default="data/train.csv", required=True)
    parser.add_argument("--content_field_name", type=str, default="content", required=False)
    parser.add_argument("--features_dst", type=str, help="path where the features dictionary is saved", required=False)
    parser.add_argument("--user_id_field_name", type=str, default="user_id", required=False)
    parser.add_argument("--post_id_field_name", type=str, default="post_id", required=False)
    parser.add_argument("--aggregated_features_dst", type=str, help="path where the aggregated features are saved", required=False)
    parser.add_argument("--non_aggregated_embs_src", type=str, help="path where the non-aggregated embeddings are saved", required=False)
    args = parser.parse_args()
    df_src = args.df_src
    content_field_name = args.content_field_name
    features_dst = args.features_dst
    user_id_field_name = args.user_id_field_name
    post_id_field_name = args.post_id_field_name
    aggregated_features_dst = args.aggregated_features_dst
    non_aggregated_embs_src = args.non_aggregated_embs_src

    df = pd.read_csv(df_src)
    if features_dst:
        main(df, content_field_name, features_dst, post_id_field_name)

    if aggregated_features_dst:
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

