import sys
import argparse
import pandas as pd
import time
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from content_analysis.text_preprocessing import TextPreprocessing
from utils import save_to_pickle, load_from_pickle
from os import makedirs
from content_analysis.WordEmb import WordEmb


def train_word2vec(field_text, field_id, embedding_size, epochs, model_dir, train_df):
    tok = TextPreprocessing()
    posts_content = tok.token_list(text_field_name=field_text, df=train_df)
    name = "w2v_{}.pkl".format(embedding_size)
    if not os.path.exists(os.path.join(model_dir, name)):
        start_emb = time.time()
        print("Training word2vec model")
        w2v_model = WordEmb(posts_content, embedding_size=embedding_size, window=10, epochs=epochs, model_dir=model_dir)
        w2v_model.train_w2v()
        save_to_pickle(os.path.join(model_dir, name), w2v_model)
        print("Elapsed time for training {} w2v: {}".format(embedding_size, time.time() - start_emb))
    else:
        print("Loading word2vec model")
        w2v_model = load_from_pickle(os.path.join(model_dir, name))
    # split content in safe and dangerous
    all_users_tokens = tok.token_dict(train_df, text_field_name=field_text, id_field_name=field_id)
    all_users_embeddings = w2v_model.text_to_vec(users=all_users_tokens)  # Get a dict of all the embeddings of each user, keeping the association with the key
    save_to_pickle(os.path.join(model_dir, f"users_embs_{embedding_size}.pkl"), all_users_embeddings)
    return all_users_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--field_text", type=str, default="text")
    parser.add_argument("--field_id", type=str, default="id")
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--train_df", type=str)
    args = parser.parse_args()

    field_text = args.field_text
    field_id = args.field_id
    embedding_size = args.embedding_size
    epochs = args.epochs
    model_dir = args.model_dir
    train_df_path = args.train_df
    train_df = pd.read_csv(train_df_path)
    train_df = train_df.drop(columns=[c for c in train_df.columns if c not in [field_text, field_id]])
    makedirs(model_dir, exist_ok=True)
    train_word2vec(field_text, field_id, embedding_size, epochs, model_dir, train_df)