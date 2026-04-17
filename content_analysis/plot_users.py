"""
Qui fai il plot delle features.
Note importanti:
- Se lo devi lanciare sul server, non lanciarlo da linea di comando perche non vedresti le immagini. Lancialo con tasto destro+run
e cambia i path dai derfault dell'argprse
- Il campo 'unit' serve solo per il titolo del plot, non fa nessuna aggregazione. Se vuoi plottare gli users devi passare
i path delle user features, se vuoi plottare i post devi passare i path ai post
"""
import pickle
import numpy as np
import argparse
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
print(os.curdir)
print(os.listdir("../dataset"))


def main(synthetic_feats_path, real_feats_path, consider_label, method, unit="users"):
    with open(synthetic_feats_path, 'rb') as f:
        synthetic_feats = pickle.load(f)
    with open(real_feats_path, 'rb') as f:
        real_feats = pickle.load(f)
    print(len(real_feats), len(synthetic_feats))

    synthetic_matrix = np.array([v for v in synthetic_feats.values()])

    if consider_label:
        labeled_users = pd.read_csv("../dataset/vicuna_classification_results.csv")
        safe_users = labeled_users[labeled_users["binary_label"] == 0]["user_id"].tolist()
        risky_users = labeled_users[labeled_users["binary_label"] == 1]["user_id"].tolist()
        safe_matrix = np.array([real_feats[k] for k in list(real_feats.keys()) if k in safe_users])
        risky_matrix = np.array([real_feats[k] for k in list(real_feats.keys()) if k in risky_users])
        matrix = np.vstack((safe_matrix, risky_matrix, synthetic_matrix))
        colors = ["green"] * safe_matrix.shape[0] + ["orange"] * risky_matrix.shape[0] + ["red"] * len(synthetic_feats)
        labels = ["safe"] * safe_matrix.shape[0] + ["risky"] * risky_matrix.shape[0] + ["synthetic"] * len(synthetic_feats)
    else:
        real_matrix = np.array([v for v in real_feats.values()])
        matrix = np.vstack((real_matrix, synthetic_matrix))
        colors = ["blue"] * len(real_feats) + ["red"] * len(synthetic_feats)
        labels = ["real"] * len(real_feats) + ["synthetic"] * len(synthetic_feats)

    if method.lower()=="pca":
        reduction = PCA(n_components=2)  # or however many components you want
    else:
        reduction = TSNE(n_components=2)
    reduced = reduction.fit_transform(matrix)
    print(reduced.shape)
    plt.figure(figsize=(8, 6))

    if consider_label:
        n_safe = safe_matrix.shape[0]
        n_risky = risky_matrix.shape[0]
        plt.scatter(reduced[:n_safe, 0], reduced[:n_safe, 1], color='green', alpha=.3, s=10, label='safe')
        plt.scatter(reduced[n_safe:n_safe+n_risky, 0], reduced[n_safe:n_safe+n_risky, 1], color='orange', alpha=.3, s=10, label='risky')
        plt.scatter(reduced[n_safe+n_risky:, 0], reduced[n_safe+n_risky:, 1], color='red', alpha=.3, s=10, label='synthetic')
    else:
        plt.scatter(reduced[:len(real_feats), 0], reduced[:len(real_feats), 1], color='blue', alpha=.3, s=10, label='real')
        plt.scatter(reduced[len(real_feats):, 0], reduced[len(real_feats):, 1], color='red', alpha=.3, s=10, label='synthetic')

    plt.legend()

    keys = list(real_feats.keys()) + list(synthetic_feats.keys())
    # Optionally, annotate each point with its key

    plt.xlabel(f'{method}1')
    plt.ylabel(f'{method}2')
    plt.title(f'{method} Projection - {unit}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_feats_path", type=str, default="../dataset/features_bert/bert_features_aggregated_irix_few_shot.pkl")
    parser.add_argument("--real_feats_path", type=str, default="../dataset/features_bert/bert_features_real_users.pkl")
    parser.add_argument("--consider_label", action="store_true")
    parser.add_argument("--method", type=str, default="pca")
    parser.add_argument("--unit", type=str, default="users")
    args = parser.parse_args()

    main(synthetic_feats_path=args.synthetic_feats_path, real_feats_path=args.real_feats_path, consider_label=args.consider_label,
         method=args.method, unit=args.unit)