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
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def main(synthetic_feats_path, real_feats_path, consider_label, method, unit="users", dimensions=2, outlier_analysis=False):
    with open(synthetic_feats_path, 'rb') as f:
        synthetic_feats = pickle.load(f)
    with open(real_feats_path, 'rb') as f:
        real_feats = pickle.load(f)
    print(len(real_feats), len(synthetic_feats))

    synthetic_matrix = np.array([v for v in synthetic_feats.values()])

    if consider_label:
        labeled_users = pd.read_csv("../dataset/vicuna_classification_results_digennaro.csv")
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

    if method.lower()=="pca":
        reduction = PCA(n_components=dimensions)
    else:
        reduction = TSNE(n_components=dimensions)
    reduced = reduction.fit_transform(matrix)
    dimensions =3
    if dimensions==2:
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
        plt.xlabel(f'{method}1')
        plt.ylabel(f'{method}2')
        plt.title(f'{method} Projection - {unit}')
        plt.tight_layout()
        plt.show()

    elif dimensions==3:
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=reduced[:len(real_feats), 0],
            y=reduced[:len(real_feats), 1],
            z=reduced[:len(real_feats), 2],
            mode='markers',
            marker=dict(color='blue', opacity=0.3, size=3),
            name='real'
        ))

        fig.add_trace(go.Scatter3d(
            x=reduced[len(real_feats):, 0],
            y=reduced[len(real_feats):, 1],
            z=reduced[len(real_feats):, 2],
            mode='markers',
            marker=dict(color='red', opacity=0.3, size=3),
            name='synthetic'
        ))
        fig.write_html('plot.html')

    if outlier_analysis:
        X_blue = matrix[:len(real_feats)]
        X_red = matrix[len(real_feats):]

        # fit GMM on blue data
        gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=0)
        gmm.fit(X_blue)

        # compute log-likelihoods
        loglik_blue = gmm.score_samples(X_blue)
        loglik_red = gmm.score_samples(X_red)

        # define threshold (e.g., 5th percentile of blue likelihood)
        threshold = np.percentile(loglik_blue, 5)

        # fraction of red outside (low likelihood under blue)
        outside_fraction = np.mean(loglik_red < threshold)

        print(f"Fraction of red outside: {outside_fraction:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_feats_path", type=str, default="../dataset/features_bert/bert_features_few_shot_enriched.pkl")
    parser.add_argument("--real_feats_path", type=str, default="../dataset/features_bert/bert_features_real_posts.pkl")
    parser.add_argument("--consider_label", action="store_true")
    parser.add_argument("--method", type=str, default="pca")
    parser.add_argument("--unit", type=str, default="posts")
    parser.add_argument("--dimensions", type=int, default=3)
    args = parser.parse_args()

    main(synthetic_feats_path=args.synthetic_feats_path, real_feats_path=args.real_feats_path, consider_label=args.consider_label,
         method=args.method, unit=args.unit, dimensions=args.dimensions, outlier_analysis=False)