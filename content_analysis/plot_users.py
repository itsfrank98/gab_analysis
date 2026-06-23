"""
Qui fai il plot delle features.
Note importanti:
- Se lo devi lanciare sul server, non lanciarlo da linea di comando perche non vedresti le immagini. Lancialo con tasto destro+run
e cambia i path dai default dell'argparse
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

def show_extremes(values, post_ids, post_texts, is_synthetic, embedding_type, k=10, axis_name="PC1"):
    order = np.argsort(values)
    low_idx = order[:k]
    high_idx = order[-k:]

    print(f"=== {axis_name} LOW extreme ===")
    c = 0
    d = 0
    for i in low_idx:
        try:
            pid = post_ids[i]
            tag = "SYN" if is_synthetic[pid] else "REAL"
            with open(f"{axis_name}_low_extreme_{embedding_type}.txt", "a+") as f:
                f.write(f"[{tag}] ({values[i]:.2f}) {post_texts[pid]}\n")
            #print(f"[{tag}] ({values[i]:.2f}) {post_texts[pid]}")
        except IndexError:
            c += 1

    print(f"\n=== {axis_name} HIGH extreme ===")
    for i in high_idx:
        try:
            pid = post_ids[i]
            tag = "SYN" if is_synthetic[pid] else "REAL"
            with open(f"{axis_name}_high_extreme_{embedding_type}.txt", "a+") as f:
                f.write(f"[{tag}] ({values[i]:.2f}) {post_texts[pid]}\n")
            #print(f"[{tag}] ({values[i]:.2f}) {post_texts[pid]}")
        except IndexError:
            d += 1
    print("C=", c)
    print("D=", d)

def main(synthetic_feats_path, real_feats_path, consider_label, method, real_df_path, synthetic_df_path, embedding_type,
         unit="users", dimensions=2, outlier_analysis=False):
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

        df1 = pd.read_csv(real_df_path)[["id", "content"]]
        df2 = pd.read_csv(synthetic_df_path)
        df2 = df2.rename(columns={"posts": "content"})
        df2 = df2[["id", "content"]]
        df = pd.concat([df1, df2])
        is_synth1 = dict(zip(df1["id"], [False] * len(df1)))
        is_synth2 = dict(zip(df2["id"], [True] * len(df2)))
        is_synth1.update(is_synth2)
        post_texts = dict(zip(df["id"], df["content"]))
        reduced = reduction.fit_transform(matrix)
        pc1 = reduced[:, 0]
        pc2 = reduced[:, 1]
        show_extremes(values=pc1, post_ids=df["id"].tolist(), post_texts=post_texts, is_synthetic=is_synth1, axis_name="PC1", k=100, embedding_type=embedding_type)
        show_extremes(values=pc2, post_ids=df["id"].tolist(), post_texts=post_texts, is_synthetic=is_synth1, axis_name="PC2", k=100, embedding_type=embedding_type)
    else:
        reduction = TSNE(n_components=dimensions)
    reduced = reduction.fit_transform(matrix)

    if dimensions==2:
        plt.figure(figsize=(8, 6))

        if consider_label:
            n_safe = safe_matrix.shape[0]
            n_risky = risky_matrix.shape[0]
            plt.scatter(reduced[:n_safe, 0], reduced[:n_safe, 1], color='green', alpha=.3, s=10, label='safe')
            plt.scatter(reduced[n_safe:n_safe+n_risky, 0], reduced[n_safe:n_safe+n_risky, 1], color='orange', alpha=.1, s=10, label='risky')
            plt.scatter(reduced[n_safe+n_risky:, 0], reduced[n_safe+n_risky:, 1], color='red', alpha=.1, s=10, label='synthetic')
        else:
            #plt.scatter(reduced[:len(real_feats), 0], reduced[:len(real_feats), 1], color='blue', alpha=.6, s=10, label='real')
            #plt.scatter(reduced[len(real_feats):, 0], reduced[len(real_feats):, 1], color='red', alpha=.3, s=10, label='synthetic')
            plt.scatter(reduced[:len(real_feats), 0], reduced[:len(real_feats), 1], facecolors='none', edgecolors='blue',
                        s=10, label='real', alpha=.1)
            plt.scatter(reduced[len(real_feats):, 0], reduced[len(real_feats):, 1], facecolors='none', edgecolors='red',
                        s=10, label='synthetic', alpha=.1)

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
            marker=dict(color='blue', opacity=0.1, size=2),
            name='real',
        ))
        fig.add_trace(go.Scatter3d(
            x=reduced[len(real_feats):, 0],
            y=reduced[len(real_feats):, 1],
            z=reduced[len(real_feats):, 2],
            mode='markers',
            marker=dict(color='red', opacity=0.1, size=2),
            name='synthetic'
        ))


        fig.write_html(f'plot_blue_red_{embedding_type}.html')

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
    parser.add_argument("--synthetic_feats_path", type=str, default="features_merged_sonar.pkl")     #"../synthetic_dataset/bert_features_few_shot_moody_1000.pkl"
    parser.add_argument("--real_feats_path", type=str, default="../dataset/features/sonar_features.pkl")
    parser.add_argument("--synthetic_df_path", type=str, default="merged_id.csv")
    parser.add_argument("--real_df_path", type=str, default="../synthetic_dataset/gab_posts_labeled_qwen.csv")
    parser.add_argument("--consider_label", action="store_true")
    parser.add_argument("--method", type=str, default="pca")
    parser.add_argument("--unit", type=str, default="users")
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--embedding_type", type=str, default="sonar")
    args = parser.parse_args()

    main(synthetic_feats_path=args.synthetic_feats_path, real_feats_path=args.real_feats_path,
         consider_label=args.consider_label, method=args.method, unit=args.unit, dimensions=args.dimensions,
         outlier_analysis=False, real_df_path=args.real_df_path, synthetic_df_path=args.synthetic_df_path,
         embedding_type=args.embedding_type)