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

def manage_labeled(unit, features, df):
    if unit == "posts":
        key = "id"
    elif unit == "users":
        key = "user_id"
    safe = df[df["binary_label"] == 0][key].tolist()
    risky = df[df["binary_label"] == 1][key].tolist()
    safe = set(safe) & set(list(features.keys()))
    risky = set(risky) & set(list(features.keys()))
    safe_matrix = np.array([features[k] for k in safe])
    risky_matrix = np.array([features[k] for k in risky])
    return safe_matrix, risky_matrix


def main(synthetic_feats_path, real_feats_path, method, real_df_path, synthetic_df_path, embedding_type, unit="users",
         dimensions=2, random_sampling=False, labeled_synthetic_posts_path=None, labeled_real_posts_path=None,
         plot_safe=False, plot_risky=False, show_extremes=False):
    with open(synthetic_feats_path, 'rb') as f:
        synthetic_feats = pickle.load(f)
    with open(real_feats_path, 'rb') as f:
        real_feats = pickle.load(f)
    print(f"Real features: {len(real_feats)}, Synthetic features: {len(synthetic_feats)}")
    if type(synthetic_feats) == dict:
        synthetic_matrix = np.array([v for v in synthetic_feats.values()])
    else:
        synthetic_matrix = synthetic_feats

    if labeled_synthetic_posts_path and labeled_real_posts_path:
        labeled_synthetic_posts = pd.read_csv(labeled_synthetic_posts_path)
        labeled_real_posts = pd.read_csv(labeled_real_posts_path)
        safe_synthetic_matrix, risky_synthetic_matrix = manage_labeled(unit=unit, features=synthetic_feats,
                                                                       df=labeled_synthetic_posts)
        safe_real_matrix, risky_real_matrix = manage_labeled(unit=unit, features=real_feats,
                                                                       df=labeled_real_posts)
        matrix = np.vstack((safe_real_matrix, risky_real_matrix))

    else:
        if type(real_feats) == dict:
            real_matrix = np.array([v for v in real_feats.values()])
        else:
            real_matrix = real_feats
        matrix = real_matrix

    if method.lower()=="pca":
        reduction = PCA(n_components=dimensions)
        if show_extremes:
            df1 = pd.read_csv(real_df_path)[["id", "content"]]
            df2 = pd.read_csv(synthetic_df_path)
            df2 = df2.rename(columns={"posts": "content"})
            df2 = df2[["id", "content"]]
            df = pd.concat([df1, df2])
            is_synth1 = dict(zip(df1["id"], [False] * len(df1)))
            is_synth2 = dict(zip(df2["id"], [True] * len(df2)))
            is_synth1.update(is_synth2)
            post_texts = dict(zip(df["id"], df["content"]))

            #pc1 = reduced_real[:, 0]
            #reduced_real[:, 1]
            #show_extremes(values=pc1, post_ids=df["id"].tolist(), post_texts=post_texts, is_synthetic=is_synth1, axis_name="PC1", k=100, embedding_type=embedding_type)
            #show_extremes(values=pc2, post_ids=df["id"].tolist(), post_texts=post_texts, is_synthetic=is_synth1, axis_name="PC2", k=100, embedding_type=embedding_type)
    else:
        reduction = TSNE(n_components=dimensions)
    reduction.fit(matrix)
    # The random sampling is only made on the real posts, which are more than the synthetic ones. We take a sample of the same size of the synthetic dataset
    if random_sampling:
        mat_2_see = matrix[np.random.choice(matrix.shape[0], synthetic_matrix.shape[0], replace=False)]
    else:
        mat_2_see = matrix
    reduced_real = reduction.transform(mat_2_see)
    reduced_synthetic = reduction.transform(synthetic_matrix)

    if dimensions==2:
        plt.figure(figsize=(8, 6))
        if labeled_real_posts_path:
            n_safe_real, n_risky_real = safe_real_matrix.shape[0], risky_real_matrix.shape[0]
            if plot_safe:
                plt.scatter(reduced_real[:n_safe_real, 0], reduced_real[:n_safe_real, 1], color='green', alpha=.1, s=10, label='safe_real')
            if plot_risky:
                plt.scatter(reduced_real[n_safe_real:, 0], reduced_real[n_safe_real:, 1], color='red', alpha=.1, s=10, label='risky_real')
            if labeled_synthetic_posts_path:
                n_safe_synth, n_risky_synth = safe_synthetic_matrix.shape[0], risky_synthetic_matrix.shape[0]
                if plot_safe:
                    plt.scatter(reduced_synthetic[:n_safe_synth, 0], reduced_synthetic[:n_safe_synth, 1], color='blue', alpha=.3, s=10, label='safe_synthetic')
                if plot_risky:
                    plt.scatter(reduced_synthetic[n_safe_synth:, 0], reduced_synthetic[n_safe_synth:, 1], color='orange', alpha=.1, s=10, label='risky_synthetic')
            else:
                plt.scatter(reduced_synthetic[:, 0], reduced_synthetic[:, 1], color='red', alpha=.1, s=10, label='synthetic')
        else:
            plt.scatter(reduced_real[:, 0], reduced_real[:, 1], facecolors='none', edgecolors='blue', s=10, label='real',
                        alpha=.1)
            plt.scatter(reduced_synthetic[:, 0], reduced_synthetic[:, 1], facecolors='none', edgecolors='red',
                        s=10, label='synthetic', alpha=.1)

        plt.legend()
        plt.xlabel(f'{method}1')
        plt.ylabel(f'{method}2')
        plt.title(f'{method} Projection - {unit}')
        plt.tight_layout()
        embedding_type = "goddammit"
        plt.savefig(f'plot_blue_red_{embedding_type}.png')
        plt.show()

    elif dimensions==3:
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=reduced_real[:, 0],
            y=reduced_real[:, 1],
            z=reduced_real[:, 2],
            mode='markers',
            marker=dict(color='blue', opacity=0.1, size=2),
            name='real',
        ))
        fig.add_trace(go.Scatter3d(
            x=reduced_synthetic[:, 0],
            y=reduced_synthetic[:, 1],
            z=reduced_synthetic[:, 2],
            mode='markers',
            marker=dict(color='red', opacity=0.1, size=2),
            name='synthetic'
        ))

        fig.write_html(f'plot_blue_red_{embedding_type}.html')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_feats_path", type=str, default="tgn_embeddings/train_embeddings.pkl")   #"../dataset/features/bert_features/bert_features_real_posts.pkl"
    parser.add_argument("--synthetic_feats_path", type=str, default="tgn_embeddings/inference_embeddings_1000.pkl")     #"../synthetic_dataset/features/bert_features/bert_features_synthetic_moody_posts_10k.pkl"
    parser.add_argument("--synthetic_df_path", type=str, default="merged_id.csv")
    parser.add_argument("--real_df_path", type=str, default="../dataset/gab_posts_labeled_qwen.csv")
    parser.add_argument("--method", type=str, default="pca")
    parser.add_argument("--unit", type=str, default="TGN embeddings")
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--embedding_type", type=str, default="sonar")
    parser.add_argument("--labeled_real_posts_path", type=str, default=None)    # ../dataset/gab_posts_labeled_qwen.csv
    parser.add_argument("--labeled_synthetic_posts_path", type=str, default=None) #../synthetic_dataset/synthetic_posts/synthetic_posts_labeled.csv
    parser.add_argument("--random_sampling", action="store_true", default=False, help="Set it to true if you want to randomly sample from the real dataset")
    parser.add_argument("--plot_safe", type=bool, default=False)
    parser.add_argument("--plot_risky", type=bool, default=False)
    args = parser.parse_args()

    main(synthetic_feats_path=args.synthetic_feats_path, real_feats_path=args.real_feats_path,
         synthetic_df_path=args.synthetic_df_path, real_df_path=args.real_df_path, method=args.method, unit=args.unit,
         dimensions=args.dimensions, embedding_type=args.embedding_type, labeled_real_posts_path=args.labeled_real_posts_path,
         labeled_synthetic_posts_path=args.labeled_synthetic_posts_path, random_sampling=args.random_sampling,
         plot_safe=args.plot_safe, plot_risky=args.plot_risky)