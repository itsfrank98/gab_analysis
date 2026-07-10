import argparse
import numpy as np
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import os

def cluster(embeddings_dict: dict, n_clusters: int = 50, random_state: int = 42):
    """
    Sample posts using a mix of cluster-stratified and peripheral sampling.

    Args:
        embeddings_dict: {post_id: np.array of shape (768,)}
        n_clusters: number of KMeans clusters
    Returns:
        list of sampled post IDs
    """
    post_ids = list(embeddings_dict.keys())
    embeddings = np.stack([embeddings_dict[pid].numpy() for pid in post_ids])

    # Normalize before clustering (cosine-like behavior)
    embeddings_norm = normalize(embeddings)

    # --- Clustering ---
    print("Clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(embeddings_norm)
    print("Clustering done")

    # Distance of each point to its own cluster centroid
    centroids = kmeans.cluster_centers_
    distances_to_centroid = np.linalg.norm(
        embeddings_norm - centroids[labels], axis=1
    )

    return distances_to_centroid, labels, post_ids


def sample(distances_to_centroid, labels, post_ids, n_samples=10, peripheral_ratio=0.4, random_state=42):
    """
    n_samples: total number of post IDs to return
    peripheral_ratio: fraction of n_samples to draw from peripheral posts,
                      sampled from the periphery of the chosen cluster
    """
    rng = np.random.default_rng(random_state)
    # --- Pick a random cluster ---
    cluster_ids = np.unique(labels)
    chosen_cluster = rng.choice(cluster_ids)
    cluster_indices = np.where(labels == chosen_cluster)[0]

    # Distances to centroid, restricted to the chosen cluster
    cluster_distances = distances_to_centroid[cluster_indices]

    # --- Split the cluster into peripheral and core ---
    peripheral_threshold = np.percentile(cluster_distances, 100 * (1 - peripheral_ratio))
    is_peripheral = cluster_distances >= peripheral_threshold

    peripheral_indices = cluster_indices[is_peripheral]
    core_indices = cluster_indices[~is_peripheral]

    # --- Sample from each pool ---
    n_peripheral = int(n_samples * peripheral_ratio)
    n_core = n_samples - n_peripheral

    sampled_peripheral = rng.choice(
        peripheral_indices, size=min(n_peripheral, len(peripheral_indices)), replace=False
    )
    sampled_core = rng.choice(
        core_indices, size=min(n_core, len(core_indices)), replace=False
    )

    all_sampled_indices = np.concatenate([sampled_peripheral, sampled_core])
    return [post_ids[i] for i in all_sampled_indices]

def main(path_to_dict, path_to_labeled_posts, desired_level, n_clusters, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    with open(path_to_dict, 'rb') as f:
        emb_dict = pickle.load(f)
    posts_labeled = pd.read_csv(path_to_labeled_posts)
    posts_desired = posts_labeled[posts_labeled["exact_level_found"] == desired_level]["id"].tolist()
    desired_embeddings = dict(filter(lambda item: item[0] in posts_desired, emb_dict.items()))
    dst, labels, ids = cluster(desired_embeddings, n_clusters)
    with open(os.path.join(dst_dir, f"distances_level_{desired_level}.pkl"), "wb") as f:
        pickle.dump(dst, f)
    with open(os.path.join(dst_dir, f"labels_level_{desired_level}.pkl"), "wb") as f:
        pickle.dump(labels, f)
    with open(os.path.join(dst_dir, f"postids_level_{desired_level}.pkl"), "wb") as f:
        pickle.dump(ids, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_emb_dict", type=str, default="../dataset/bert_features/bert_features_real_posts.pkl")
    parser.add_argument("--path_to_labeled_posts", type=str, default="../dataset/gab_posts_labeled_qwen.csv")
    parser.add_argument("--desired_level", type=int, default=0)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--dst_dir", type=str)

    args = parser.parse_args()
    main(path_to_dict=args.path_to_emb_dict, path_to_labeled_posts=args.path_to_labeled_posts,
         desired_level=args.desired_level, n_clusters=args.n_clusters, dst_dir=args.dst_dir)

