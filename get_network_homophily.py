import networkx as nx
import numpy as np
import pandas as pd

src_edges = "downstream_task/dataset/real_social_network.edg"
labels = pd.read_csv("downstream_task/dataset/labeling/by_network/real_users.tsv", sep="\t")

network = nx.read_edgelist(src_edges, create_using=nx.DiGraph, nodetype=int)
labels = labels.drop(columns=[c for c in labels.columns if c not in ["account_id", "label"]])
labels = labels.set_index("account_id")
di = labels.to_dict()["label"]
same = sum(1 for u, v in network.edges() if di[u] == di[v])
print("Computed homophily")
print(same/len(network.edges()))


counts = labels["label"].value_counts()
s = 0
for el in counts:
    s += (el/len(labels))**2
print("Homophily if the edges were randomly created")
print(s)

K = 6  # classes 0..5
M = np.zeros((K, K))
for u, v in network.edges():
    M[di[u], di[v]] += 1

row_sums = M.sum(axis=1, keepdims=True)
M_norm = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums != 0)

np.set_printoptions(precision=3, suppress=True)
print(M_norm)