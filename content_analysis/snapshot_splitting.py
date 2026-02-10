import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from synthetic_dataset.network_creation import read_edg_file, write_edg_file

def plot_figure(posts_per_month, xlabel, ylabel, title):
    plt.figure(figsize=(14, 6))
    if type(posts_per_month) == pd.DataFrame:
        plt.plot(posts_per_month.index, posts_per_month.values, marker='o', markersize=3, linewidth=1)
    else:
        plt.plot(posts_per_month.keys(), posts_per_month.values(), marker='o', markersize=3, linewidth=1)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def compute_statistics(csv_path="../dataset/posts_processed.csv"):
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["created_at"])
    posts_per_month = df.groupby(df['timestamp'].dt.to_period('M')).size()
    posts_per_month.index = posts_per_month.index.to_timestamp()

    until21 = set(df[df['timestamp'].dt.year <= 2021]["account_id"].tolist())
    #y21 = set(df[df['timestamp'].dt.year == 2021]["account_id"].tolist())
    y22 = set(df[df['timestamp'].dt.year == 2022]["account_id"].tolist())
    y23 = set(df[df['timestamp'].dt.year == 2023]["account_id"].tolist())
    y24 = set(df[df['timestamp'].dt.year==2024]["account_id"].tolist())
    janjul25 = set(df[(df['timestamp'].dt.year==2025) & (df['timestamp'].dt.month >=1) & (df['timestamp'].dt.month <7)]["account_id"].tolist())
    jul25 = set(df[(df['timestamp'].dt.year==2025) & (df['timestamp'].dt.month>=7) & (df['timestamp'].dt.day<26)]["account_id"].tolist())
    valid = set(df[(df['timestamp'].dt.year==2025) & (df['timestamp'].dt.month==7) & (df['timestamp'].dt.day>=26) & (df['timestamp'].dt.day<29)]["account_id"].tolist())
    test = set(df[(df['timestamp'].dt.year==2025) & (df['timestamp'].dt.month==7) & (df['timestamp'].dt.day>29)]["account_id"].tolist())

    print("Until 2021: \n")
    print(f"Posts: {len(df[df['timestamp'].dt.year <= 2021])}")
    print(f"Users: {len(until21)}")
    union = until21
    until_21_l = len(union)
    print("\n")

    """print("2021: \n")
    print(f"Posts: {len(df[df['timestamp'].dt.year==2021])}")
    print(f"Users: {len(y21)}")
    newunion = union.union(y21)
    on_21_l = len(newunion) - len(union)
    print(f"New users wrt previous: {len(newunion) - len(union)}")
    print("\n")"""

    print("2022: \n")
    print(f"Posts: {len(df[df['timestamp'].dt.year==2022])}")
    print(f"Users: {len(y22)}")
    newunion = union.union(y22)

    on_22_l = len(newunion) - len(union)
    print(f"New users wrt previous: {len(newunion) - len(union)}")
    print("\n")

    print("2023: \n")
    print(f"Posts: {len(df[df['timestamp'].dt.year==2023])}")
    print(f"Users: {len(y23)}")
    union = newunion
    newunion = union.union(y23)
    on_23_l = len(newunion) - len(union)
    print(f"New users wrt previous: {len(newunion) - len(union)}")
    print("\n")

    print("2024: \n")
    print(f"Posts: {len(df[df['timestamp'].dt.year==2024])}")
    print(f"Users: {len(y24)}")
    union = newunion
    newunion = union.union(y24)
    on_24_l = len(newunion) - len(union)
    print(f"New users wrt previous: {len(newunion) - len(union)}")
    print("\n")

    print("January-July 2025: \n")
    print(f"Posts: {len(df[(df['timestamp'].dt.year==2025) & (df['timestamp'].dt.month >=1) & (df['timestamp'].dt.month < 7)])}")
    print(f"Users: {len(janjul25)}")
    union = newunion
    newunion = union.union(janjul25)
    janjul25_len = len(newunion) - len(union)
    print(f"New users wrt previous: {len(newunion) - len(union)}")
    print("\n")

    print("July 2025: \n")
    print(f"Posts: {len(df[(df['timestamp'].dt.year == 2025) & (df['timestamp'].dt.month >= 7)])}")
    print(f"Users: {len(jul25)}")
    union = newunion
    newunion = union.union(jul25)
    jul25_len = len(newunion) - len(union)

    print("Valid: \n")
    print(f"Users: {len(valid)}")
    union = newunion
    newunion = union.union(valid)
    valid_len = len(newunion) - len(union)

    print("Test: \n")
    print(f"Users: {len(test)}")
    union = newunion
    newunion = union.union(test)
    test_len = len(newunion) - len(union)

    print(f"New users wrt previous: {len(newunion) - len(union)}")
    print(len(newunion))
    print("\n")

    vals = {
        "until 2021": until_21_l,
        #"2021": on_21_l,
        "2022": on_22_l,
        "2023": on_23_l,
        "2024": on_24_l,
        "Jan-Jul 25": janjul25_len,
        "Jul 25 -": jul25_len,
        "Valid": valid_len,
        "Test": test_len,
    }

    vals_progressive = {
        "until 2021": until_21_l,
        "2022": on_22_l+until_21_l,
        "2023": on_23_l+on_22_l+until_21_l,
        "2024": on_24_l+on_23_l+on_22_l+until_21_l,
        "Jan-Jul 25": janjul25_len+on_24_l+on_23_l+on_22_l+until_21_l,
        "Jul 25 -": jul25_len+janjul25_len+on_24_l+on_23_l+on_22_l+until_21_l,
        "Valid": valid_len + jul25_len + janjul25_len + on_24_l + on_23_l + on_22_l + until_21_l,
        "Test": test_len + valid_len + jul25_len + janjul25_len + on_24_l + on_23_l + on_22_l + until_21_l,
    }

    plot_figure(vals_progressive, xlabel="Snapshot", ylabel="users", title="Growth of the number of users")

posts_src = "../dataset/posts_processed.csv"
network_src = "../dataset/social_network.edg"
dst = "../dataset/snapshots_new1"
os.makedirs(dst, exist_ok=True)
snapshots = {
    "2016-2021": (2021, 1, 12),
    "2022": (2022, 1, 12),
    "2023": (2023, 1, 12),
    "2024": (2024, 1, 12),
    "2025_01-03": (2025, 1, 3),
    "2025_04-05": (2025, 4, 5),
    "2025_06": (2025, 6, 6),
    "2025_07_01_28": (2025, 7, 1, 28),
    "2025_07_29_29_val": (2025, 7, 29, 29),
    "2025_07_30_31_test": (2025, 7, 30, 31),
}

posts_df = pd.read_csv(posts_src)
network_edges = read_edg_file(network_src)
timestamp_field = "created_at"
account_id_field = "account_id"
posts_df["timestamp"] = pd.to_datetime(posts_df[timestamp_field])
posts_df = posts_df.drop(columns=[c for c in posts_df.columns if c.startswith("Unnamed")])  # Remove useless columns

previous_users = set()
previous_snapshot_posts = pd.DataFrame()
for i, snapshot in tqdm(enumerate(snapshots)):
    if len(snapshots[snapshot]) == 3:
        posts_of_snapshot = posts_df[(posts_df["timestamp"].dt.year == snapshots[snapshot][0]) &
                                     (posts_df["timestamp"].dt.month >= snapshots[snapshot][1]) &
                                     (posts_df["timestamp"].dt.month <= snapshots[snapshot][2])]
    elif len(snapshots[snapshot]) == 4:
        posts_of_snapshot = posts_df[(posts_df["timestamp"].dt.year == snapshots[snapshot][0]) &
                                     (posts_df["timestamp"].dt.month == snapshots[snapshot][1]) &
                                     (posts_df["timestamp"].dt.day >= snapshots[snapshot][2]) &
                                     (posts_df["timestamp"].dt.day <= snapshots[snapshot][3])]

    os.makedirs(os.path.join(dst, snapshot), exist_ok=True)

    posts_incremental = pd.concat([previous_snapshot_posts, posts_of_snapshot])
    previous_snapshot_posts = posts_incremental

    users_in_snapshot = set(posts_of_snapshot[account_id_field].tolist())
    users = previous_users.union(users_in_snapshot)
    previous_users = users

    edgelist_current_snapshot = []
    for ed in network_edges:
        if int(ed[0]) in users and int(ed[1]) in users:
            edgelist_current_snapshot.append((ed[0], ed[1]))
    posts_incremental.to_csv(os.path.join(dst, snapshot, "posts_incremental.csv"), index=False)
    posts_of_snapshot.to_csv(os.path.join(dst, snapshot, "posts_current_snapshot.csv"), index=False)
    write_edg_file(edgelist_current_snapshot, os.path.join(dst, snapshot, "social_network.edg"))

