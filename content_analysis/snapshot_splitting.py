import matplotlib.pyplot as plt
import pandas as pd
import os
from synthetic_dataset.network_creation import read_edg_file

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
    jul25 = set(df[(df['timestamp'].dt.year==2025) & (df['timestamp'].dt.month>=7)]["account_id"].tolist())

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
    }

    vals_progressive = {
        "until 2021": until_21_l,
        #"2021": on_21_l+until_21_l,
        "2022": on_22_l+until_21_l,
        "2023": on_23_l+on_22_l+until_21_l,
        "2024": on_24_l+on_23_l+on_22_l+until_21_l,
        "Jan-Jul 25": janjul25_len+on_24_l+on_23_l+on_22_l+until_21_l,
        "Jul 25 -": jul25_len+janjul25_len+on_24_l+on_23_l+on_22_l+until_21_l,
    }

    plot_figure(vals_progressive, xlabel="Snapshot", ylabel="users", title="Growth of the number of users")

posts_src = "../dataset/posts_processed.csv"
network_src = "../dataset/network/social_network.edg"
snapshots = {
    "2016-2021": (2021,),
    "2022": (2022,),
    "2023": (2023,),
    "2024": (2024,),
    "Jan-Jul 25": (2025,7),
    "Jul 25": (2025,7),
}

posts_df = pd.read_csv(posts_src)
network_edges = read_edg_file(network_src)
timestamp_field = "created_at"
account_id_field = "account_id"
posts_df["timestamp"] = pd.to_datetime(posts_df[timestamp_field])
posts_df = posts_df.drop(columns=[c for c in posts_df.columns if c.startswith("Unnamed")])  # Remove useless columns

previous_users = set()
for i, snapshot in enumerate(snapshots):
    os.makedirs(snapshot, exist_ok=True)
    if i == 0:
        posts_of_snapshot = posts_df[posts_df[timestamp_field].dt.year <= snapshots[snapshot][0]]
    elif i == len(snapshots) - 1:
        posts_of_snapshot = posts_df[(posts_df[timestamp_field].dt.year == snapshots[snapshot][0]) &
                                     (posts_df[timestamp_field].dt.month >= snapshots[snapshot][1])]
    else:
        posts_of_snapshot = posts_df[posts_df[timestamp_field].dt.year == snapshots[snapshot][0]]
    users_in_snapshot = set(posts_of_snapshot[account_id_field].tolist())
    users = previous_users.union(users_in_snapshot)

