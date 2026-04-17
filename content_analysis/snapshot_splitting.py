import matplotlib.pyplot as plt
import pandas as pd
import os
import random
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

SHUFFLE = False
posts_src = "../dataset/comments_only_posting_users.csv"  # "../dataset/posts_processed.csv"
network_src = None # "../dataset/social_network.edg"

dst = "../dataset/snapshots"
csv_name = "comments_current_snapshot.csv"       # name to use when saving the csv in the snapshots  "posts_current_snapshot.csv"
incremental_csv_name = "comments_incremental.csv"  # posts_incremental.csv
timestamp_field = "created_at"
account_id_field = "account_id"

os.makedirs(dst, exist_ok=True)
snapshots = {
    "2016-2021": (2010, 1, 1, 2021, 12, 31),
    "2022": (2022, 1, 1, 2022, 12, 31),
    "2023": (2023, 1, 1, 2023, 12, 31),
    "2024": (2024, 1, 1, 2024, 12, 31),
    "2025_01-03": (2025, 1, 1, 2025, 3, 31),
    "2025_04-05": (2025, 4, 1, 2025, 5, 31),
    "2025_06": (2025, 6, 1, 2025, 6, 30),
    "2025_07_01_20_val": (2025, 7, 1, 2025, 7, 20),
    "2025_07_21_28_test1": (2025, 7, 21, 2025, 7, 28),
    "2025_07_29_31_test2": (2025, 7, 29, 2025, 12, 31),
}
if SHUFFLE:
    keys_to_shuffle = ["2016-2021", "2022", "2023", "2024", "2025_01-03", "2025_04-05", "2025_06"]
    random.shuffle(keys_to_shuffle)
    snapshots = {
        "0_"+keys_to_shuffle[0]: snapshots[keys_to_shuffle[0]],
        "1_"+keys_to_shuffle[1]: snapshots[keys_to_shuffle[1]],
        "2_"+keys_to_shuffle[2]: snapshots[keys_to_shuffle[2]],
        "3_"+keys_to_shuffle[3]: snapshots[keys_to_shuffle[3]],
        "4_"+keys_to_shuffle[4]: snapshots[keys_to_shuffle[4]],
        "5_"+keys_to_shuffle[5]: snapshots[keys_to_shuffle[5]],
        "6_"+keys_to_shuffle[6]: snapshots[keys_to_shuffle[6]],
        "7_2025_07_01_20_val": snapshots["2025_07_01_20_val"],
        "8_2025_07_21_28_test1": snapshots["2025_07_21_28_test1"],
        "9_2025_07_29_31_test2": snapshots["2025_07_29_31_test2"],
    }
print(snapshots)
posts_df = pd.read_csv(posts_src)


posts_df["timestamp"] = pd.to_datetime(posts_df[timestamp_field])
posts_df = posts_df.drop(columns=[c for c in posts_df.columns if c.startswith("Unnamed")])  # Remove useless columns

previous_users = set()
previous_snapshot_posts = pd.DataFrame()
if network_src:
    network_edges = set(read_edg_file(network_src, type_pairs="tuple"))
for snapshot_name, (start_year, start_month, start_day, end_year, end_month, end_day) in snapshots.items():
    start_date = pd.Timestamp(start_year, start_month, start_day, tz="UTC")
    end_date = pd.Timestamp(end_year, end_month, end_day, tz="UTC")
    posts_of_snapshot = posts_df[(posts_df["timestamp"] >= start_date) & (posts_df["timestamp"] <= end_date)]

    os.makedirs(os.path.join(dst, snapshot_name), exist_ok=True)

    posts_incremental = pd.concat([previous_snapshot_posts, posts_of_snapshot])
    previous_snapshot_posts = posts_incremental

    users_in_snapshot = set(posts_of_snapshot[account_id_field].tolist())
    users = previous_users.union(users_in_snapshot)
    previous_users = users

    posts_of_snapshot.to_csv(os.path.join(dst, snapshot_name, csv_name), index=False)
    posts_incremental.to_csv(os.path.join(dst, snapshot_name, incremental_csv_name), index=False)
    print(len(posts_incremental))

    if network_src:
        edgelist_current_snapshot = []
        for ed in network_edges:
            if int(ed[0]) in users and int(ed[1]) in users:
                edgelist_current_snapshot.append((ed[0], ed[1]))
        write_edg_file(edgelist_current_snapshot, os.path.join(dst, snapshot_name, "social_network.edg"))

