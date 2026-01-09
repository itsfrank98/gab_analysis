import matplotlib.pyplot as plt
import pandas as pd

def plot_figure(posts_per_month):
    plt.figure(figsize=(14, 6))
    plt.plot(posts_per_month.index, posts_per_month.values, marker='o', markersize=3, linewidth=1)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    plt.title('Distribution of Posts Over Time (Monthly)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

df = pd.read_csv("../dataset/posts_processed.csv")
df["timestamp"] = pd.to_datetime(df["created_at"])
posts_per_month = df.groupby(df['timestamp'].dt.to_period('M')).size()
posts_per_month.index = posts_per_month.index.to_timestamp()

until21 = set(df[df['timestamp'].dt.year < 2021]["account_id"].tolist())
y21 = set(df[df['timestamp'].dt.year == 2021]["account_id"].tolist())
y22 = set(df[df['timestamp'].dt.year == 2022]["account_id"].tolist())
y23 = set(df[df['timestamp'].dt.year == 2023]["account_id"].tolist())
y24 = set(df[df['timestamp'].dt.year==2024]["account_id"].tolist())
janjul25 = set(df[(df['timestamp'].dt.year==2025) & (df['timestamp'].dt.month >=1) & (df['timestamp'].dt.month <7)]["account_id"].tolist())
jul25 = set(df[(df['timestamp'].dt.year==2025) & df['timestamp'].dt.month>=7]["account_id"].tolist())

print("Until 2021: \n")
print(f"Posts: {len(df[df['timestamp'].dt.year < 2021])}")
print(f"Users: {len(until21)}")
union = until21
print("\n")

print("2021: \n")
print(f"Posts: {len(df[df['timestamp'].dt.year==2021])}")
print(f"Users: {len(y21)}")
newunion = union.union(y21)
print(f"New users wrt previous: {len(newunion) - len(union)}")
print("\n")

print("2022: \n")
print(f"Posts: {len(df[df['timestamp'].dt.year==2022])}")
print(f"Users: {len(y22)}")
union = newunion
newunion = union.union(y22)
print(f"New users wrt previous: {len(newunion) - len(union)}")
print("\n")

print("2023: \n")
print(f"Posts: {len(df[df['timestamp'].dt.year==2023])}")
print(f"Users: {len(y23)}")
union = newunion
newunion = union.union(y23)
print(f"New users wrt previous: {len(newunion) - len(union)}")
print("\n")

print("2024: \n")
print(f"Posts: {len(df[df['timestamp'].dt.year==2024])}")
print(f"Users: {len(y24)}")
union = newunion
newunion = union.union(y24)
print(f"New users wrt previous: {len(newunion) - len(union)}")
print("\n")

print("January-July 2025: \n")
print(f"Posts: {len(df[(df['timestamp'].dt.year==2025) & (df['timestamp'].dt.month >=1) & (df['timestamp'].dt.month <7)])}")
print(f"Users: {len(janjul25)}")
union = newunion
newunion = union.union(janjul25)
print(f"New users wrt previous: {len(newunion) - len(union)}")
print("\n")

print("July 2025: \n")
print(f"Posts: {len(df[(df['timestamp'].dt.year == 2025) & (df['timestamp'].dt.month>=7)])}")
print(f"Users: {len(jul25)}")
union = newunion
newunion = union.union(jul25)
original = set(df["account_id"].tolist())

print(f"New users wrt previous: {len(newunion) - len(union)}")
print(len(newunion))
print(original - newunion)
print("\n")


