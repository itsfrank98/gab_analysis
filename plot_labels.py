import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

order = [0, 1, 2, 3, 4, 5]
real_df = pd.read_csv("dataset/labeling/gab_posts_labeled_qwen.csv")
synthetic_df = pd.read_csv("synthetic_dataset/synthetic_posts/synthetic_posts_labeled.tsv", sep="\t", quoting=3, escapechar="\\")

real_df["exact_level_found"] = real_df["exact_level_found"].astype(int)
synthetic_df["exact_level_found"] = synthetic_df["exact_level_found"].astype(int)

realdist = real_df.value_counts(subset="exact_level_found")*100/len(real_df)
synthdist = synthetic_df.value_counts(subset="exact_level_found")*100/len(synthetic_df)
print(realdist)
print(synthdist)
realdist = realdist.reindex(order)
synthdist = synthdist.reindex(order)

labels = sorted(set(realdist.index) | set(synthdist.index))
cmap = cm.get_cmap('OrRd', len(labels))
color_map = {label: cmap(i) for i, label in enumerate(labels)}

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
realdist.plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=90, colors=[color_map[l] for l in realdist.index], title="Label distribution real posts")
synthdist.plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90, colors=[color_map[l] for l in synthdist.index], title="Label distribution synthetic posts")
plt.ylabel('')
plt.show()
