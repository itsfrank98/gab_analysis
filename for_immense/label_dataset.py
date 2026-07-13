import pandas as pd

DF_PATH_1 = "gab_real_posts_labeled_qwen.tsv"
DF_PATH_2 = "synthetic_posts_10k_final.tsv"
POST_SEPARATOR = " "

def load_df(path):
    sep = "\t" if path.endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep)
    df["exact_level_found"] = df["exact_level_found"].astype(int)
    return df


def label_users_multi(dataframe):
    """For each account_id, assign the highest post level found and
    concatenate all of that user's posts."""
    grouped = dataframe.groupby("account_id").agg(
        exact_level_found=("exact_level_found", "max"),
        posts=("content", lambda posts: POST_SEPARATOR.join(posts.astype(str))),
    ).reset_index()
    return grouped[["account_id", "exact_level_found", "posts"]]

def label_users_binary(dataframe):
    dataframe = dataframe.copy()
    dataframe["binary_level"] = (dataframe["exact_level_found"] >= 2).astype(int)
    grouped = dataframe.groupby("account_id").agg(
        label=("binary_level", lambda levels: int(levels.mean() > 0.3)),
        posts=("content", lambda posts: POST_SEPARATOR.join(posts.astype(str))),
    ).reset_index()
    return grouped[["account_id", "label", "posts"]]



if __name__ == "__main__":
    df1 = load_df(DF_PATH_1)
    df2 = load_df(DF_PATH_2)
    label_multi = False
    label_binary = True
    if label_multi:
        df1_labeled_multi = label_users_multi(df1)
        df2_labeled_multi = label_users_multi(df2)
        df1_labeled_multi.to_csv("dataset/real_users_multi.tsv", sep="\t", index=False)
        df2_labeled_multi.to_csv("dataset/synthetic_users_multi.tsv", sep="\t", index=False)
    if label_binary:
        df1_labeled_binary = label_users_binary(df1)
        df2_labeled_binary = label_users_binary(df2)
        print(df1_labeled_binary["label"].value_counts())
        print(df2_labeled_binary["label"].value_counts())
        df1_labeled_binary.to_csv("dataset/real_users_binary.tsv", sep="\t", index=False)
        df2_labeled_binary.to_csv("dataset/synthetic_users_binary.tsv", sep="\t", index=False)