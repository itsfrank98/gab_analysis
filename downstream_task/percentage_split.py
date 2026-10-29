import pandas as pd
import os
from sklearn.model_selection import train_test_split

percentages = [1, 2, 4, 8, 16, 32, 64]   #
folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
label_key = "label"     # "exact_level_found"
id_key = "account_id"   # "id"
setting = "r"    # ["r", "rs", "s"]
BASE_DIR = "user_classification"
dst_dir_suffix = "" #_covelli     # the dst dir is join(BASE_DIR, f"percented_train_{setting}" + dst_dir_suffix). The suffix

if BASE_DIR.__contains__("post"):
    synthetic_posts_df_path = "../synthetic_dataset/covelli dataset/synthetic_posts_covelli_labeled.tsv" # "synthetic_posts.tsv"
    synthetic_df = pd.read_csv(synthetic_posts_df_path, sep="\t")
    synthetic_df = synthetic_df.drop(columns=[c for c in synthetic_df.columns if c not in ["account_id", "id", "content", "exact_level_found"]])
elif BASE_DIR.__contains__("user"):
    synthetic_users_df_path = "../synthetic_dataset/covelli dataset/raw/user_profiles_covelli.tsv"
    synthetic_df = pd.read_csv(synthetic_users_df_path, sep="\t")
    synthetic_df = synthetic_df.drop(
        columns=[c for c in synthetic_df.columns if c not in ["account_id", "user_label"]])
    synthetic_df = synthetic_df.rename(columns={"user_label": "label"})

print(synthetic_df.columns)
for percentage in percentages:
    print(f"PERCENTAGE: {percentage}\n")
    for fold in folds:
        dst_dir = os.path.join(BASE_DIR, f"percented_train_{setting}"+dst_dir_suffix, f"{percentage}_perc", "cross_validation",
                               f"fold_{fold}")
        os.makedirs(dst_dir, exist_ok=True)
        if setting == "r":
            train_dir = os.path.join(BASE_DIR, "cross_validation", f"fold_{fold}")
            train_df = pd.read_csv(os.path.join(train_dir, "train.tsv"), sep="\t")
            real_df = train_df[train_df["account_id"].astype(str).str.startswith("r")]
            reduced_real, _ = train_test_split(
                real_df,
                train_size=percentage / 100,
                stratify=real_df[label_key],
                random_state=42  # for reproducibility
            )
            percented_train_df = reduced_real
        elif setting == "rs":
            train_dir = os.path.join(BASE_DIR, "percented_train_r"+dst_dir_suffix, f"{percentage}_perc", "cross_validation", f"fold_{fold}")
            train_df = pd.read_csv(os.path.join(train_dir, "train.tsv"), sep="\t")
            percented_train_df = pd.concat([train_df, synthetic_df], ignore_index=True)
        else:
            raise ValueError("Invalid setting: must either be 'r' or 'rs'")

        test_df = pd.read_csv(os.path.join(train_dir, "test.tsv"), sep="\t")
        percented_train_ids = percented_train_df.drop(columns=[c for c in percented_train_df.columns if c != id_key])
        test_ids = test_df.drop(columns=[c for c in test_df.columns if c != id_key])

        percented_train_df.to_csv(os.path.join(dst_dir, "train.tsv"), sep="\t", index=False)
        percented_train_ids.to_csv(os.path.join(dst_dir, "train_ids.tsv"), sep="\t", index=False)
        test_df.to_csv(os.path.join(dst_dir, "test.tsv"), sep="\t", index=False)
        test_ids.to_csv(os.path.join(dst_dir, "test_ids.tsv"), sep="\t", index=False)

        print(f"ORIGINAL LENGTH: {len(train_df)}")
        print(f"CURRENT LENGTH: {len(percented_train_df)}")