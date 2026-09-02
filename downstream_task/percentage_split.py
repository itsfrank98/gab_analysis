import pandas as pd
import os
from sklearn.model_selection import train_test_split

BASE_DIR = "percented_train/"
percentages = [10, 20, 30, 40, 50, 60]
folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for percentage in percentages:
    print(f"PERCENTAGE: {percentage}\n")
    for fold in folds:
        train_dir = os.path.join(BASE_DIR, f"{percentage}_perc", "cross_validation", f"fold_{fold}")
        train_df = pd.read_csv(os.path.join(train_dir, "train.tsv"), sep="\t")
        real_df = train_df[train_df["account_id"].astype(str).str.startswith("r")]
        synthetic_df = train_df[train_df["account_id"].astype(str).str.startswith("s")]

        reduced_real, _ = train_test_split(
            real_df,
            train_size=percentage/100,
            stratify=real_df["label"],
            random_state=42  # for reproducibility
        )

        final_df = pd.concat([reduced_real, synthetic_df], ignore_index=True)
        final_df_ids = final_df.drop(columns=[c for c in final_df.columns if c!="account_id"])
        final_df.to_csv(os.path.join(train_dir, "train.tsv"), sep="\t", index=False)
        final_df_ids.to_csv(os.path.join(train_dir, "train_ids.tsv"), sep="\t", index=False)
        print(f"ORIGINAL LENGTH: {len(train_df)}")
        print(f"CURRENT LENGTH: {len(final_df)}")
        print(f"ORIGINAL REAL LENGTH: {len(real_df)}")
        print(f"REDUCED REAL LENGTH: {len(reduced_real)}\n")