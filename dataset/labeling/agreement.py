import pandas as pd
from sklearn.metrics import cohen_kappa_score
from os.path import join

def unify_dataframes(d1, d_human):
    d1 = d1.drop(columns=[c for c in d1.columns if c not in ["id", "predicted_by_xlmt"]])
    d_human = d_human.drop(columns=[c for c in d_human.columns if c not in ["id", "Radical level"]])
    merged = pd.merge(d1, d_human, on="id")

    merged = merged.rename(columns={"predicted_by_xlmt": "exact_level_found"})
    return merged

if __name__ == "__main__":
    #df_path = "labeled_real_gab_posts_human_evaluation.xlsx"
    df_paths = ["counter_synthetic_posts.tsv", "counter_real_posts.tsv", "qwen35_synthetic_posts", "qwen35_real_posts.csv",
                "qwen27_synthetic_posts.tsv",  "labeled_real_gab_posts_human_evaluation.xlsx"]
    for e in df_paths:
        print("\n",e)
        df_path = join("agreement_evaluation", e)

        #dpred = pd.read_csv("real_posts_predicted.tsv", sep="\t")

        if df_path.endswith(".xlsx"):
            df = pd.read_excel(df_path)
        elif df_path.endswith(".csv"):
            df = pd.read_csv(df_path)
        elif df_path.endswith(".tsv"):
            df = pd.read_csv(df_path, sep="\t")
        if "predicted_by_xlmt" in df.columns:
            k = "predicted_by_xlmt"
        else:
            k = "exact_level_found"
        labels = df[k].unique().tolist()
        labels_annotator = df["radical level"].tolist()
        labels_llm = df[k].tolist()
        kscore = cohen_kappa_score(labels_llm, labels_annotator, labels=labels)
        print("Cohen's kappa score: ", kscore)

        # Squared error
        se = 0
        be = 0
        labels_annotator_binary = []
        labels_llm_binary = []
        for a, b in zip(labels_annotator, labels_llm):
            se += (a - b) ** 2
            if (a>2) != (b>2):
                be += 1
            labels_annotator_binary.append(0 if a < 2 else 1)
            labels_llm_binary.append(0 if b < 2 else 1)
        mse = se/len(labels_annotator)
        kscore_binary = cohen_kappa_score(labels_llm_binary, labels_annotator_binary, labels=[0,1])
        print("Root mean square error: ", mse**(.5))
        print("Mean binary error: ", be/len(labels_annotator))
        print("Binary Kappa score: ", kscore_binary)
