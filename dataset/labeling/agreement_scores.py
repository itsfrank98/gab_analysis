import pandas as pd
from sklearn.metrics import cohen_kappa_score
from os.path import join

if __name__ == "__main__":
    #df_path = "labeled_real_gab_posts_human_evaluation.xlsx"
    df_paths = ["labeled_real_gab_posts_human_evaluation.tsv",  "labeled_synthetic_gab_posts_human_evaluation.tsv"]
    for e in df_paths:
        print("\n",e)
        df_path = join("agreement_evaluation", e)

        df = pd.read_csv(df_path, sep="\t")

        labels = [0, 1, 2, 3, 4, 5]
        labels_annotator = df["human_label"].tolist()
        labels_llm = df["llm_label"].tolist()
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
