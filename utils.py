import pickle

import pandas as pd


def save_to_pickle(name, c):
    with open(name, 'wb') as f:
        pickle.dump(c, f)


def load_from_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def create_dataframes(df, afl, model_name, dim):
    for k in list(afl.keys()):
        if k.__contains__("duplicate"):
            afl.pop(k)
    far_right = [int(k) for k in list(afl.keys()) if afl[k] == "far right"]
    far_left = [int(k) for k in list(afl.keys()) if afl[k] == "far left"]
    left = [int(k) for k in list(afl.keys()) if afl[k] == "left"]
    right = [int(k) for k in list(afl.keys()) if afl[k] == "right"]
    apolitical = [int(k) for k in list(afl.keys()) if afl[k] == "non political"]
    center = [int(k) for k in list(afl.keys()) if afl[k] == "center"]

    df = df.drop(columns=[k for k in df.columns if k not in ["account_id", "content"]])
    df_left = df[df.account_id.isin(left)]
    df_right = df[df.account_id.isin(right)]
    df_far_right = df[df.account_id.isin(far_right)]
    df_far_left = df[df.account_id.isin(far_left)]
    df_center = df[df.account_id.isin(center)]
    df_apolitical = df[df.account_id.isin(apolitical)]


    with pd.ExcelWriter(f"stance/{model_name}_{dim}.xlsx") as writer:
        df_right.to_excel(writer, sheet_name="right")
        df_left.to_excel(writer, sheet_name="left")
        df_far_left.to_excel(writer, sheet_name="far_left")
        df_far_right.to_excel(writer, sheet_name="far_right")
        df_center.to_excel(writer, sheet_name="center")
        df_apolitical.to_excel(writer, sheet_name="apolitical")