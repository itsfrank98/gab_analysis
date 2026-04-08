import os
import json
import pandas as pd
import re


def clean_dirty_jsons(src, profiles_df):
    ld = []
    for f in os.listdir(src):
        user_row = profiles_df[profiles_df["username"] == f.split(".")[0]]
        d = {
            "account_id": user_row["account_id"],
            "username": f.split(".")[0],
            "user_bio": user_row["user_bio"],
            "nationality": 'american',
            "state_of_origin": user_row["state_of_origin"],
            "gender": user_row["gender"],
            "ethnicity": user_row["ethnicity"],
            "religion": user_row["religion"],
            "political_leaning": user_row["political_leaning"].replace("_", "-"),
            "interests": user_row["interests"],
            "age_interval": user_row["age_interval"],
            "profession": user_row["profession"],
        }
        with open(os.path.join(src, f), "r", encoding="utf-8") as f:
            content = str(f.read())
        match = re.search(r'\{.*}', content, re.DOTALL)
        if not match:
            print("No JSON object found")
        d["posts"] = json.loads(match.group())["response"]
        ld.append(d)
        print(f)
    return ld

profiles_df = pd.read_csv("synthetic_user_profiles.csv")
ld = clean_dirty_jsons(src="tokeep", profiles_df=profiles_df)