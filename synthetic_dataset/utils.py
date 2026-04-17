import os
import json
import pandas as pd
import re


def clean_dirty_jsons(src, profiles_df):
    ld = []
    for user in os.listdir(src):
        user_row = profiles_df[profiles_df["account_id"] == user.split(".")[0]]
        if len(user_row) > 1:
            os.remove(os.path.join(src, user))
            continue
        d = {
            "account_id": user_row["account_id"].values[0],
            "username": user_row["username"].values[0],
            "user_bio": user_row["user_bio"].values[0],
            "nationality": 'american',
            "state_of_origin": user_row["state_of_origin"].values[0],
            "gender": user_row["gender"].values[0],
            "ethnicity": user_row["ethnicity"].values[0],
            "religion": user_row["religion"].values[0],
            "political_leaning": user_row["political_leaning"].values[0].replace("_", "-"),
            "interests": user_row["interests"].values[0],
            "age_interval": user_row["age_interval"].values[0],
            "profession": user_row["profession"].values[0],
        }
        with open(os.path.join(src, user), "r", encoding="utf-8") as f:
            content = str(f.read())
        match = re.search(r'\{.*}', content, re.DOTALL)
        if not match:
            print("No JSON object found")
        try:
            d["posts"] = json.loads(match.group())["response"]
            ld.append(d)
            print(user, " GOOD!")
        except Exception as e:
            print(f"ERROR: file {user}")
            os.remove(os.path.join(src, user))
    return ld

def scan_files(src):
    for fn in os.listdir(src):
        with open(os.path.join(src, fn), "r", encoding="utf-8") as f:
            content = str(f.read())
        if "{" not in content or "}" not in content:
            os.remove(os.path.join(src, fn))
    print(len(os.listdir(src)))

src = "users"
scan_files(src)
profiles_df = pd.read_csv("synthetic_user_profiles.csv")
ld = clean_dirty_jsons(src=src, profiles_df=profiles_df)
df = pd.DataFrame(ld)
df = df.explode("posts")

df.to_csv("synthetic_posts2.csv", index=False)
