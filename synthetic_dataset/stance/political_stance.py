import json
import os
import re
import time
import matplotlib.pyplot as plt
import pandas as pd
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from openai import OpenAI, RateLimitError
from utils import create_dataframes


def compute_stance(client, df, affiliations_fname, model, affiliations=None):
    if affiliations:
        df = df[~df.account_id.astype(str).isin(list(affiliations.keys()))]
    else:
        affiliations = {}

    i = 0
    step = 15
    df = df.drop(columns=[c for c in df.columns if c not in ["account_id", "posts_count", "content"]])
    df = df.reset_index()
    while i < len(df):
        print(i)
        sub_df = df.loc[i:i + step]
        dictionary = {r['account_id']: r['content'] + "\n" for _, r in sub_df.iterrows()}
        content = ("Instruction: I will now give you a dictionary. The keys represent IDs, and the values are "
                   "texts associated to each ID. I need you to look at each text and tell if its political "
                   "leaning is 'far left', 'left', 'center', 'right', 'far right'. If you can't decide a label, mark it"
                   " as 'unknown'. If the content is not about politics, mark it as 'non political'.\n"
                   "OUTPUT INSTRUCTION: Return the answer in form of a json dictionary where the keys are the same"
                   "of the dictionary I provide, and the values are the labels associated to the text. "
                   "Don't write anything else."
                   f"Input dictionary: {dictionary}")
        try:
            if model == "local-model":
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}]
                )
                answer = response.choices[0].message.content
            else:
                chat_response = client.chat.complete(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        },
                    ]
                )
                answer = chat_response.choices[0].message.content
            answer = re.sub(r'```(?:json)?\s*|\s*```', '', answer).strip()
            try:
                answer_dict = json.loads(answer)
                affiliations.update(answer_dict)
                with open(affiliations_fname, 'w') as f:
                    json.dump(obj=affiliations, fp=f, indent=2)
            except json.JSONDecodeError as e:
                print(f"Error!\n {answer}")
            i += step
        except SDKError:
            print("sleeping for 5 seconds then trying again")
            time.sleep(5)
        except RateLimitError:
            print("sleeping for 5 seconds then trying again")
            time.sleep(5)
    return affiliations


def plot_stance(affiliations):
    c = list(affiliations.values())
    dc = {v: c.count(v) for v in ["far left", "left", "center", "non political", "right", "far right", "unknown"]}     # list(set(c))
    plt.bar(list(dc.keys()), list(dc.values()))
    plt.show()


if __name__ == "__main__":
    api_key_mistral = "dW87PLUULfArg0ys0XevF6HyOJUeJJHP"
    model = "mistral-large-latest"  #local-model
    dim = 4000
    sampled_fname = f"sampled_for_stance_{dim}.csv"
    if model == "local-model":
        base_url = "http://127.0.0.1:1234/v1"
        client = OpenAI(base_url=base_url, api_key="foo")
    else:
        client = Mistral(api_key=api_key_mistral)

    affiliations_fname = f"affiliations_{dim}_{model}.json"     #_newlabel
    if not os.path.exists(sampled_fname):
        df = pd.read_csv("../../dataset/posts_processed_stopwords_without_sampled.csv")  # posts_processed_stopwords.csv
        df = df[df.content.str.split().str.len().between(40, 1000)]
        sampled = df.sample(dim)
        sampled = sampled.reset_index()
        sampled.to_csv(sampled_fname)
    sampled = pd.read_csv(sampled_fname)
    affiliations = None
    if os.path.exists(affiliations_fname):
        with open(affiliations_fname, 'r') as f:
            affiliations = json.load(f)
    affiliations = compute_stance(client=client, df=sampled, affiliations_fname=affiliations_fname, model=model, affiliations=affiliations)

    plot_stance(affiliations)
    create_dataframes(df=sampled, afl=affiliations, model_name=model, dim=dim, dst_dir="dataframes")
