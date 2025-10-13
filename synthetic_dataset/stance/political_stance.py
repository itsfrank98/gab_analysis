import json
import os
import re
import time
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from openai import OpenAI, RateLimitError
from stance_utils import *

API_KEY_MISTRAL = "DodgOOtH2qzN13X0xowpPQZqTE1glFI2"

def compute_stance(client, df, affiliations_fname, model, political_leanings, affiliations=None):
    if affiliations:
        df = df[~df.account_id.astype(str).isin(list(affiliations.keys()))]
    else:
        affiliations = {}

    i = 0
    step = 10
    #df = df.drop(columns=[c for c in df.columns if c not in ["account_id", "posts_count", "content"]])
    df = df.reset_index()
    political_leaning_str = ", ".join(political_leanings)
    while i < len(df):
        print(i)
        sub_df = df.loc[i:i + step-1]
        dictionary = {r['account_id']: r['content'] for _, r in sub_df.iterrows()}
        content = ("Instruction: I will now give you a dictionary. The keys represent IDs, and the values are "
                   "texts associated to each ID. I need you to look at each text and tell if its political "
                   f"leaning is {political_leaning_str}. Classify the texts considering the american political point"
                   "of view. If you can't decide a label, mark it as 'unknown'.If the content is not about politics, "
                   "mark it as 'non_political'. Assign each text exactly one leaning.\n"
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


def plot_stance(affiliations, political_leanings, model):
    c = list(affiliations.values())
    dc = {v: c.count(v) for v in political_leanings}     # list(set(c))
    plt.bar(list(dc.keys()), list(dc.values()))
    plt.xticks(rotation=45, ha='right')
    plt.show()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/political_stance_{}.svg'.format(model))


def main(model, dim, affiliations_fname, sampled_fname, dataframe_dst_dir, compute_stance_flag=False):
    political_leanings = ["far-left", "left", "center", "right", "far-right"]
    # political_leanings = ['panafricanist', 'unknown', 'conservative', 'centrist', 'member of ISIS', 'far-right', 'republican', 'liberal', 'far-left']

    if model == "local-model":
        base_url = "http://127.0.0.1:1234/v1"
        client = OpenAI(base_url=base_url, api_key="foo")
    else:
        client = Mistral(api_key=API_KEY_MISTRAL)

    affiliations = None

    if not os.path.exists(sampled_fname):
        df = pd.read_csv("../../dataset/posts_processed_stopwords_without_sampled.csv")  # posts_processed_stopwords.csv
        df = df[df.content.str.split().str.len().between(40, 1000)]
        sampled = df.sample(dim)
        sampled = sampled.reset_index()
        sampled.to_csv(sampled_fname)
    sampled = pd.read_csv(sampled_fname)

    if os.path.exists(affiliations_fname):
        with open(affiliations_fname, 'r') as f:
            affiliations = json.load(f)
    if compute_stance_flag:
        affiliations = compute_stance(client=client, df=sampled, affiliations_fname=affiliations_fname, model=model,
                                      affiliations=affiliations, political_leanings=political_leanings)

    plot_stance(affiliations, political_leanings + ["unknown", "non-political"], model)
    #create_dataframes(df=sampled, afl=affiliations, model_name=model, dim=dim, dst_dir=dataframe_dst_dir)
    real_afl_df = pd.read_csv("counter_affiliations_broader.csv")
    real_afl = real_afl_df["political_view"].tolist()
    predicted_afl = list(affiliations.values())
    #plot_confusion_matrix(y_true=real_afl, y_pred=predicted_afl, model=model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--dim")
    parser.add_argument("--affiliations_fname", default="affiliations.json")
    parser.add_argument("--sampled_fname", default="sampled_for_stance_4000.csv", required=True)
    parser.add_argument("--compute_stance", action="store_true")
    parser.add_argument("--dataframe_dst_dir")
    args = parser.parse_args()
    main(model=args.model, dim=args.dim, affiliations_fname=args.affiliations_fname, sampled_fname=args.sampled_fname,
         compute_stance_flag=args.compute_stance, dataframe_dst_dir=args.dataframe_dst_dir)
    #affiliations_fname = f"affiliations_{dim}_{model}_newlabel.json"     #_newlabel