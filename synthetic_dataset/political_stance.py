import json
import os
import re
import time
import matplotlib.pyplot as plt
import pandas as pd
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from openai import OpenAI, RateLimitError


def compute_stance(client, df, affiliations_fname, model, affiliations=None):
    if affiliations:
        df = df[~df.account_id.astype(str).isin(list(affiliations.keys()))]
    else:
        affiliations = {}

    i = 0
    step = 15
    df = df.drop(columns=[c for c in df.columns if c not in ["account_id", "content"]])
    df = df.reset_index()
    while i < len(df):
        print(i)
        sub_df = df.loc[i:i + step]
        dictionary = {r['account_id']: r['content'] + "\n" for _, r in sub_df.iterrows()}
        content = ("Instruction: I will now give you a dictionary. The keys represent IDs, and the values are"
                   "texts associated to each ID. I need you to look at each text and tell if its political "
                   "leaning is 'far left', 'left', 'center', 'non political', 'right', 'far right'. "
                   "OUTPUT INSTRUCTION: Return the answer in form of a json dictionary where the keys are the same"
                   "of the dictionary I provide, and the values are the labels associated to the text. "
                   "Don't write anything else. The answer should look like this:\n"
                   "{\n "
                   f"Input dictionary: {dictionary}")
        try:
            if model != "gpt-5":
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
            else:
                response = client.responses.create(
                    model=model,
                    input=content
                )
                answer = response.output_text
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
    dc = {v: c.count(v) for v in ["far left", "left", "center", "non political", "right", "far right"]}     # list(set(c))
    plt.bar(list(dc.keys()), list(dc.values()))
    plt.show()


if __name__ == "__main__":
    api_key_mistral = "dW87PLUULfArg0ys0XevF6HyOJUeJJHP"
    api_gpt = "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"
    model = "gpt-5"
    sampled_fname = "sampled_for_stance_4000.csv"
    if model == "gpt-5":
        client = OpenAI(api_key=api_gpt)
    else:
        client = Mistral(api_key=api_key_mistral)

    df = pd.read_csv("../dataset/posts_processed_stopwords_without_sampled.csv")    # posts_processed_stopwords.csv
    affiliations_fname = f"affiliations_4000_{model}.json"

    df = df[df.content.str.split().str.len().between(40, 1000)]
    if not os.path.exists(sampled_fname):
        sampled = df.sample(4000)
        sampled = sampled.reset_index()
        sampled.to_csv(sampled_fname)
    sampled = pd.read_csv(sampled_fname)
    affiliations = None
    if os.path.exists(affiliations_fname):
        with open(affiliations_fname, 'r') as f:
            affiliations = json.load(f)
    affiliations = compute_stance(client=client, df=sampled, affiliations_fname=affiliations_fname, model=model, affiliations=affiliations)

    plot_stance(affiliations)

