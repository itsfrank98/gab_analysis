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
from numpy import array


API_KEY_MISTRAL = "DodgOOtH2qzN13X0xowpPQZqTE1glFI2"

def compute_stance(client, sampled_dict, affiliations_fname, model, political_leanings, id_column, affiliations=None):
    if affiliations:
        for el in list(affiliations.keys()):
            sampled_dict.pop(el)
    else:
        affiliations = {}

    i = 0
    step = 100
    political_leaning_str = ", ".join(political_leanings)

    while i < len(sampled_dict):
        print(i)
        sub_dict = dict(list(sampled_dict.items())[i:i+step])
        content = ("Instruction: I will now give you a dictionary. The keys represent IDs, and the values are "
                   "texts associated to each ID. I need you to look at each text and tell if its political "
                   f"leaning is {political_leaning_str}. Classify the texts considering the american political point "
                   "of view. If you can't decide a label, mark it as 'unknown'. If the content is not about politics, "
                   "mark it as 'non_political'. Assign each text exactly one leaning.\n"
                   "OUTPUT INSTRUCTION: Return the answer in form of a json dictionary where the keys are the same "
                   "of the dictionary I provide, and the values are the labels associated to the text. "
                   "Don't write anything else."
                   f"Input dictionary: {sub_dict}")
        try:
            if model == "local-model":
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.8
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
    lc = array(list(dc.values()))
    total = sum(lc)
    percentages = lc/total * 100

    fig, ax = plt.subplots()
    bars = ax.bar(list(dc.keys()), list(dc.values()))
    #plt.bar(list(dc.keys()), list(dc.values()))
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x position
            height,  # y position (top of bar)
            f'{pct:.1f}%',  # formatted percentage
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    plt.xticks(rotation=45, ha='right')
    os.makedirs('plots', exist_ok=True)
    plt.tight_layout()
    plt.savefig('plots/political_stance_{}.svg'.format(model))
    plt.show()


def main(model, dim, affiliations_fname, sampled_fname, id_column, real_affiliations_path=None, dataframe_dst_dir=None,
         compute_stance_flag=False):
    political_leanings = ["far_left", "left", "center", "right", "far_right"]
    #political_leanings = ['neo_nazi', 'member_of_national_socialist_network', 'far_left', 'conservative', 'far_right', 'feminist', 'member_of_isis', 'white_supremacist', 'radical_zionist', 'black_supremacist', 'radical_feminist', 'radical_feminism', 'republican', 'liberal', 'eco_terrorism', 'liberal_middle_left', 'panafricanist', 'conservative_middle_right', 'anti_feminist', 'centrist', 'conservative (middle-right), feminist']
    # mistral-large-latest
    if model == "local-model":
        base_url = "http://127.0.0.1:1234/v1"
        client = OpenAI(base_url=base_url, api_key="foo")
    else:
        client = Mistral(api_key=API_KEY_MISTRAL)

    affiliations = None

    if not os.path.exists(sampled_fname):
        if sampled_fname.endswith(".csv"):
            df = pd.read_csv("../../dataset/posts_processed_stopwords_without_sampled.csv")  # posts_processed_stopwords.csv
            df = df[df.content.str.split().str.len().between(40, 1000)]
            sampled = df.sample(dim)
            sampled = sampled.reset_index()
            sampled.to_csv(sampled_fname)

    if os.path.exists(affiliations_fname):
        with open(affiliations_fname, 'r') as f:
            affiliations = json.load(f)

    if sampled_fname.endswith(".csv"):
        df = pd.read_csv(sampled_fname)
        sampled_dict = {r[id_column]: r['content'] for _, r in df.iterrows()}
    elif sampled_fname.endswith(".json"):
        with open(sampled_fname, 'r') as f:
            sampled = json.load(f)
            if type(sampled) == dict:
                sampled_dict = sampled
            elif type(sampled) == list:
                sampled_dict = {i: sampled[i] for i in range(len(sampled))}

    if compute_stance_flag:
        affiliations = compute_stance(client=client, sampled_dict=sampled_dict, affiliations_fname=affiliations_fname, model=model,
                                      affiliations=affiliations, political_leanings=political_leanings, id_column=id_column)

    plot_stance(affiliations, political_leanings + ["unknown", "non_political"], model)  # , "non-political"
    #plot_stance(affiliations, political_leanings + ["unknown"], model)
    if dataframe_dst_dir:
        os.makedirs(dataframe_dst_dir, exist_ok=True)
        create_dataframes(df=sampled, afl=affiliations, leanings=political_leanings + ["unknown", "non_political"],  #, "non-political"
                          model_name=model, dim=dim, dst_dir=dataframe_dst_dir, id_field=id_column)
    if real_affiliations_path:
        real_afl_df = pd.read_csv(real_affiliations_path)
        real_afl = real_afl_df["political_view"].tolist()
        predicted_afl = list(affiliations.values())
        plot_confusion_matrix(y_true=real_afl, y_pred=predicted_afl, model=model)


if __name__ == "__main__":
    #"""
    parser = ArgumentParser()
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--dim", required=False, default=0)
    parser.add_argument("--affiliations_fname", default="affiliations.json", help="Name of the file "
                                                                      "containing the text. Can be a json or a csv")
    parser.add_argument("--content_fname", default="sampled_for_stance_4000.csv", required=True)
    parser.add_argument("--compute_stance", action="store_true")
    parser.add_argument("--perform_test", action="store_true")
    parser.add_argument("--dataframe_dst_dir", required=False)
    parser.add_argument("--real_affiliations_path", required=False, default=None)
    parser.add_argument("--id_column", required=False, default="account_id")
    args = parser.parse_args()

    main(model=args.model, dim=args.dim, affiliations_fname=args.affiliations_fname, sampled_fname=args.content_fname,
         compute_stance_flag=args.compute_stance, dataframe_dst_dir=args.dataframe_dst_dir,
         real_affiliations_path=args.real_affiliations_path, id_column=args.id_column)
    #"""

    """
    args = {
        "model": "local-model",
        "dim": 10,
        "affiliations_fname": "counter/affiliations_llama_coarse.json",
        "sampled_fname": "counter/counter_concatenated.csv",
        "compute_stance": False,
        "perform_test": True,
        "dataframe_dst_dir": "dataframes_counter",
        "real_affiliations_path": ""
    }
    main(model=args["model"], dim=args["dim"], affiliations_fname=args["affiliations_fname"], sampled_fname=args["sampled_fname"],
         compute_stance_flag=args["compute_stance"], dataframe_dst_dir=args["dataframe_dst_dir"],
         real_affiliations_path=args["real_affiliations_path"])
    """
    #affiliations_fname = f"affiliations_{dim}_{model}_newlabel.json"     #_newlabel

    #python .\political_stance.py --model mistral-large-latest --affiliations_fname ..\hateful_bios\afl.json --content_fname ..\hateful_bios\hateful_bios.json --compute_stance