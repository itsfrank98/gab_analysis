# In this file we create some left and far-left bios since there are too few of them. We will create them starting from
# the posts instead of the bios.
import json
import pandas as pd
from openai import OpenAI, BadRequestError

leaning = "non_political"
posts_source = "../stance/sampled_for_stance_4000.csv"
afl_source = "../stance/affiliation_dicts/on_gab_dataset/affiliations_4000_mistral.json"
users_to_generate = 100

base_url = "http://127.0.0.1:1234/v1"
client = OpenAI(
    base_url=base_url,
    api_key="OPENAI_API_KEY",
)

posts = pd.read_csv(posts_source)
posts_dict = posts.to_dict("records")
with open(afl_source, "r") as f:
    afl = json.load(f)

left_afl_k = [int(k) for k in list(afl.keys()) if afl[k] == leaning]
leaning_posts = posts[posts.account_id.isin(left_afl_k)]

i = 0
bios_d = {}
while i < len(posts_dict) and len(bios_d) < users_to_generate:
    print(i)
    ds = posts_dict[i:i+20]
    posts_subdict = {el["account_id"]: el["content"] for el in ds}
    prompt = ("I am now going to provide you a dictionary. The keys represent user IDs, and each value is a text "
              "obtained by concatenating the posts that a the user published on a social network website called gab.com. "
              "I need you to look at the texts and generate a bio for that specific user. The bio should contain 50 "
              "words at most, and has to mirror the opinions expressed in the posts. Here are some bios you can take as "
              "example. Replicate the same style in the ones you generate. "
              "{ Bio1: \"I'm here to troll republicans and fascists alike. If you can't take the heat, get out of the kitchen! I'm not afraid to offend anyone with my bigotry. #TrollLife #OffendedBro\","
              "Bio2: \"tired of right wing cucks ruining this country\", "
              "Bio3: \"they can't censor us if we don't let them. \"}\n"
              f"INPUT DICTIONARY: {posts_subdict} \n"
              "OUTPUT INSTRUCTIONS: Your response should be a dictionary, having as keys the user IDs and as values "
              "their generated bio. This is an example of the structure: {\"user1_id\": \"user1_bio\", "
              "\"user2_id\": \"user2_bio\", ...}. \n Do not write anything else.")
    try:
        resp = client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
    except BadRequestError as e:
        print(e)
    try:
        d = json.loads(resp.choices[0].message.content)
        bios_d.update(d)
    except:
        print("ERROR!")
        print(resp.choices[0].message.content)
        print("\n")
#TODO devo computare il political stance delle nuove bio, quelle non political
    with open(f"generated_left_farleft/{leaning}_bios.json", "w") as f:
        json.dump(bios_d, f, indent=4)

    i += 20
