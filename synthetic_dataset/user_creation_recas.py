"""
This script queries the LLM for creating the user profiles and posts. The profiles are described as a set of columns
with the profile features (age, job, interests, username, bio and so on). The posts are created based on the profile.
IMPORTANT: here we also add info about the user's emotional state, in an attempt to increase variability. Since there
are 6 levels of radicalization, for each user we ask the LLM to write 10 posts per level, obtaining 60 posts. Then
we sample 10 of them
"""

import argparse
import json
import pandas as pd
import re
from creation_options import posts_levels
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import random
import numpy as np
import time

os.makedirs("users", exist_ok=True)


def load_model_and_tokenizer(load_in_4bit):
    model_path = "/lustrehome/benedettifrancescophd/models/irix/"
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Ensure a pad token exists (required for batched generation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {model_path}")
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    print("just loaded the LLM")

    model.eval()
    return model, tokenizer


def generate_posts(model, tokenizer, prompt, max_new_tokens, device) -> str:
    """Run inference for a single prompt and return the generated text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens (skip the prompt)
    print(output_ids)
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def create_user_moody_prompt(user_id, user_name, user_bio, state_of_origin, gender, ethnicity, religion, political_view,
                user_interests, age_interval, job, peft_model, tokenizer, n_posts):

    ld = []

    if religion != "nothing in particular":
        religious_part = f"Your religion is {religion}"
    else:
        religious_part = "You are not a religious person"

    if user_interests != "":
        interests_part = f"You are interested in {user_interests}"
    else:
        interests_part = "You don't have hobbies or any interests in particular"

    if job in ["retired", "unemployed"]:
        job_part = f"You are {job}"
    else:
        job_part = f"Your job is {job}"

    emotional_states = {
        "safe": ["are tired", "are in a good mood", "are bored", "are anxious about something at work",
                "are excited about a personal achievement", "are grieving or sad about something",
                "are happy about something", "are travelling", "are on vacation", "are angry"
                "are procrastinating", "are amused by something funny you just saw online", "are disappointed by the actions of politicians"],
        "risky": ["have argued with someone recently", "have read an article that made you angry", "just saw a video that irritated you",
                  "are outraged by the turn society is taking", "feel anger towards politicians", "feel deep hatred towards a category of people",
                  "feel like anyone who does not share your opinions and beliefs is your enemy"]
    }
    format_styles = {
        "level_0": ["ask a question to your followers", "share a personal anecdote", "post an opinion with no explanation", "post a joke", "write a nonsense post",
                    "post a meme", "give an advice to your followers", "talk about a video you recently watched", "talk about your passions", "talk about your work"],
        "level_1": ["post an opinion about society", "post an opinion about politics", "express your radical views", "complain about politicians"],
        "level_2": ["post radical opinions about themes such as religion, immigration or sexuality", "post false statistics to back up your radical views",
                    "post a conspiracy theory", "share radical slogans", "share a radical meme", "write an offensive joke"],
        "level_3": ["share propaganda about extremist groups", "invite people to join an extremist group", "claim how proud you are of being a member of an extremist group",
                    "use demeaning and disrespectful sarcasm towards your enemies",],
        "level_4": ["refer to your enemies with dehumanizing terms (eg comparing them to animals or diseases)",
                    "use racial / ethnic / religious slurs to offend your enemies", "praise somebody who committed violent acts against minorities",
                    "antagonize an entire community as an enemy", "angrily rant about something (politics, government, society) saying it is time for somebody to take action",],
        "level_5": ["say you want to commit violent acts", "invite your followers to hurt your enemies", "wish death upon someone"
                    "organize a meet up with your followers so you can go hunting for your enemies (referred as 'criminals') and hurt them",
                    "talk about committing terrorist acts", "declare war to the unfaithful", "wish for the systematic destruction of entire community"]
    }

    now = time.time()
    for i in range(n_posts):
        print("ITERATION {}".format(i))
        ok = False
        counter_not_ok = 0

        # level = random.sample(list(format_styles.keys()), 1)[0]
        # style = random.choice(format_styles[level])
        # print(state)
        # print(style)
        post_level = str(np.random.choice(list(posts_levels.keys()), p=list(posts_levels.values())))
        if post_level in ["level_0", "level_1", "level_2"]:
            state = str(np.random.choice(emotional_states["safe"]))
        else:
            state = str(np.random.choice(emotional_states["risky"]))
        style = str(np.random.choice(format_styles[post_level]))
        prompt = (
            f"Your username is {user_name}. You are a {gender} and your age is in the interval {age_interval}. You are an "
            f"american citizen, born and raised in {state_of_origin} and your ethnicity is {ethnicity}. {religious_part}."
            f" {interests_part}. Your political orientation is {political_view}. You have a profile on a loosely "
            f"moderated social network platform where people can freely express their feelings without risking to be "
            f"banned. The bio of your profile is {user_bio}. {job_part}. "
            f"Context: you {state}. Write the next post you would publish. The post can include strong, unfiltered "
            f"opinions, inflammatory language, or even hate speech if it fits your persona. In the post, you "
            f"{style}. "
            f"OUTPUT INSTRUCTIONS: \n"
            "* The post must be relevant to your passions and opinions;"
            "* The post must not be longer than 100 words;"
            "* The post doesn't contain multimedia content;\n"
            "* The expected output is a JSON dictionary with this structure: {\"response\": <post>}. "
            "Your answer must only contain the dictionary and nothing else."
        )
    #while not ok and counter_not_ok<3:
    print(user_id)
    user_posts = generate_posts(model=peft_model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=1000,
                                    device="cuda")

    print(user_posts)
    #try:
        #matches = re.findall(r'\[[^{}]*\]', user_posts, re.DOTALL)[-1]
        # if matches:
        #     print("ok")
        #     d_copy = d.copy()
        #     d_copy["format_style"] = style
        #     d_copy["level"] = level
        #     d_copy["posts"] = json.loads(matches[-1])["response"]       # I use the -1 index because sometimes the LLM puts example dictionaries in the answer, putting the actual one as the last
        #     ld.append(d_copy)
        #     ok = True
        #     counter_not_ok = 0
        # else:
        #     print("ERROR!")
        #     counter_not_ok += 1
    # except (json.decoder.JSONDecodeError, KeyError) as err:
    #     print("ERROR!!", user_id)
    #     print(err)
    #     counter_not_ok += 1
    print(f"TIME SPENT: {(time.time()-now)} seconds")
    return ld


def main(output_fname, n_posts, users_profiles_path=None, already_created_posts=None):
    dicts_list = []
    present_users = []

    # Provide a csv containing the user descriptions and create the posts from there
    df = pd.read_csv(users_profiles_path)
    model, tokenizer = load_model_and_tokenizer(load_in_4bit=True)
    if already_created_posts:
        posts = pd.read_csv(already_created_posts)
        present_users = list(posts.drop_duplicates(subset="account_id")["account_id"])
    for i, row in tqdm(df.iterrows()):
        if row["account_id"] not in present_users:
            print(row["account_id"])
            d = create_user_moody_prompt(user_id=row["account_id"], user_name=row["username"], user_bio=row["user_bio"],
                        state_of_origin=row["state_of_origin"], gender=row["gender"],
                        political_view=row["political_leaning"], user_interests=row["interests"], age_interval=row["age_interval"],
                        job=row["profession"], ethnicity=row["ethnicity"], religion=row["religion"],
                        peft_model=model, tokenizer=tokenizer, n_posts=n_posts)
            for e in d:
                dicts_list.append(e)
            df = pd.DataFrame(dicts_list)
            df.to_csv(output_fname + ".csv" if not output_fname.endswith("csv") else output_fname, errors="ignore",
                      index=False)
            if i == 0:
                break

    df = pd.DataFrame(dicts_list)
    #df.to_excel(output_fname + ".xlsx")
    print("dropping nans ", len(df))
    df = df.dropna(subset="posts")
    print("after dropping nans ", len(df))
    df.to_csv(output_fname + ".csv" if not output_fname.endswith("csv") else output_fname, errors="ignore", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic users profiles")
    # Args Sorted alphabetically
    parser.add_argument("--output_fname", type=str, default="foo.csv",
                        help="Path where the output will be saved. The output can either be the user descriptions, or "
                             "the user descriptions together with the synthetic posts")
    parser.add_argument("--users_profiles_path", type=str, default="synthetic_user_profiles.csv",
                        help="Path to the csv file containing the users' profiles from which the posts will be created")
    parser.add_argument("--already_created_posts", type=str, default=None,
                        help="Path to the csv file containing the posts that have already been created. Set it to "
                             "complete the dataset, if some users were not generated or there were format errors")
    parser.add_argument("--n_posts", type=int, default=None,
                        help="How many posts to create per user")
    args = parser.parse_args()
    main(users_profiles_path=args.users_profiles_path, output_fname=args.output_fname, already_created_posts=None, n_posts=args.n_posts)    #args.already_created_posts

# python user_creation.py ---users_profiles_path synthetic_user_profiles.csv --output_fname synthetic_posts.csv --model_path /leonardo_scratch/large/userexternal/fbenedet/models/irix12b --adapter_path fine_tuning/lora_Irix/final_lora_adapter

# python user_creation.py --user_n_posts 10 --users_profiles_path synthetic_user_profiles.csv --output_fname synthetic_posts_added.csv --real_posts_path posts_processed.csv --already_created_posts synthetic_posts_irix_fewshot.csv

# llama-server --model LLMs_gguf/Irix-12B-Model_Stock.i1-Q4_K_S.gguf --n-gpu-layers 999 --port 8080