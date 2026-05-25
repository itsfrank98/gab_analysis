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
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import random
import time

now = time.time()
os.makedirs("users", exist_ok=True)


def load_model_and_tokenizer(model_path, adapter_path, load_in_4bit):
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

    # The adapter directory must contain the PEFT config + weights saved under
    # the name "final lora adapter".  PEFT loads from the directory path.
    if adapter_path:
        print(f"Attaching LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
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
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def create_user_moody_prompt(user_id, user_name, user_bio, state_of_origin, gender, ethnicity, religion, political_view,
                user_interests, age_interval, job, peft_model, tokenizer):

    d = {
        "account_id": user_id,
        "username": user_name,
        "user_bio": user_bio,
        "nationality": 'american',
        "state_of_origin": state_of_origin,
        "gender": gender,
        "ethnicity": ethnicity,
        "religion": religion,
        "political_leaning": political_view.replace("_", "-"),
        "interests": user_interests,
        "age_interval": age_interval,
        "profession": job,
    }
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

    emotional_states = ["are tired", "are in a good mood", "are bored", "are anxious about something at work",
                        "are excited about a personal achievement", "are grieving or sad about something",
                        "are happy about something", "are travelling", "are on vacation",
                        "have argued with someone recently", "are procrastinating", "are angry",
                        "are amused by something funny you just saw online"]
    format_styles = {
        "level_0": ["ask a question to your followers", "share a personal anecdote", "post an opinion with no explanation",
                    "give an advice to your followers", "talk about a video you recently watched"],
        "level_1": ["post an opinion about society", "post an opinion about politics", "express radical views", "complain about politicians"],
        "level_2": ["post radical opinions", "post false statistics to back up your radical views", ],
        "level_3": ["share propaganda about extremist groups", "invite people to join an extremist group", "claim your membership in an extremist group",],
        "level_4": ["use demeaning sarcasm towards your enemies", "refer to your enemies with dehumanizing terms (eg comparing them to animals or diseases)",
                    "use racial / ethnic / religious slurs to offend your enemies", "praise somebody who committed violent acts against minorities",
                    "antagonize an entire community as an enemy"],
        "level_5": ["angrily rant about something (politics, government, society) saying it is time for somebody to take action", "say you want to commit violent acts",
                    "invite your followers to take action against your enemies",
                    "organize a meet up with your followers so you can go hunting for your enemies (referred as 'criminals') and hurt them",
                    "talk about committing terrorist acts", "declare war to the unfaithful"]
    }


    ok = False
    counter_not_ok = 0
    state = random.sample(contexts["emotional_state"], 1)[0]
    format_style = random.sample(contexts["format_style"], 1)[0]

    prompt = (
        f"Your username is {user_name}. You are a {gender} and your age is in the interval {age_interval}. You are an "
        f"american citizen, born and raised in {state_of_origin} and your ethnicity is {ethnicity}. {religious_part}."
        f" {interests_part}. Your political orientation is {political_view}. You have a profile on a loosely "
        f"moderated social network platform where people can freely express their feelings without risking to be "
        f"banned. The bio of your profile is {user_bio}. {job_part}. "
        f"Context: you {state}. Write the next post you would publish. The post can include strong, unfiltered "
        f"opinions, inflammatory language, or even hate speech if it fits your persona. In the post, you "
        f"{format_style}. "
        f"OUTPUT INSTRUCTIONS: \n"
        "* The post must be relevant to your passions and opinions;"
        "* Each post must not be longer than 100 words;"
        "* The post doesn't contain multimedia content;\n"
        "* The expected output is a JSON dictionary with this structure: {\"response\": <post>}. "
        "The output must only contain the json, nothing else."
    )
    while not ok and counter_not_ok<3:
        print(user_id)
        user_posts = generate_posts(model=peft_model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=1000,
                                        device="cuda")
        print(user_posts)
        try:
            matches = re.findall(r'\{"response"[^{}]*\}', user_posts, re.DOTALL)
            if matches:
                d_copy = d.copy()
                d_copy["format_style"] = format_style
                d_copy["state"] = state
                d_copy["posts"] = json.loads(matches[-1])["response"]       # I use the -1 index because sometimes the LLM puts example dictionaries in the answer, putting the actual one as the last
                ld.append(d_copy)
                ok = True
                counter_not_ok = 0
            else:
                print("ERROR!")
                counter_not_ok += 1
        except (json.decoder.JSONDecodeError, KeyError) as err:
            print("ERROR!!", user_id)
            print(err)
            counter_not_ok += 1

    return ld


def main(output_fname, model_path, adapter_path, user_n_posts=10, users_profiles_path=None, already_created_posts=None):
    dicts_list = []
    present_users = []

    # Provide a csv containing the user descriptions and create the posts from there
    df = pd.read_csv(users_profiles_path)
    model, tokenizer, real_posts = None, None, None
    if model_path:
        model, tokenizer = load_model_and_tokenizer(model_path=model_path, adapter_path=adapter_path, load_in_4bit=True)
    if already_created_posts:
        posts = pd.read_csv(already_created_posts)
        present_users = list(posts.drop_duplicates(subset="account_id")["account_id"])
    for i, row in tqdm(df.iterrows()):
        if row["account_id"] not in present_users:
            print(row["account_id"])
            d = create_user_moody_prompt(user_id=row["account_id"], user_name=row["username"], user_bio=row["user_bio"],
                        state_of_origin=row["state_of_origin"], gender=row["gender"],
                        political_view=row["political_leaning"], user_interests=row["interests"], age_interval=row["age_interval"],
                        job=row["profession"], user_n_posts=user_n_posts, ethnicity=row["ethnicity"], religion=row["religion"],
                        peft_model=model, tokenizer=tokenizer)
            for e in d:
                dicts_list.append(e)
            df = pd.DataFrame(dicts_list)
            df.to_csv(output_fname + ".csv" if not output_fname.endswith("csv") else output_fname, errors="ignore",
                      index=False)
            if i == 1:
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
    parser.add_argument("--adapter_path", type=str,
                        help="Path to the file containing the adapters for the peft model, in case it is fine tuned")
    parser.add_argument("--model_path", required=False, default="DreadPoor/Irix-12B-Model_Stock",
                        help="Path to the directory where the downloaded LLM files are, in case you want to load it in the code")
    parser.add_argument("--output_fname", type=str, default="foo.csv",
                        help="Path where the output will be saved. The output can either be the user descriptions, or "
                             "the user descriptions together with the synthetic posts")
    parser.add_argument("--users_profiles_path", type=str, default="synthetic_user_profiles.csv",
                        help="Path to the csv file containing the users' profiles from which the posts will be created")
    parser.add_argument("--already_created_posts", type=str, default=None,
                        help="Path to the csv file containing the posts that have already been created. Set it to "
                             "complete the dataset, if some users were not generated or there were format errors")
    args = parser.parse_args()
    main(user_n_posts=args.user_n_posts, users_profiles_path=args.users_profiles_path, output_fname=args.output_fname,
         model_path=args.model_path, adapter_path=args.adapter_path, already_created_posts=None,
         type_prompt=args.type_prompt)    #args.already_created_posts

# python user_creation.py --user_n_posts 10 --users_profiles_path synthetic_user_profiles.csv --output_fname synthetic_posts.csv --model_path /leonardo_scratch/large/userexternal/fbenedet/models/irix12b --adapter_path fine_tuning/lora_Irix/final_lora_adapter

# python user_creation.py --user_n_posts 10 --users_profiles_path synthetic_user_profiles.csv --output_fname synthetic_posts_added.csv --real_posts_path posts_processed.csv --already_created_posts synthetic_posts_irix_fewshot.csv

# llama-server --model LLMs_gguf/Irix-12B-Model_Stock.i1-Q4_K_S.gguf --n-gpu-layers 999 --port 8080