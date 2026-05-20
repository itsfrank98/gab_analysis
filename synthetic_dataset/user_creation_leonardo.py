"""
This script queries the LLM for creating the user profiles and posts. The profiles are described as a set of columns
with the profile features (age, job, interests, username, bio and so on). The posts are created based on the profile.
The LLM can be queried in three ways:
- As an LLM deployed on LMStudio
- On llama.cpp
- Loded as a pretrained model with the transformers library. In this case, it is also possible to use a fine-tuned model
  by providing the path to the adapters

The LLM outputs for each user a list of 10 posts in json format and then saved in a unique file with all the posts from
the users. If any error happens in processing the response, the output is saved in a txt file as raw text for further
processing.
By providing the path to a csv file containing posts, they will be used for teaching the LLM the style to adopt in the posts
"""

import argparse
import requests
import json
import numpy as np
import pandas as pd
import re
import os
import torch
from tqdm import tqdm
from creation_options import *
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import random
import time

now = time.time()
os.makedirs("users", exist_ok=True)


def load_model_and_tokenizer(model_name, adapter_path, load_in_4bit):
    print(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure a pad token exists (required for batched generation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {model_name}")
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    # The adapter directory must contain the PEFT config + weights saved under
    # the name "final lora adapter".  PEFT loads from the directory path.
    if adapter_path:
        print(f"Attaching LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer

def generate_posts(model, tokenizer, prompt, max_new_tokens, device) -> str:
    """Run inference for a single prompt and return the generated text."""
    print("tokenizing...")
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
    print("decoding...")
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def create_user_initial_prompt(user_id, user_n_posts, user_name, user_bio, state_of_origin, gender, ethnicity, religion,
                               political_view, user_interests, age_interval, job, peft_model, tokenizer,
                               real_posts=None):
    """
    Create the user profiles and/or the posts for that user fitting his profile
    :param user_id: ID of the considered user
    :param user_n_posts: Number of posts to generate for the user
    :param user_name, user_bio, state_of_origin, gender, ethnicity, religion, political_view,
                user_interests, age_interval, job: Socio-demographical traits of the user
    to None if the LLM is loaded as a Peft model
    :param peft_model: Peft LLM to use. Set it to None if the LLM is running on LMStudio or llama.cpp. If this parameter is
    not None, the llm_provider param is ignored.
    :param tokenizer: tokenizer for the peft model
    """
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


    if real_posts:
        sampled_posts = random.sample(real_posts, 10)
        sampled_posts = [p.replace('\n', ' ').replace('\r', ' ') for p in sampled_posts]
        few_shot_part = (
            "Here are some example posts published on the platform. Analyze the tone, vocabulary register and sentence "
            "structure. Ignore what the posts are about entirely.: "
            f"- \"{sampled_posts[0]}\""
            f"- \"{sampled_posts[1]}\""
            f"- \"{sampled_posts[2]}\""
            f"- \"{sampled_posts[3]}\""
            f"- \"{sampled_posts[4]}\""
            f"- \"{sampled_posts[5]}\""
            f"- \"{sampled_posts[6]}\""
            f"- \"{sampled_posts[7]}\""
            f"- \"{sampled_posts[8]}\""
            f"- \"{sampled_posts[9]}\""
            )

        prompt = (f"Your username is {user_name}. You are a {gender} and your age is in the interval {age_interval}. You are an "
                  f"american citizen, born and raised in {state_of_origin} and your ethnicity is {ethnicity}. {religious_part}."
                  f" {interests_part}. Your political orientation is {political_view}. You have a profile on the gab.com "
                  f"social network. The bio of your profile is {user_bio}. {job_part}. \n"
                  f"Write the next {user_n_posts} posts you would publish on a loosely moderated "
                  f"social media platform where people can freely express their feelings without risking to be banned. "
                  f"The posts can include strong, unfiltered opinions, inflammatory language, or even hate speech or "
                  f"incitement to violence if it fits your persona. \n"
                  f"{few_shot_part}"
                  f"OUTPUT INSTRUCTIONS: \n"
                  "* The post must be relevant to your passions and opinions."
                  "* Write only textual posts, without multimedia content;\n"
                  "* The expected output is a list of posts in plain text. Feel free to use hashtags or emojis."
                  "Use them in different parts of the post. Return the posts as a list in JSON format."
                  "The expected output has the following structure: {\"response\": [\"post1\", \"post2\", ..., \"post_n\"]}\n"
                  )
        ok = False
        counter_not_ok = 0
        while not ok and counter_not_ok<3:
            print(user_id)
            if peft_model:
                generated_text = generate_posts(model=peft_model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=1000,
                                                device="cuda")
            try:
                matches = re.findall(r'\{"response"[^{}]*\}', user_posts, re.DOTALL)
                if matches:
                    d["posts"] = json.loads(matches[-1])["response"]        # I use the -1 index because sometimes the LLM puts example dictionaries in the answer, putting the actual one as the last
                    ok = True
                    counter_not_ok = 0
                else:
                    counter_not_ok += 1
            except (json.decoder.JSONDecodeError, KeyError) as err:
                print("ERROR!!", user_id)
                print(err)
                counter_not_ok += 1
    return d


def create_user_moody_prompt(user_id, user_n_posts, user_name, user_bio, state_of_origin, gender, ethnicity, religion, political_view,
                user_interests, age_interval, job, peft_model, tokenizer, real_posts=None):
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
    ld = [d] * user_n_posts

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

    if real_posts:
        sampled_posts = random.sample(real_posts, 10)
        sampled_posts = [p.replace('\n', ' ').replace('\r', ' ') for p in sampled_posts]
        few_shot_part = (
            "Here are some example posts published on the platform. Analyze the tone, vocabulary register and sentence "
            "structure. Ignore what the posts are about entirely.: "
            f"- \"{sampled_posts[0]}\""
            f"- \"{sampled_posts[1]}\""
            f"- \"{sampled_posts[2]}\""
            f"- \"{sampled_posts[3]}\""
            f"- \"{sampled_posts[4]}\""
            f"- \"{sampled_posts[5]}\""
            f"- \"{sampled_posts[6]}\""
            f"- \"{sampled_posts[7]}\""
            f"- \"{sampled_posts[8]}\""
            f"- \"{sampled_posts[9]}\""
            )
        contexts = {
            "emotional_state": [
                "are tired after a long work day",
                "are in a good mood, just had dinner with friends",
                "are bored on a Sunday afternoon",
                "are anxious about something at work",
                "are excited about a personal achievement",
                "are grieving or sad about something",
                "are happy about something",
                "are travelling",
                "are on vacation",
                "have argued with someone recently",
                "are procrastinating",
                "are amused by something funny you just saw online",
                "are angry because you just found out your train is late"
            ],
            "format_style": [
                "rant about something",
                "ask a question to your followers",
                "share a personal anecdote",
                "post an opinion with no explanation",
                "give an advice to your followers",
                "use heavy sarcasm",
            ]
        }

        for i in range(user_n_posts):
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
                # f"{few_shot_part}"
                f"OUTPUT INSTRUCTIONS: \n"
                "* The post must be relevant to your passions and opinions;"
                "* The post must not be longer than 100 words;"
                "* The post doesn't contain multimedia content;\n"
                "* The expected output is a JSON dictionary with this structure: {\"response\": <post>}. "
                "The output must only contain the json, nothing else."
            )
            while not ok and counter_not_ok<3:
                print(user_id)
                generated_text = generate_posts(model=peft_model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=1000,
                                                device="cuda")
                print(generated_text)
                try:
                    matches = re.findall(r'\{"response"[^{}]*\}', user_posts, re.DOTALL)
                    if matches:
                        ld[i]["format_style"] = format_style
                        ld[i]["state"] = state
                        ld[i]["posts"] = json.loads(matches[-1])["response"]        # I use the -1 index because sometimes the LLM puts example dictionaries in the answer, putting the actual one as the last
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


def main(output_fname, model_name, adapter_path, type_prompt, real_posts_path=None, real_posts_text_column=None,
         user_n_posts=10,  users_profiles_path=None, already_created_posts=None):
    user_id = "p{}"
    dicts_list = []
    present_users = []

    # Provide a csv containing the user descriptions and create the posts from there
    df = pd.read_csv(users_profiles_path)
    model, tokenizer, real_posts = None, None, None
    if model_name:
        model, tokenizer = load_model_and_tokenizer(model_name=model_name, adapter_path=adapter_path, load_in_4bit=True)
    if real_posts_path:
        real_posts = pd.read_csv(real_posts_path)
        real_posts = real_posts[real_posts["language"]=="en"]
        real_posts["_word_count"] = real_posts[real_posts_text_column].str.split().str.len()
        real_posts = real_posts[(real_posts["_word_count"] >= 5) & (real_posts["_word_count"] <= 300)]
        real_posts = real_posts.drop(columns=["_word_count"]).drop_duplicates(subset=real_posts_text_column)
        real_posts = real_posts[real_posts_text_column].tolist()
    if already_created_posts:
        posts = pd.read_csv(already_created_posts)
        present_users = list(posts.drop_duplicates(subset="account_id")["account_id"])
    for i, row in tqdm(df.iterrows()):
        if row["account_id"] not in present_users:
            print(row["account_id"])
            if type_prompt == "initial":
                d = create_user_initial_prompt(user_id=row["account_id"], user_name=row["username"], user_bio=row["user_bio"],
                            state_of_origin=row["state_of_origin"], gender=row["gender"],
                            political_view=row["political_leaning"], user_interests=row["interests"], age_interval=row["age_interval"],
                            job=row["profession"], user_n_posts=user_n_posts, ethnicity=row["ethnicity"], religion=row["religion"],
                            peft_model=model, tokenizer=tokenizer, real_posts=real_posts)
                dicts_list.append(d)
            else:
                d = create_user_moody_prompt(user_id=row["account_id"], user_name=row["username"], user_bio=row["user_bio"],
                            state_of_origin=row["state_of_origin"], gender=row["gender"],
                            political_view=row["political_leaning"], user_interests=row["interests"], age_interval=row["age_interval"],
                            job=row["profession"], user_n_posts=user_n_posts, ethnicity=row["ethnicity"], religion=row["religion"],
                            peft_model=model, tokenizer=tokenizer, real_posts=real_posts)
                for e in d:
                    dicts_list.append(e)
        if i == 1:
            break

    no_duplicated_information = False
    df = pd.DataFrame(dicts_list)
    if "posts" in df.columns and type_prompt=="initial":
        df = df.explode("posts").reset_index(drop=True)
        if no_duplicated_information:
            for col in df.columns[:-1]:
                df.loc[df.duplicated(subset=["profile_id"]), col] = None
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
    parser.add_argument("--model_name", required=False,
                        help="Name of the model to use, in case you want to load it in the code")

    parser.add_argument("--output_fname", type=str, default="foo.csv",
                        help="Path where the output will be saved. The output can either be the user descriptions, or "
                             "the user descriptions together with the synthetic posts")
    parser.add_argument("--real_posts_text_column", type=str, default="content",
                        help="Name of the column in the csv file containing the content")
    parser.add_argument("--real_posts_path", type=str,
                        help="Path to the file containing the real posts to use for learning style")
    parser.add_argument("--user_n_posts", type=int, required=False, default=5,
                        help="Number of posts to create for each user")
    parser.add_argument("--users_profiles_path", type=str, default="synthetic_user_profiles.csv",
                        help="Path to the csv file containing the users' profiles from which the posts will be created")
    parser.add_argument("--already_created_posts", type=str, default=None,
                        help="Path to the csv file containing the posts that have already been created. Set it to "
                             "complete the dataset, if some users were not generated or there were format errors")
    parser.add_argument("--type_prompt", type=str, default="with_mood", choices=["initial", "with_mood"],
                        help="Type of prompt to use. 'with_mood' includes information about the user's mood")
    args = parser.parse_args()
    main(user_n_posts=args.user_n_posts, users_profiles_path=args.users_profiles_path, output_fname=args.output_fname,
         real_posts_path=args.real_posts_path, model_name=args.model_name, adapter_path=args.adapter_path,
         real_posts_text_column=args.real_posts_text_column, already_created_posts=None, type_prompt=args.type_prompt)    #args.already_created_posts

# python user_creation.py --user_n_posts 10 --users_profiles_path synthetic_user_profiles.csv --output_fname synthetic_posts.csv --model_name DreadPoor/Irix-12B-Model_Stock --adapter_path fine_tuning/lora_Irix/final_lora_adapter

# python user_creation.py --user_n_posts 10 --users_profiles_path synthetic_user_profiles.csv --output_fname synthetic_posts_added.csv --real_posts_path posts_processed.csv --already_created_posts synthetic_posts_irix_fewshot.csv

# llama-server --model LLMs_gguf/Irix-12B-Model_Stock.i1-Q4_K_S.gguf --n-gpu-layers 999 --port 8080