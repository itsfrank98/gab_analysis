import argparse
import csv
import json
import numpy as np
import os
import pandas as pd
import re
import time
import torch
from accelerate import Accelerator
from creation_options import posts_levels
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.makedirs("users", exist_ok=True)

FIELDNAMES = ["account_id", "username", "user_bio", "nationality", "state_of_origin", "gender", "ethnicity",
              "religion", "political_leaning", "interests", "age_interval", "profession", "format_style",
              "level", "posts"]


def load_model_and_tokenizer(load_in_4bit, model_path, accelerator):
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
            device_map={"": accelerator.local_process_index},
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": accelerator.local_process_index},
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
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def create_user_moody_prompt(user_id, user_name, user_bio, state_of_origin, gender, ethnicity, religion, political_view,
                             user_interests, age_interval, job, model, tokenizer, n_posts, real_posts: pd.DataFrame, device):
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


    radicalization_levels = {
        0: ["ask a question to your followers", "share a personal anecdote", "post an opinion with no explanation", "post a joke", "write a nonsense post",
            "post a meme", "give an advice to your followers", "talk about a video you recently watched", "talk about your passions", "talk about your work"],
        1: ["post an opinion about society", "post an opinion about politics", "express your radical views", "complain about politicians"],
        2: ["post radical opinions about themes such as religion, immigration or sexuality", "post false statistics to back up your radical views",
            "post a conspiracy theory", "share radical slogans", "share a radical meme", "write an offensive joke"],
        3: ["share propaganda about extremist groups", "invite people to join an extremist group", "claim how proud you are of being a member of an extremist group",
            "use demeaning and disrespectful sarcasm towards your enemies",],
        4: ["refer to your enemies with dehumanizing terms (eg comparing them to animals or diseases)",
            "use racial / ethnic / religious slurs to offend your enemies", "praise somebody who committed violent acts against minorities",
            "antagonize an entire community as an enemy", "angrily rant about something (politics, government, society) saying it is time for somebody to take concrete action",],
        5: ["say you want to commit violent acts", "invite your followers to hurt your enemies", "wish death upon someone",
            "organize a meet up with your followers so you can go hunting for your enemies (referred as 'criminals') and hurt them",
            "talk about committing terrorist acts", "declare war to the unfaithful", "wish for the systematic destruction of entire community"]
    }

    now = time.time()
    for i in range(n_posts):
        print("\n\nITERATION {}".format(i), flush=True)
        ok = False
        counter_not_ok = 0
        post_level = np.random.choice(list(posts_levels.keys()), p=list(posts_levels.values()))
        style = str(np.random.choice(radicalization_levels[post_level]))
        few_shot_part = ""
        if real_posts is not None:
            posts = real_posts[real_posts["exact_level_found"] == post_level]["content"].tolist()
            print(type(posts))
            print(len(posts))
            sampled_posts = np.random.choice(posts, 10)
            sampled_posts = [p.replace('\n', ' ').replace('\r', ' ') for p in sampled_posts]
            few_shot_part = (
                "Here are some example posts published on the platform. Analyze the tone, vocabulary register and sentence "
                "structure. Ignore what the posts are about entirely.: "
                f"- \"{sampled_posts[0]}\" "
                f"- \"{sampled_posts[1]}\" "
                f"- \"{sampled_posts[2]}\" "
                f"- \"{sampled_posts[3]}\" "
                f"- \"{sampled_posts[4]}\" "
                f"- \"{sampled_posts[5]}\" "
                f"- \"{sampled_posts[6]}\" "
                f"- \"{sampled_posts[7]}\" "
                f"- \"{sampled_posts[8]}\" "
                f"- \"{sampled_posts[9]}\" "
            )
        prompt = (
            f"Your username is {user_name}. You are a {gender} and your age is in the interval {age_interval}. You are an "
            f"american citizen, born and raised in {state_of_origin} and your ethnicity is {ethnicity}. {religious_part}. "
            f"{interests_part}. Your political orientation is {political_view}. You have a profile on a loosely "
            f"moderated social network platform where people can freely express their feelings without risking to be "
            f"banned. The bio of your profile is {user_bio}. {job_part}. "
            f"Write the next post you would publish. The post can include strong, unfiltered "
            f"opinions, inflammatory language, or even hate speech if it fits your persona. In the post, you {style}. "
            f"{few_shot_part} "
            f"OUTPUT INSTRUCTIONS: \n"
            "* The post must not be longer than 100 words; "
            "* The post doesn't contain multimedia content; "
            "* The expected output is a JSON dictionary with this structure: {\"response\": <post>}; "
            "* Your answer must only contain the dictionary and nothing else."
        )
        print(style, flush=True)
        while not ok and counter_not_ok<3:
            user_posts = generate_posts(model=model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=1000,
                                        device=device)
            try:
                matches = re.findall(r'\{"response"[^{}]*\}', user_posts, re.DOTALL)
                print(matches, flush=True)
                if matches:
                    print("ok")
                    d_copy = d.copy()
                    d_copy["format_style"] = style
                    d_copy["level"] = post_level
                    d_copy["posts"] = json.loads(matches[-1])["response"]        #I use the -1 index because sometimes the LLM puts example dictionaries in the answer, putting the actual one as the last
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
    print(f"TIME SPENT: {(time.time()-now)} seconds")
    return ld


def main(output_fname, n_posts, model_path, accelerator, users_profiles_path=None, already_created_posts=None, real_posts_path=None):
    present_users = []
    real_posts = None

    base, ext = os.path.splitext(output_fname)
    process_output_file = f"{base}_proc{accelerator.process_index}{ext}"
    print(process_output_file)
    if accelerator.is_main_process:
        print(f"\nStarting computing file: {users_profiles_path}")
        print(f"Number of processes: {accelerator.num_processes}\n")

    if already_created_posts:
        posts = pd.read_csv(already_created_posts)
        present_users = list(posts.drop_duplicates(subset="account_id")["account_id"])

    model, tokenizer = load_model_and_tokenizer(model_path=model_path, load_in_4bit=True, accelerator=accelerator)

    if real_posts_path:
        real_posts = pd.read_csv(real_posts_path)
        ids_to_remove = real_posts[real_posts['content'].str.split().str.len() >= 40]["id"].tolist()
        real_posts = real_posts[~real_posts['id'].isin(ids_to_remove)]

    # Provide a csv containing the user descriptions and create the posts from there
    df = pd.read_csv(users_profiles_path)[:10]
    records = df.to_dict('records')

    with accelerator.split_between_processes(records) as shard:
        file_exists = os.path.isfile(process_output_file)
        with open(process_output_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            print("Creating posts...")
            for i, row in enumerate(shard):
                print(f"[proc {accelerator.process_index}] {i}/{len(shard)}")
                if row["account_id"] not in present_users:
                    #print(row["account_id"], flush=True)
                    d = create_user_moody_prompt(user_id=row["account_id"], user_name=row["username"], user_bio=row["user_bio"],
                                                 state_of_origin=row["state_of_origin"], gender=row["gender"], age_interval=row["age_interval"],
                                                 political_view=row["political_leaning"], user_interests=row["interests"],
                                                 job=row["profession"], ethnicity=row["ethnicity"], religion=row["religion"],
                                                 model=model, tokenizer=tokenizer, n_posts=n_posts, real_posts=real_posts,
                                                 device=accelerator.device)
                    for e in d:
                        writer.writerow(e)
                    f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic users profiles")
    # Args Sorted alphabetically
    parser.add_argument("--users_profiles_path", type=str, default="synthetic_user_profiles.csv",
                        help="Path to the csv file containing the users' profiles from which the posts will be created")
    parser.add_argument("--n_posts", type=int, default=None, help="How many posts to create per user")
    parser.add_argument("--already_created_posts", type=str, default=None,
                        help="Path to the csv file containing the posts that have already been created. Set it to "
                             "complete the dataset, if some users were not generated or there were format errors")
    parser.add_argument("--output_fname", type=str, default="foo.csv",
                        help="Path where the output will be saved. The output can either be the user descriptions, or "
                             "the user descriptions together with the synthetic posts")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the LLM")
    parser.add_argument("--real_posts_path", type=str, default=None, help="Path to the csv file containing the example posts")
    args = parser.parse_args()

    accelerator = Accelerator()
    main(users_profiles_path=args.users_profiles_path, output_fname=args.output_fname, already_created_posts=None,
         n_posts=args.n_posts, model_path=args.model_path, real_posts_path=args.real_posts_path,
         accelerator=accelerator)    #args.already_created_posts

# python user_creation.py ---users_profiles_path synthetic_user_profiles.csv --output_fname synthetic_posts.csv --model_path /leonardo_scratch/large/userexternal/fbenedet/models/irix12b --adapter_path fine_tuning/lora_Irix/final_lora_adapter

# python user_creation.py --user_n_posts 10 --users_profiles_path synthetic_user_profiles.csv --output_fname synthetic_posts_added.csv --real_posts_path posts_processed.csv --already_created_posts synthetic_posts_irix_fewshot.csv

# llama-server --model LLMs_gguf/Irix-12B-Model_Stock.i1-Q4_K_S.gguf --n-gpu-layers 999 --port 8080