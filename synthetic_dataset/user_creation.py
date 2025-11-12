import argparse

from openai import OpenAI
from tqdm import tqdm
from creation_options import *
import json
import numpy as np
import pandas as pd
import re
import os
import time

now = time.time()
os.makedirs("users", exist_ok=True)
base_url = "http://127.0.0.1:1234/v1"
client = OpenAI(
    base_url=base_url,
    api_key="OPENAI_API_KEY",
)

def create_user(user_id, user_n_posts, j, user_name, user_bio, state_of_origin, gender, ethnicity, religion, political_view,
                user_interests, age, job, create_posts=True):
    d = {
        "profile_id": user_id.format(j),
        "username": user_name,
        "user_bio": user_bio,
        "nationality": 'american',
        "state_of_origin": state_of_origin,
        "gender": gender,
        "ethnicity": ethnicity,
        "religion": religion,
        "political_leaning": political_view.replace("_", "-"),
        "interests": user_interests,
        "age": age,
        "profession": job,
        # "radicalization_level": f"{user_rad_level}/100"
    }

    if create_posts:
        prompt = f"Take this profile description in dictionary format: {d}"
        """
        "{\n"
        f"username: '{user_name}',\n"
        f"user_bio: '{user_bio}',\n"
        f"nationality: 'american',\n"
        f"state_of_origin: {state_of_origin},\n"
        f"gender: '{gender},'\n"
        f"political_leaning: '{political_view}',\n"
        f"interests: '{user_interests}',\n"
        f"age: '{age}',\n"
        f"radicalization_level: '{user_rad_level}/100',\n"
        f"language_register: '{user_register}',\n"
        f"profession: '{job}'"
        "}\n"
        """

        sampled_probability = np.random.choice(np.arange(.1, 1, .1))
        d["probability"] = sampled_probability
        resp = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",
                 "content": f"Generate n={user_n_posts} social media posts that {user_name} would write on "
                            f"gab.com. Gab.com is a social media platform that allows "
                            f"people to freely express their feelings without risking of being banned. \n"
                            "In the posts, the user can discuss about his/her passions, express "
                            "feelings about society, politics and religion. Please sample from the tail of the distribution "
                            f"such that the probability of each post is less than or equal to 0.4. You can use offensive and insulting slurs, but not too many times.\n \n"
                            f"OUTPUT INSTRUCTIONS: \n"
                            "* Write only textual posts, without multimedia content."
                            "* The output should be a list of posts in plain text. Feel free to use "
                            "hashtags or emojis. Use them in different parts of the post.\n"
                            f"Return the posts in JSON format. Don't write anything else. The output should "
                            "have the following structure: {\"response\": [\"post1\", \"post2\", ..., \"post_n\"]}"}
            ],
            temperature=1.0
        )

        user_posts = resp.choices[0].message.content
        try:
            d["posts"] = json.loads(user_posts)["response"]

        except json.decoder.JSONDecodeError:
            print("JSON error goddammit")
            with open(f"{user_name}.txt", "w", encoding="utf-8") as f:
                f.write(user_posts)
            d["posts"] = [""] * 10
    return d

def determine_gender(user_bio):
    for w in female_keywords:
        if re.search(f"\b{w}\b", user_bio):
            return "female"
    for w in male_keywords:
        if re.search(f"\b{w}\b", user_bio):
            return "male"
    return np.random.choice(["female", "male"])

def main(user_n_posts=10, n_of_users=5, output_fname="synthetic_users", src_path=None, create_posts=True,
         bios_path="bios/bios_united.json", bios_afl_path="bios/afl.json", usernames_path="usernames.json"):
    user_id = "p{}"
    dicts_list = []

    if not src_path:
        ages = np.arange(16, 61)

        political_items = list(political_leanings.keys())
        political_probs = list(political_leanings.values())

        ethnicity_items = list(ethnicities.keys())
        ethnicity_probs = list(ethnicities.values())

        with open(usernames_path, "r") as f:
            usernames = json.load(f)
        with open(bios_path, "r", encoding="utf-8") as f:
            bios = json.load(f)
            if type(bios) == list:
                bios = {str(k): bios[k] for k in range(len(bios))}
        with open(bios_afl_path, "r", encoding="utf-8") as f:
            bios_affiliations = json.load(f)

        # These lists have a precise goal. When we generate a user, we first pick his leaning, sampling from the distribution.
        # When we pick the bio, it won't be a completely random sampling. If the sampled leaning is for instance right, or far
        # right, the bio cannot belong to the left or far left leanings. It will be right, far right, unknown or non-political,
        # since the user may have a bio where he doesn't express political views. The same applies for the left leanings. On the
        # other hand, if the leaning is unknown or non-political, we don't want the bio to express (far) right/left views.
        # In this way we generate less profiles that don't make sense

        right_bios_keys = [k for k in list(bios_affiliations.keys()) if bios_affiliations[k] in ["right", "far_right"]]
        left_bios_keys = [k for k in list(bios_affiliations.keys()) if bios_affiliations[k] in ["left", "far_left"]]
        any_bios_keys = [k for k in list(bios_affiliations.keys()) if
                         bios_affiliations[k] in ["unknown", "non_political"]]

        bios_keys_for_right = right_bios_keys + any_bios_keys
        bios_keys_for_left = left_bios_keys + any_bios_keys

        for j in tqdm(range(n_of_users)):
            user_name = np.random.choice(usernames)
            usernames.remove(user_name)
            n_hobbies = np.random.randint(0, 3)
            political_view = np.random.choice(political_items, p=political_probs)
            ethnicity = np.random.choice(ethnicity_items, p=ethnicity_probs)
            if political_view in ["right", "far_right"]:
                user_bio_k = np.random.choice(bios_keys_for_right)
            elif political_view in ["left", "far_left"]:
                user_bio_k = np.random.choice(bios_keys_for_left)
            else:
                user_bio_k = np.random.choice(any_bios_keys)

            user_bio = bios[user_bio_k]
            # We remove the bio that was just picked so it's not sampled again
            for kl in [bios_keys_for_right, bios_keys_for_left, any_bios_keys]:
                if user_bio_k in kl:
                    kl.remove(user_bio_k)

            state_of_origin = np.random.choice(us_states)

            gender = determine_gender(user_bio)

            hobbies_in_bio = []
            for hobby in interests:
                if user_bio.__contains__(hobby):
                    hobbies_in_bio.append(hobby)

            if len(hobbies_in_bio) >= n_hobbies:
                user_interests = ", ".join(hobbies_in_bio)
            else:
                n_hobbies -= len(hobbies_in_bio)
                user_interests = ", ".join(hobbies_in_bio + list(np.random.choice(interests, size=n_hobbies, replace=False)))

            if user_bio.lower().__contains__("christian"):
                religion = np.random.choice(list(christian_religions_norm.keys()), p=list(christian_religions_norm.values()))
            else:
                religion = np.random.choice(list(religions.keys()), p=list(religions.values()))
            age = np.random.choice(ages)
            job = np.random.choice(professions)
            """user_description = ("As they say #HitlerwasRight. You can't vote evil out of office they all need to die, like alot. "
                               "Just a dude who's been pushed to far by the corrupt system and tried of hiding what's really going on."
                               "Total Jew/Nigger/Faggot Death")"""

            #user_n_posts = np.random.choice(number_of_posts)    # How many posts will be generated for the user
            #user_posts_lengths = np.random.choice(post_lengths, user_n_posts, replace=True)
            #post_length_instruction = [f"Post {n+1}: {user_posts_lengths[n]}" for n in range(user_n_posts)]
            #post_length_instruction = "\n ".join(post_length_instruction)

            d = create_user(user_id=user_id, user_name=user_name, user_bio=user_bio, state_of_origin=state_of_origin,
                            gender=gender, ethnicity=ethnicity, political_view=political_view, job=job, j=j, age=age,
                            user_interests=user_interests, religion=religion, user_n_posts=user_n_posts,
                            create_posts=create_posts)
            dicts_list.append(d)
    else:
        # Provide a csv containing the user descriptions and create the posts from there
        df = pd.read_csv(src_path)

        for i, row in tqdm(df.iterrows()):
            d = create_user(user_id=row["profile_id"], user_name=row["username"], user_bio=row["user_bio"],
                            state_of_origin=row["state_of_origin"], gender=row["gender"],
                            political_view=row["political_leaning"], user_interests=row["interests"], age=row["age"],
                            job=row["profession"], j=i, user_n_posts=user_n_posts, ethnicity=row["ethnicity"])
            dicts_list.append(d)

    df = pd.DataFrame(dicts_list)
    if "posts" in df.columns:
        df = df.explode("posts").reset_index(drop=True)
        for col in df.columns[:-1]:
            df.loc[df.duplicated(subset=["profile_id"]), col] = None
    df.to_excel(output_fname + ".xlsx")
    df.to_csv(output_fname + ".csv", errors="ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic users probabilities")
    parser.add_argument("--user_n_posts", type=int, required=False, help="Number of posts to create for each user")
    parser.add_argument("--n_of_users", type=int, required=False, help="Number of users to create")
    parser.add_argument("--src_path", type=str, default=None, help="Path to the csv file containing the "
                                                       "users' profiles from which the posts will be created")
    parser.add_argument("--output_fname", type=str, required=True, help="Path where the output will be saved."
                                                       "The output can either be the user descriptions, or the user "
                                                       "descriptions together with the synthetic posts")
    parser.add_argument("--create_posts", action="store_true", help="If true, creates both the users and "
                                                        "their posts. If false, only creates the users.")
    parser.add_argument("--bios_path", type=str, help="Path to the file containing the bios")
    parser.add_argument("--bios_afl_path", type=str, help="Path to the file containing the bios predicted affiliations")
    parser.add_argument("--usernames_path", type=str, help="Path to the file containing the usernames")
    args = parser.parse_args()
    main(user_n_posts=args.user_n_posts, n_of_users=args.n_of_users, src_path=args.src_path, create_posts=args.create_posts,
         output_fname=args.output_fname, bios_path=args.bios_path, bios_afl_path=args.bios_afl_path,
         usernames_path=args.usernames_path)

    #  python .\user_creation.py --user_n_posts 10 --n_of_users 20 --output_fname users/users_posts --create_posts