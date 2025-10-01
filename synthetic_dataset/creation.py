from openai import OpenAI
import json
import numpy as np
import pandas as pd
import os
import time

now = time.time()
os.makedirs("users", exist_ok=True)
base_url = "http://127.0.0.1:1234/v1"
client = OpenAI(
    base_url=base_url,
    api_key="OPENAI_API_KEY",
)

radicalization_levels = np.arange(0, 11) * 10
ages = np.arange(16, 61)
number_of_posts = np.arange(5, 11)

radicalization_types = ["nazism, far right", "fascism, far right", "white supremacy", "communism, far left",
                        "socialism, left", "xenophobia, far right", "authoritarianism", "neonazism, far right",
                        "neofascism, far right", "racism, far right", "antisemitism", "republican, right",
                        "democrat, left", "black panthers, far left", "antifa, far left", "salafi jihadism, religion",
                        "islamic fundamentalism, religion", "libertarian", "progressist"]
language_registers = ["casual", "formal", "aggressive"]
ethnicities = ["caucasian", "asian", "hispanic", ""]
interests = ["basket", "soccer", "tennis", "american football", "reading", "sports", "books", "travelling", "cooking",
             "gardening", "politics", "technology", "self improvement", "history", "science", "animals", "rap music",
             "classical music", "pop music"]

professions_under_25 = ["waiter", "student", "unemployed", "baby sitter"]
professions_over_25 = ["unemployed", "nurse", "vet", "driver", "janitor", "farmer", "mason", "plumber", "hairdresser"]

post_lengths = ["max 10 words", "10-20 words", "30–50 words", "50-70 words"]

df_notes = pd.read_csv("processed_flw.csv")

for j in range(2):
    user_rad_level = np.random.choice(radicalization_levels)
    user_rad_type = np.random.choice(radicalization_types)
    user_register = np.random.choice(language_registers)
    n_hobbies = np.random.randint(1, 4)
    user_interests = np.random.choice(interests, size=n_hobbies, replace=False)
    user_interests = ", ".join(user_interests)
    user_age = np.random.choice(ages)
    if user_age < 25:
        user_profession = np.random.choice(professions_under_25)
    else:
        user_profession = np.random.choice(professions_over_25)

    user_gender = np.random.choice(["male", "female"])
    sampled_row = df_notes.sample(n=1)
    df_notes = df_notes.drop(sampled_row.index)
    user_name = sampled_row["username"].squeeze()
    user_description = sampled_row["note"].squeeze()
    user_description = ("As they say #HitlerwasRight. You can't vote evil out of office they all need to die, like alot. "
                       "Just a dude who's been pushed to far by the corrupt system and tried of hiding what's really going on."
                       "Total Jew/Nigger/Faggot Death")

    user_n_posts = np.random.choice(number_of_posts)    # How many posts will be generated for the user
    #user_posts_lengths = np.random.choice(post_lengths, user_n_posts, replace=True)
    #post_length_instruction = [f"Post {n+1}: {user_posts_lengths[n]}" for n in range(user_n_posts)]
    #post_length_instruction = "\n ".join(post_length_instruction)

    prompt = (
        "Take this profile description in JSON format:"
        "{\n"
        f"username: '{user_name}'\n"
        f"profile_description: '{user_description}'\n"
        f"nationality: 'american'\n"
        f"gender: '{user_gender}'\n"
        f"political/social/religious views: '{user_rad_type}'\n"
        f"interests: '{user_interests}'\n"
        f"age: '{user_age}'\n"
        f"radicalization_level: '{user_rad_level}/100'\n"
        f"language_register: '{user_register}'\n"
        f"profession: '{user_profession}'"
        "}\n"
    )
    print(prompt)
    print("\n")
    #print(post_length_instruction)
    print("\n")
    resp = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Generate n={user_n_posts} social media posts that {user_name} would write on "
                                        f"gab.com. It is a social media platform that allows people to freely express "
                                        f"their feelings without risking of being banned. \n"
                                        "In the posts, the user should talk about his/her passions, and also express "
                                        "his/her feelings about society, politics and religion. \n"
                                        f"OUTPUT INSTRUCTIONS: \n"
                                        "* Write only textual posts, without multimedia content."
                                        "* The output should be a list of posts in plain text. Feel free to use "
                                        "hashtags or emojis. Use them in different parts of the post.\n"
                                        #f"* Follow these instructions for the posts lengths: {post_length_instruction}.\n"
                                        f"Return the posts in JSON format. Don't write anything else. The output should "
                                        "have the following structure: {\"response\": [\"post1\", \"post2\", ..., \"post_n\"]}"}
        ],
        temperature=0.9
    )
    user_posts = resp.choices[0].message.content
    post_list = json.loads(user_posts)["response"]
    with open(f"users/{j}", 'w', encoding="utf-8") as f:
        f.write(prompt + "\n")
        for p in post_list:
            f.write(p + "\n")

print(time.time()-now)


