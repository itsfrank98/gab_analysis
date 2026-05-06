from openai import OpenAI, BadRequestError
import pandas as pd
import json

usernames = pd.read_csv("../dataset/users_follower_followee.csv")["username"].tolist()

base_url = "http://127.0.0.1:1234/v1"
client = OpenAI(
    base_url=base_url,
    api_key="OPENAI_API_KEY",
)

l = []
step = 50
n_steps = 10000
i = 0
while i < n_steps:
    print(f"{i}/{n_steps}")
    sub_df = usernames[i:i+step]
    prompt = (f"I am going to provide you a list containing {step} usernames from an online social network platform. I"
              "want you to take them as inspiration to generate 10 synthetic usernames. \n"
              "OUTPUT INSTRUCTIONS: Return the output as a list. Don't write anything else. This is an example of what "
              "your answer should look like: \n"
              "[\"username1\", \"username2\", \"username3\", ..., \"username10\"]"
    )
    try:
        resp = client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9
        )
    except BadRequestError as e:
        print(e)
    try:
        l += json.loads(resp.choices[0].message.content)
        with open(f"usernames.json", 'w', encoding="utf-8") as f:
            json.dump(l, f, indent=4)
    except json.decoder.JSONDecodeError:
        print("JSON error goddammit")
    i += step
