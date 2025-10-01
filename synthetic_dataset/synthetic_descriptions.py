from openai import OpenAI, BadRequestError
import pandas as pd
import json
import os
from tqdm import tqdm


df_notes = pd.read_csv("processed_desc.csv")
base_url = "http://127.0.0.1:1234/v1"
client = OpenAI(
    base_url=base_url,
    api_key="OPENAI_API_KEY",
)

for i in tqdm(range(100)):
    if os.path.exists(f"bios/{i}.csv"):
        sampled_df = pd.read_csv(f"bios/{i}.csv")
    else:
        sampled_df = df_notes.sample(50)
        sampled_df.to_csv(f"bios/{i}.csv")
    df_notes = df_notes.drop(sampled_df["Unnamed: 0"])

    if not os.path.exists(f"bios/{i}.json"):
        notes_list = sampled_df["note"].to_list()
        prompt = ("I will now provide you a list of 100 profile bios found online. Please, look at them and write 10 bios "
                  "that follow a similar style, tone and opinions expressed in these bios.\n"
                  f"{notes_list}")
        try:
            resp = client.chat.completions.create(
                model="local-model",
                messages=[
                    #{"role": "system", "content": prompt},
                    {"role": "user", "content": f"{prompt}"
                                                "\nOUTPUT INSTRUCTIONS: \n"
                                                "* Return the bios in list format. Don't write anything else. The output should "
                                                "have the following structure: {\"bio1\": \"...\", \"bio2\": ..., \"bio3\": ..., "
                                                "\"bio4\": ..., \"bio5\": ..., \"bio6\": ..., \"bio7\": ..., \"bio8\": ..., "
                                                "\"bio9\": ..., \"bio10\": ...}"}
                ],
                temperature=0.9
            )
        except BadRequestError as e:
            print(e)
        try:
            bios_dict = json.loads(resp.choices[0].message.content)
            with open(f"bios/{i}.json", 'w', encoding="utf-8") as f:
                json.dump(bios_dict, f, indent=2)
        except json.decoder.JSONDecodeError:
            print("JSON error goddammit")

