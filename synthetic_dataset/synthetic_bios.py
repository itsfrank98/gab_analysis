import argparse
import pandas as pd
import json
import os
from openai import OpenAI, BadRequestError
from tqdm import tqdm


def main(num_samples, epochs, load, hateful, dst_dir):
    bios_list = []
    df_notes = pd.read_csv("../dataset/users_follower_followee.csv")
    base_url = "http://127.0.0.1:1234/v1"
    client = OpenAI(
        base_url=base_url,
        api_key="OPENAI_API_KEY",
    )

    for i in tqdm(range(epochs)):
        if load:
            if os.path.exists(f"bios/{i}.csv"):
                sampled_df = pd.read_csv(f"bios/{i}.csv")
        else:
            if hateful:
                hateful_keywords = ["nigger", "niggers", "faggot", "faggots", "jew", "racist", "kkk", "heil hitler",
                                    "hitler was right", "anti-immigrant"]
                df_notes = df_notes[df_notes["note"].apply(lambda x: any(w in str(x).lower() for w in hateful_keywords))]
            sampled_df = df_notes.sample(num_samples)
            #sampled_df.to_csv(f"bios/{i}.csv")
        #df_notes = df_notes.drop(sampled_df["Unnamed: 0"])

        if not os.path.exists(f"bios/{i}.json"):
            notes_list = sampled_df["note"].to_list()
            prompt = (f"I will now provide you a list of {num_samples} profile bios found on the gab.com social network. "
                      "Please, look at them and write 10 bios that follow a similar style, tone and opinions expressed "
                      "in these bios. Please sample from the tail of the distribution "
                      f"such that the probability of each response is less than or equal to 0.2. You can use offensive and insulting slurs, but not too many times.\n"
                      f"{notes_list}"
                      f"\nOUTPUT INSTRUCTIONS: \n"
                        "* Return the bios in list format. Don't write anything else. The output should "
                        "have the following structure: [\"...\", \"...\", \"...\"]")
            try:
                resp = client.chat.completions.create(
                    model="local-model",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0
                )
            except BadRequestError as e:
                print(e)
            try:
                bios_list += json.loads(resp.choices[0].message.content)
                with open(f"{dst_dir}/{i}.json", 'w', encoding="utf-8") as f:
                    json.dump(bios_list, f, indent=2)
            except json.decoder.JSONDecodeError:
                print("\nJSON error goddammit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, help="Number of samples to consider")
    parser.add_argument("--epochs", type=int, help="Number of samples to consider")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--hateful", action="store_true")
    parser.add_argument("--dst_dir", type=str)

    args = parser.parse_args()
    os.makedirs(args.dst_dir, exist_ok=True)
    main(args.num_samples, args.epochs, args.load, args.hateful, args.dst_dir)

