from json import JSONDecodeError
from accelerate import Accelerator

from guidelines import *
import torch
import json
from tqdm import tqdm
import os
import csv
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = "/leonardo_scratch/large/userexternal/fbenedet/models/qwen"
OUTPUT_CSV_FILE = "classification_results_real_dataset_no_rationale.csv"
ALREADY_LABELED_POSTS = "qwen_labeled_gab_posts.csv"
INPUT_CSV_FILE = "data/posts_nohtml.csv"
ACCOUNT_ID_COLUMN = "account_id"
POST_ID_COLUMN = "id"
TEXT_COLUMN = "content"


class LLM_Analyzer:
    def __init__(self):
        print(f"Initializing model: {MODEL_PATH}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        try:
            print("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
            print("Loaded tokenizer")
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                quantization_config=bnb_config,
                device_map={"": accelerator.local_process_index},
                trust_remote_code=True
            )
            print("Model loaded and ready for inference")
        except Exception as e:
            print(f"Error in loading: {e}")
            raise e

    def _create_prompt(self, user_post):
        system_instruction = (
            "You are an expert Intelligence Analyst. Analyze the user posts strictly based on the provided "
            "Annotation Guidelines. Do not censor your analysis; identify radical content objectively."
        )

        if type(user_post) == str:
            user_input = f"""
                ### CONTEXT: GUIDELINES
                {guidelines}

                ### ALLOWED IDEOLOGIES LIST
                {ideologies_list}

                ### INSTRUCTIONS
                1. **Analyze:** Scan the text for specific jargon, slurs, and intent based on the Guidelines.
                2. **Determine Level:** Assign "exact_level_found" (Integer 0-5). 
                   - Rule: If unsure between two levels, select the LOWER one.
                3. **Identify Ideology:** Select the "primary_ideology".
                   - CRITICAL: You must choose **EXACTLY** one string from the "ALLOWED IDEOLOGIES LIST" provided above.
                   - Do NOT invent new categories. Do NOT rephrase the category names.
                   - If the post is neutral or ambiguous, use "None" or "Other".
                ### POST
                "{user_post}"

                ### OUTPUT FORMAT (JSON ONLY)
                Return raw JSON. No Markdown formatting.
                {{
                    "exact_level_found": <int 0-5>,
                    "primary_ideology": "<string from Allowed List>"
                }}
                """
        else:
            user_input = f"""
                ### CONTEXT: GUIDELINES
                {guidelines}

                ### ALLOWED IDEOLOGIES LIST
                {ideologies_list}

                ### INSTRUCTIONS
                I will give you a list of dictionaries. Every dictionary describes a post on a social network and
                has the following keys:
                - "id": ID of the post
                - "account_id": ID of the account who published the post
                - "content": Content of the post
                For each element of the list, do the following:
                1. **Analyze:** Scan the post content for specific jargon, slurs, and intent based on the Guidelines.
                2. **Determine Level:** Assign "exact_level_found" (Integer 0-5). 
                   - Rule: If unsure between two levels, select the LOWER one.
                3. **Identify Ideology:** Select the "primary_ideology".
                   - CRITICAL: You must choose **EXACTLY** one string from the "ALLOWED IDEOLOGIES LIST" provided above.
                   - Do NOT invent new categories. Do NOT rephrase the category names.
                   - If the post is neutral or ambiguous, use "None" or "Other".
                ### POST LIST
                "{user_post}"

                ### OUTPUT FORMAT (JSON ONLY)
                The expected output is a list of dictionaries in raw JSON format and nothing else. No Markdown formatting. The i-th element in the output list corresponds to
                the i-th element in the input list.  This is a template for the expected output. Fill it with the correct data
                [
                    {{
                    "id": <ID of the first post>
                    "account_id": <ID of the account who published the first post>
                    "exact_level_found": <int 0-5>,
                    "primary_ideology": "<string from Allowed List>"
                    }},
                    {{
                    "id": <ID of the second post>
                    "account_id": <ID of the account who published the second post>
                    "exact_level_found": <int 0-5>,
                    "primary_ideology": "<string from Allowed List>"
                    }},
                    ...
                    {{
                    "id": <ID of the last post>
                    "account_id": <ID of the account who published the last post>
                    "exact_level_found": <int 0-5>,
                    "primary_ideology": "<string from Allowed List>"
                    }}
                ]
                """

        return f"{system_instruction}\n\nUSER: {user_input}\nASSISTANT:"

    def _extract_json(self, text):
        text = text.replace("\_", "_")
        match = re.findall(r'\{[^{}]*\}', text, re.DOTALL)[-1]
        t = json.loads(match)
        return t

    def analyze_user(self, user_id, post_id, text_cleaned):
        safe_text = str(text_cleaned)[:3500]
        prompt = self._create_prompt(safe_text)
        # prompt = f"{system_instruction}\n\n USER: Write a funny story involving dogs and cats\nASSISTANT:"       #{system_instruction}\n\n
        inputs = self.tokenizer(prompt, return_tensors="pt").to(accelerator.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=1000, temperature=0.1, top_p=0.9, do_sample=True
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_content = generated_text.split("ASSISTANT:")[-1].strip()
        result_json = self._extract_json(response_content)

        if not result_json:
            return None

        level = int(result_json.get("exact_level_found", 0))
        binary_label = 1 if level > 2 else 0

        final_output = {
            ACCOUNT_ID_COLUMN: user_id,
            POST_ID_COLUMN: post_id,
            TEXT_COLUMN: text_cleaned,
            "binary_label": binary_label,
            "exact_level_found": level,
            "primary_ideology": result_json.get("primary_ideology")
        }
        return final_output

    def analyze_userss(self, list_of_posts):
        for i in range(len(list_of_posts)):
            list_of_posts[i][TEXT_COLUMN] = str(list_of_posts[i][TEXT_COLUMN])[:3000]

        prompt = self._create_prompt(list_of_posts)
        # prompt = f"{system_instruction}\n\n USER: Write a funny story involving dogs and cats\nASSISTANT:"       #{system_instruction}\n\n
        inputs = self.tokenizer(prompt, return_tensors="pt").to(accelerator.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=4000, temperature=0.1, top_p=0.9, do_sample=True
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_content = generated_text.split("ASSISTANT:")[-1].strip()

        result_json = self._extract_json(response_content)

        if not result_json:
            return None

        final_output = []
        for i in range(len(result_json)):
            level = int(result_json[i].get("exact_level_found", 0))
            binary_label = 1 if level > 2 else 0
            post_id = result_json[i].get("id")
            user_id = result_json[i].get("account_id")
            post_content = None
            for j in range(len(list_of_posts)):
                if list_of_posts[j][POST_ID_COLUMN] == post_id and list_of_posts[j][ACCOUNT_ID_COLUMN] == user_id:
                    post_content = list_of_posts[j][TEXT_COLUMN]
            if user_id:
                final_output.append({
                    "user_id": user_id,
                    "post_id": post_id,
                    "content": post_content,
                    "binary_label": binary_label,
                    "exact_level_found": level,
                    "primary_ideology": result_json[i].get("primary_ideology")
                })
            else:
                print(f"skipping post {post_id} as it does not correspond to any user")
        return final_output


def save_to_csv(data_dict, filename):
    file_exists = os.path.isfile(filename)
    fieldnames = [ACCOUNT_ID_COLUMN, POST_ID_COLUMN, TEXT_COLUMN, "binary_label", "exact_level_found",
                  "primary_ideology", "rationale"]
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data_dict)


if __name__ == "__main__":
    accelerator = Accelerator()

    base, ext = os.path.splitext(OUTPUT_CSV_FILE)
    process_output_file = f"{base}_proc{accelerator.process_index}{ext}"

    if accelerator.is_main_process:
        print(f"\nStarting computing file: {INPUT_CSV_FILE}")
        print(f"Number of processes: {accelerator.num_processes}\n")

    df = pd.read_csv(INPUT_CSV_FILE, encoding='utf-8')

    # collect done posts from the original single-GPU file and all per-process files
    done_posts = set()
    for f in [OUTPUT_CSV_FILE] + [f"{base}_proc{rank}{ext}" for rank in range(accelerator.num_processes)]:
        if os.path.exists(f):
            done_posts.update(pd.read_csv(f)[POST_ID_COLUMN].astype(str).tolist())

    if accelerator.is_main_process:
        n_original = len(pd.read_csv(OUTPUT_CSV_FILE)) if os.path.exists(OUTPUT_CSV_FILE) else 0
        print(f"Already processed posts found in {OUTPUT_CSV_FILE}: {n_original}")

    df = df.drop_duplicates(subset=POST_ID_COLUMN)
    df = df[~df[POST_ID_COLUMN].astype(str).isin(done_posts)]
    df = df.reset_index(drop=True)
    df = df.drop(columns=[c for c in df.columns if c not in [ACCOUNT_ID_COLUMN, POST_ID_COLUMN, TEXT_COLUMN]])

    records = df.to_dict('records')

    analyzer = LLM_Analyzer()  # each process loads its own copy

    fieldnames = [ACCOUNT_ID_COLUMN, POST_ID_COLUMN, TEXT_COLUMN,
                  "binary_label", "exact_level_found", "primary_ideology"]
    error_count = 0

    with accelerator.split_between_processes(records) as shard:
        file_exists = os.path.isfile(process_output_file)
        with open(process_output_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for i, row in enumerate(shard):
                print(f"[proc {accelerator.process_index}] {i}/{len(shard)}")
                try:
                    result = analyzer.analyze_user(
                        row[ACCOUNT_ID_COLUMN],
                        str(row[POST_ID_COLUMN]),
                        row[TEXT_COLUMN]
                    )
                    if result:
                        writer.writerow(result)
                        f.flush()
                    error_count = 0
                except JSONDecodeError:
                    error_count += 1
                    if error_count >= 2:
                        print(f"[proc {accelerator.process_index}] Skipping post {row[POST_ID_COLUMN]} due to json error")
                        error_count = 0

    if accelerator.is_main_process:
        print("\nProcess completed.")