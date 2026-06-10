import argparse
import csv
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import torch

from accelerate import Accelerator
from guidelines import *
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


ACCOUNT_ID_COLUMN = "account_id"
POST_ID_COLUMN = "id"
TEXT_COLUMN = "content"



class LLM_Analyzer:
    def __init__(self, model_path):
        print(f"Initializing model: {model_path}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        try:
            print("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
            print("Loaded tokenizer")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map={"": accelerator.local_process_index},
                trust_remote_code=True,
                local_files_only=True
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_csv", required=True, type=str)
    parser.add_argument("--labeled_posts", required=True, type=str)
    args = parser.parse_args()

    model_path = args.model_path
    output_dir = args.output_dir
    data_dir = args.data_dir
    output_csv_file = os.path.join(output_dir, args.output_csv)
    already_labeled_posts = os.path.join(output_dir, args.labeled_posts)
    input_csv_file = os.path.join(data_dir, args.input_csv)
    labeled_posts_path = os.path.join(output_dir, args.labeled_posts)

    accelerator = Accelerator()

    base, ext = os.path.splitext(output_csv_file)
    process_output_file = f"{base}_proc{accelerator.process_index}{ext}"

    if accelerator.is_main_process:
        print(f"\nStarting computing file: {input_csv_file}")
        print(f"Number of processes: {accelerator.num_processes}\n")

    df = pd.read_csv(input_csv_file, encoding='utf-8')

    # collect done posts from the original single-GPU file and all per-process files
    done_posts_ids = set()
    if os.path.exists(already_labeled_posts):
        done_posts = pd.read_csv(already_labeled_posts, encoding='utf-8')
        done_posts_ids.update(done_posts[POST_ID_COLUMN].astype(str).tolist())
    print("ALREADY LABELED POSTS SOURCE: ", already_labeled_posts)
    print(f"Already processed posts found: {len(done_posts_ids)}")

    df = df.drop_duplicates(subset=POST_ID_COLUMN)
    df = df[~df[POST_ID_COLUMN].astype(str).isin(done_posts_ids)]
    df = df.reset_index(drop=True)
    print(f"TOTAL NUMBER OF POSTS TO PROCESS: {len(df)}")
    df = df.drop(columns=[c for c in df.columns if c not in [ACCOUNT_ID_COLUMN, POST_ID_COLUMN, TEXT_COLUMN]])

    records = df.to_dict('records')

    analyzer = LLM_Analyzer(model_path=model_path)  # each process loads its own copy

    fieldnames = [ACCOUNT_ID_COLUMN, POST_ID_COLUMN, TEXT_COLUMN, "binary_label", "exact_level_found", "primary_ideology"]
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
                except json.JSONDecodeError:
                    error_count += 1
                    if error_count >= 2:
                        print(f"[proc {accelerator.process_index}] Skipping post {row[POST_ID_COLUMN]} due to json error")
                        error_count = 0

    df = pd.read_csv("gab_posts_labeled_qwen.csv")
    df['exact_level_found'].value_counts().sort_index().plot(kind='pie', autopct='%1.1f%%', ylabel='')
    plt.title('Radicalization level distribution')
    plt.show()
    if accelerator.is_main_process:
        print("\nProcess completed.")
