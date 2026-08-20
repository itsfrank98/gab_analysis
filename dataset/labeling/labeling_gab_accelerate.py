from json import JSONDecodeError
from accelerate import Accelerator
from llama_cpp import Llama
from guidelines import *
import torch
import json
from tqdm import tqdm
import os
import csv
import pandas as pd
import re
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


ACCOUNT_ID_COLUMN = "account_id"
POST_ID_COLUMN = "id"
TEXT_COLUMN = "content"



class LLM_Analyzer:
    def __init__(self, model_path, llama_or_st):
        print(f"Initializing model: {model_path}...")
        self.llama_or_st = llama_or_st

        if llama_or_st == "llama":
            self.model =  Llama(model_path=model_path, n_ctx=131072, n_gpu_layers=99)

        else:
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

        if type(user_post) == str:  # If I am giving a single post
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
                    "primary_ideology": "<string from Allowed List>",
                    "call_for_action": <int 0-4>
                }}
                """

        else:       # If I am giving a list of posts
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

        if self.llama_or_st == "llama":
            output = self.model(
                prompt, max_tokens=1000, temperature=0.1, top_p=0.9, stop=["USER:"]
            )
            response_content = output["choices"][0]["text"].strip()
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(accelerator.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=1000, temperature=0.1, top_p=0.9, do_sample=True
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_content = generated_text.split("ASSISTANT:")[-1].strip()
        try:
            result_json = self._extract_json(response_content)

            if not result_json:
                return None

            level = int(result_json.get("exact_level_found", -1))
            call_for_action = int(result_json.get("call_for_action", -1))

            if call_for_action != -1 and level != -1:
                final_output = {
                    ACCOUNT_ID_COLUMN: user_id,
                    POST_ID_COLUMN: post_id,
                    TEXT_COLUMN: text_cleaned,
                    "exact_level_found": level,
                    "call_for_action": call_for_action,
                    "primary_ideology": result_json.get("primary_ideology")
                }
        except Exception as e:
            final_output = None

        return final_output


if __name__ == "__main__":
    """
    LEO
    --model_path /leonardo_scratch/large/userexternal/fbenedet/models/qwen
    --input_csv posts_nohtml_processed_0_-40k.csv
    --labeled_posts qwen_labeled_gab_posts.csv
    --output_csv classification_results_real_dataset_no_rationale.csv
    
    RECAS
    --model_path /lustrehome/benedettifrancescophd/models/qwen
    --input_csv posts_nohtml_processed_0_-40k.csv
    --labeled_posts qwen_labeled_gab_posts.csv
    --output_csv classification_results_real_dataset_no_rationale.csv
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_csv", required=True, type=str)
    parser.add_argument("--labeled_posts", required=False, default="labeled_posts.csv", type=str)
    parser.add_argument("--llama_or_st", required=True, type=str, choices=["llama", "st"],
                         help="'llama' to load the model with llama_cpp, 'st' to load it with transformers/bitsandbytes")
    args = parser.parse_args()

    model_path = args.model_path
    output_dir = args.output_dir
    data_dir = args.data_dir
    input_csv_file = os.path.join(data_dir, args.input_csv)
    output_csv_file = os.path.join(output_dir, args.output_csv)
    already_labeled_posts = os.path.join(output_dir, args.labeled_posts)

    accelerator = Accelerator()

    base, ext = os.path.splitext(output_csv_file)
    process_output_file = f"{base}_proc{accelerator.process_index}{ext}"

    if accelerator.is_main_process:
        print(f"\nStarting computing file: {input_csv_file}")
        print(f"Number of processes: {accelerator.num_processes}\n")

    if input_csv_file.endswith(".tsv"):
        df = pd.read_csv(input_csv_file, sep="\t", encoding="utf-8")
    elif input_csv_file.endswith(".csv"):
        df = pd.read_csv(input_csv_file, encoding='utf-8')

    # collect done posts from the original single-GPU file and all per-process files
    done_posts_ids = set()
    if os.path.exists(already_labeled_posts):
        if already_labeled_posts.endswith(".tsv"):
            done_posts = pd.read_csv(already_labeled_posts, encoding='utf-8', sep="\t")
        else:
            done_posts = pd.read_csv(already_labeled_posts, encoding='utf-8')
        done_posts_ids.update(done_posts[POST_ID_COLUMN].astype(int).tolist())
    print("ALREADY LABELED POSTS SOURCE: ", already_labeled_posts)
    print(f"Already processed posts found: {len(done_posts_ids)}")

    df = df.drop(columns=[c for c in df.columns if c not in [POST_ID_COLUMN, ACCOUNT_ID_COLUMN, TEXT_COLUMN]])
    df = df.drop_duplicates(subset=POST_ID_COLUMN)
    
    df = df[~df[POST_ID_COLUMN].astype(int).isin(done_posts_ids)]
    df = df.reset_index(drop=True)
    print(f"TOTAL NUMBER OF POSTS TO PROCESS: {len(df)}")
    df = df.drop(columns=[c for c in df.columns if c not in [ACCOUNT_ID_COLUMN, POST_ID_COLUMN, TEXT_COLUMN]])

    records = df.to_dict('records')

    analyzer = LLM_Analyzer(model_path=model_path, llama_or_st=args.llama_or_st)  # each process loads its own copy

    fieldnames = [ACCOUNT_ID_COLUMN, POST_ID_COLUMN, TEXT_COLUMN, "exact_level_found", "call_for_action", "primary_ideology"]
    error_count = 0

    with accelerator.split_between_processes(records) as shard:
        file_exists = os.path.isfile(process_output_file)
        with open(process_output_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", quoting=3, escapechar="\\")
            if not file_exists:
                writer.writeheader()

            for i, row in tqdm(enumerate(shard)):
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
                    if error_count >= 3:
                        print(f"[proc {accelerator.process_index}] Skipping post {row[POST_ID_COLUMN]} due to json error")
                        error_count = 0

    if accelerator.is_main_process:
        print("\nProcess completed.")
