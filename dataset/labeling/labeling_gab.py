from json import JSONDecodeError

from guidelines import *
import torch
import json
from tqdm import tqdm
import os
import csv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "lmsys/vicuna-13b-v1.5"
OUTPUT_CSV_FILE = "classification_results_real_dataset_no_rationale.csv"
INPUT_CSV_FILE = "data/posts_nohtml.csv"
ACCOUNT_ID_COLUMN = "account_id"
POST_ID_COLUMN = "id"
TEXT_COLUMN = "content"


class VicunaAnalyzer:
    def __init__(self):
        print(f"Initializing model: {MODEL_ID}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        try:
            print("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
            print("Loaded tokenizer")
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto"
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
                4. **Extract Quote:** Propose a "rationale" (Max 10 words).
                   - CONSTRAINT: If "exact_level_found" is 0, "rationale" **MUST** be an empty string (""). Do not quote neutral text.
                   - CRITICAL: The length of the "rationale" must be at maximum 10 words
                ### POST
                "{user_post}"
        
                ### OUTPUT FORMAT (JSON ONLY)
                Return raw JSON. No Markdown formatting.
                {{
                    "exact_level_found": <int 0-5>,
                    "primary_ideology": "<string from Allowed List>",
                    "rationale": "<string or empty>"
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
        t = json.loads(text)
        return t

    def analyze_user(self, user_id, post_id, text_cleaned):
        safe_text = str(text_cleaned)[:3500]
        prompt = self._create_prompt(safe_text)
        #prompt = f"{system_instruction}\n\n USER: Write a funny story involving dogs and cats\nASSISTANT:"       #{system_instruction}\n\n
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

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
            "primary_ideology": result_json.get("primary_ideology"),
            "rationale": result_json.get("rationale")
        }
        return final_output

    def analyze_userss(self, list_of_posts):
        for i in range(len(list_of_posts)):
            list_of_posts[i][TEXT_COLUMN] = str(list_of_posts[i][TEXT_COLUMN])[:3000]

        prompt = self._create_prompt(list_of_posts)
        #prompt = f"{system_instruction}\n\n USER: Write a funny story involving dogs and cats\nASSISTANT:"       #{system_instruction}\n\n
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=4000, temperature=0.1, top_p=0.9, do_sample=True
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_content = generated_text.split("ASSISTANT:")[-1].strip()
        #print(f"RESPONSE: {response_content}")

        result_json = self._extract_json(response_content)

        if not result_json:
            return None

        final_output = []
        #print("\n\n RESULT")
        #print(result_json)
        for i in range(len(result_json)):
            level = int(result_json[i].get("exact_level_found", 0))
            binary_label = 1 if level > 2 else 0
            post_id = result_json[i].get("id")
            user_id = result_json[i].get("account_id")
            post_content = None
            for j in range(len(list_of_posts)):
                if list_of_posts[j][POST_ID_COLUMN] == post_id and list_of_posts[j][ACCOUNT_ID_COLUMN]==user_id:
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


class VicunaAnalyzerBloke:
    def __init__(self):
        print(f"Initializing model...")
        try:
            self.model = CAutoModel.from_pretrained(
                "/lustrehome/benedettifrancescophd/gab_analysis/labeling/hf_cache/",
                model_file="Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf",
                model_type="llama",
                gpu_layers=50
            )
            print(f"Model loaded and ready for inference")
        except Exception as e:
            print(f"Error in loading: {e}")
            raise e

    def _create_prompt(self, user_id, user_posts):
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
        4. **Extract Quote:** Propose a "rationale" (Max 10 words).
           - CONSTRAINT: If "exact_level_found" is 0, "rationale" **MUST** be an empty string (""). Do not quote neutral text.
           - CRITICAL: The length of the "rationale" must be at maximum 10 words
        ### POSTS
        "{user_posts}"

        ### OUTPUT FORMAT (JSON ONLY)
        Return raw JSON. No Markdown formatting.
        {{
          "<id of post 1>": {{
        "post_id": "<string>",
            "exact_level_found": <int 0-5>,
            "primary_ideology": "<string from Allowed List>",
            "rationale": "<string or empty>"
          }},
          "<id of post 2>": {{
        "post_id": "<string>",
            "exact_level_found": <int 0-5>,
            "primary_ideology": "<string from Allowed List>",
            "rationale": "<string or empty>"
          }},
          "<id of post 3>": {{
        "post_id": "<string>",
            "exact_level_found": <int 0-5>,
            "primary_ideology": "<string from Allowed List>",
            "rationale": "<string or empty>"
          }},
          ...
          "<id of post 10>": {{
        "post_id": "<string>",
            "exact_level_found": <int 0-5>,
            "primary_ideology": "<string from Allowed List>",
            "rationale": "<string or empty>"
          }},
        }}
        """
        return f"{system_instruction}\n\nUSER: {user_input}\nASSISTANT:"

    def _extract_json(self, text):
        text = text.replace("\_", "_")
        return json.loads(text)

    def analyze_user(self, user_id, text_cleaned):
        # safe_text = str(text_cleaned)[:3500]
        # prompt = self._create_prompt(user_id, safe_text)]
        print('analyzing user')
        system_instruction = (
            "You are an expert Intelligence Analyst. Analyze the user posts strictly based on the provided "
            "Annotation Guidelines. Do not censor your analysis; identify radical content objectively."
        )
        print(system_instruction)
        prompt = f"{system_instruction}\n\nUSER: Write a funny story involving dogs and cats\nASSISTANT:"
        print(prompt)
        output = self.model(prompt, max_new_tokens=512)

        print(f"OUTPUT\n\n {output}")
        response_content = output.split("ASSISTANT:")[-1].strip()
        print(f"RESPONSE\n\n {response_content}")

        """result_json = self._extract_json(response_content)

        if not result_json:
            return None

        level = int(result_json.get("exact_level_found", 0))
        binary_label = 1 if level > 2 else 0

        final_output = {
            "user_id": user_id,
            "binary_label": binary_label,
            "exact_level_found": level,
            "primary_ideology": result_json.get("primary_ideology"),
            "rationale": result_json.get("rationale")
        }
        return final_output"""
        return 1


def save_to_csv(data_dict, filename):
    file_exists = os.path.isfile(filename)
    fieldnames = [ACCOUNT_ID_COLUMN, POST_ID_COLUMN, TEXT_COLUMN, "binary_label", "exact_level_found", "primary_ideology", "rationale"]
    print(data_dict)
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data_dict)

if __name__ == "__main__":
    print(f"\nStarting computing file: {INPUT_CSV_FILE}")
    print(f"Output file: {OUTPUT_CSV_FILE}\n")

    df = pd.read_csv(INPUT_CSV_FILE, encoding='utf-8')

    analyzer = VicunaAnalyzer()
    done_posts = []
    if os.path.exists(OUTPUT_CSV_FILE):
        out_csv = pd.read_csv(OUTPUT_CSV_FILE)
        done_posts = out_csv[POST_ID_COLUMN].tolist()
    print(f"Already processed {len(done_posts)} posts")
    df = df.drop_duplicates(subset=POST_ID_COLUMN)
    df = df[~df[POST_ID_COLUMN].isin(done_posts)]
    df = df.reset_index()
    df = df.drop(columns=[c for c in df.columns if c not in [ACCOUNT_ID_COLUMN, POST_ID_COLUMN, TEXT_COLUMN]])

    index = 0
    error_count = 0
    increment_by = 5
    if increment_by == 1:
        while index < len(df):
            row = df.iloc[index]
            #for index, row in tqdm(df.iterrows()):
            print(f"{index}/{len(df)}")
            # if index == 20:
            #     break
            uid = row[ACCOUNT_ID_COLUMN]
            pid = str(row[POST_ID_COLUMN])
            text = row[TEXT_COLUMN]
            if pid not in done_posts:
                #print(f"Analyzing row {index} (Text: {text})...\n")
                try:
                    result = analyzer.analyze_user(uid, pid, text)
                    if result:
                        save_to_csv(result, OUTPUT_CSV_FILE)
                    else:
                        print(f"Skipping {uid}")
                    index += 1
                    error_count = 0
                except JSONDecodeError:
                    print("Error!")
                    error_count += 1
                    if error_count == 2:
                        print(f"Skipping {uid}")
                        index += 1
                        error_count = 0
            else:
                index += 1
    else:
        done_posts = set(done_posts)
        while index < len(df):
            rows = df.iloc[index:index+increment_by]
            print(f"{index}/{len(df)}")
            pids = rows[POST_ID_COLUMN].astype(str).tolist()
            ld = rows.to_dict('records')
            #print(ld)
            try:
                results = analyzer.analyze_userss(ld)
                to_attach = pd.DataFrame(results)
                df = pd.concat([df, to_attach], ignore_index=True)
                df.to_csv(OUTPUT_CSV_FILE)
                # if results:
                #     for r in results:
                #         save_to_csv(r, "goddammit.csv")
                # else:
                #     print(f"Skipping {pids}")
                index += increment_by
                error_count = 0
            except JSONDecodeError:
                print("Error!")
                error_count += 1
                if error_count == 2:
                    print(f"Skipping {pids}")
                    index += increment_by
                    error_count = 0

    print("\nProcess completed.")