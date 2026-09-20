"""Evaluate dataframe posts using a locally downloaded Qwen3.8-27B.

Model loading reference: https://huggingface.co/Qwen/Qwen3.8-27B
"""

import pandas as pd
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


from coherence_prompt import prompt


MODEL_PATH = "/leonardo_scratch/large/userexternal/fbenedet/models/qwen38_27B"  # Local model directory on the server.
MAX_NEW_TOKENS = 512  # Adjust for the response format specified in your prompt.


def load_model(model_path=MODEL_PATH):
    """Load the processor and model once, distributing weights automatically."""
    if not model_path:
        raise ValueError("Set MODEL_PATH to the downloaded model directory.")

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(model_path, dtype="auto", device_map="auto", local_files_only=True)
    model.eval()
    return processor, model


@torch.inference_mode()
def evaluate_coherence(post, processor, model, bio, nationality, state, gender, ethnicity, religion, political_leaning,
                       interests, age_interval, profession, instruction, max_new_tokens=MAX_NEW_TOKENS):
    """Return the generated evaluation without imposing a label schema."""
    messages = [
        {"role": "system", "content": prompt.format(bio, nationality, state, gender, ethnicity, religion, political_leaning,
                                                    interests, age_interval, profession, instruction)},
        {"role": "user", "content": [{"type": "text", "text": post}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    # Decode only the response, excluding the prompt tokens.
    response_tokens = output[0, inputs["input_ids"].shape[-1]:]
    return processor.decode(response_tokens, skip_special_tokens=True).strip()


def label_coherence(dataframe, model_path=MODEL_PATH):

    processor, model = load_model(model_path)
    evaluations = []
    for i, r in dataframe.iterrows():
        post_id = r["id"]
        bio = r["user_bio"]
        nationality = r["nationality"]
        state_of_origin = r["state_of_origin"]
        gender = r["gender"]
        ethnicity = r["ethnicity"]
        religion = r["religion"]
        political_leaning = r["political_leaning"]
        interests = r["interests"]
        age_interval = r["age_interval"]
        profession = r["profession"]
        format_style =r["format_style"]
        post = r["content"]

        coherence = evaluate_coherence(post=str(post), processor=processor, model=model, bio=bio, nationality=nationality,
                                       state=state_of_origin, gender=gender, ethnicity=ethnicity, religion=religion,
                                       political_leaning=political_leaning, interests=interests, age_interval=age_interval,
                                       profession=profession, instruction=format_style)
        coherence["post_id"] = post_id
        evaluations.append(coherence)

    result = pd.DataFrame(evaluations)
    return result


if __name__ == "__main__":
    synthetic_df_path = "/leonardo/home/userexternal/fbenedet/labeling_gab/synthetic_df.tsv"
    synthetic_df = pd.read_csv(synthetic_df_path, sep="\t")
    coherence_df = label_coherence(dataframe=synthetic_df, text_column="text", output_column="coherence_evaluation")
    pass
