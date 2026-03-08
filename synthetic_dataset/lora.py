import os.path
import random
import torch
from fontTools.ttLib.tables.ttProgram import instructions
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
import pandas as pd
import json
from tqdm import tqdm

def lora():
    model_id = "MuXodious/Snowpiercer-15B-v4-absolute-heresy"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # use float16 if GPU doesn’t support bf16
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # keep minimal for memory
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        output_dir="./snowpiercer-lora",
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    print(training_args)

def preprocess_for_lora(df, content_field, ):
    d = {}
    instructions_list = ["Write a post for the gab.com social network", "Write a post you would publish on the gab.com website",
                         "Write a post expressing your opinions on gab.com"]
    for i, r in tqdm(df.iterrows()):
        text = r[content_field]
        if 300 > len(text.split(" ")) > 5:
            d[i] = {"instruction": random.choice(instructions_list), "response": f"{text}"}

    with open(dst_path, "w") as f:
        json.dump(d, f, indent=4)

def convert_to_snowpiercer_format(instructions):
    system_prompt = "You are a helpful assistant that writes social media posts for the gab.com platform."
    formatted = []
    for instruction in instructions:
        prompt = instructions[instruction]["instruction"]
        response = instructions[instruction]["response"]
        chatml_text = (
            "<|im_start|>system\n"
            f"{system_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{response}\n"
            "<|im_end|>"
        )
        formatted.append(chatml_text)
    return formatted


if __name__ == "__main__":
    df = pd.read_csv("../posts_concat_kirk_processed.csv")
    dst_path = "lora_dataset.json"
    content_field = "content"
    df = df.dropna(subset=[content_field])
    if not os.path.exists(dst_path):
        preprocess_for_lora(df, content_field)
    with open(dst_path, "r") as f:
        instructions = json.load(f)

    formatted_instructions = convert_to_snowpiercer_format(instructions)

    dataset = lo
    lora()


