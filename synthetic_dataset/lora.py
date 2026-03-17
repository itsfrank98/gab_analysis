import os.path
import random
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import logging
from transformers import GPT2Tokenizer
from trl import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
df = pd.read_csv("../posts_concat_kirk_processed.csv")
MODEL_ID = "openai-community/gpt2"  #"DreadPoor/Irix-12B-Model_Stock"
MIN_WORDS = 5
MAX_WORDS = 300

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT for Irix-12B")

    # Data
    parser.add_argument("--csv_path",    type=str, default="../posts_concat_kirk_processed.csv",
                        help="Path to CSV file containing posts")
    parser.add_argument("--text_column", type=str, default="content",  help="Name of the column with post text")
    parser.add_argument("--model_id",    type=str, default=MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--output_dir",  type=str, default="./irix-12b-lora", help="Directory for checkpoints and final model")

    # LoRA hyperparameters
    parser.add_argument("--lora_r",           type=int,   default=16,    help="LoRA rank")
    parser.add_argument("--lora_alpha",        type=int,   default=32,    help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout",      type=float, default=0.05,  help="LoRA dropout")
    parser.add_argument("--target_modules",    type=str,   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated list of modules to apply LoRA to")

    # Training hyperparameters
    parser.add_argument("--epochs",            type=int,   default=3,     help="Number of training epochs")
    parser.add_argument("--per_device_batch",  type=int,   default=4,     help="Per-device training batch size")
    parser.add_argument("--grad_accum",        type=int,   default=4,     help="Gradient accumulation steps")
    parser.add_argument("--learning_rate",     type=float, default=2e-4,  help="Peak learning rate")
    parser.add_argument("--warmup_ratio",      type=float, default=0.03,  help="Warmup ratio")
    parser.add_argument("--max_seq_length",    type=int,   default=512,   help="Maximum token sequence length")
    parser.add_argument("--lr_scheduler",      type=str,   default="cosine", help="LR scheduler type")
    parser.add_argument("--weight_decay",      type=float, default=0.01,  help="Weight decay")
    parser.add_argument("--save_steps",        type=int,   default=200,   help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps",     type=int,   default=50,    help="Log metrics every N steps")
    parser.add_argument("--eval_split",        type=float, default=0.02,  help="Fraction of data used for evaluation (0 to disable)")
    parser.add_argument("--seed",              type=int,   default=42,    help="Random seed")

    # Hardware
    parser.add_argument("--bits",              type=int,   default=4,     choices=[4, 8, 16],
                        help="Quantization bits: 4 (QLoRA), 8 (int8), or 16 (bf16, no quant)")
    parser.add_argument("--bf16",              action="store_true", default=True,
                        help="Use bfloat16 mixed precision (recommended on Ampere+ GPUs)")
    parser.add_argument("--fp16",              action="store_true", default=False,
                        help="Use float16 mixed precision (use if bf16 is unavailable)")

    return parser.parse_args()


def load_and_filter(csv_path: str, text_column: str) -> Dataset:
    df = pd.read_csv(csv_path)

    original_count = len(df)
    df = df[[text_column]].dropna()
    df["_word_count"] = df[text_column].str.split().str.len()
    df = df[(df["_word_count"] >= MIN_WORDS) & (df["_word_count"] <= MAX_WORDS)]
    df = df.drop(columns=["_word_count"])
    df = df.drop_duplicates(subset=text_column)

    filtered_count = len(df)
    logger.info(
        f"Posts after filtering (≥{MIN_WORDS} and ≤{MAX_WORDS} words, removed duplicates): "
        f"{filtered_count:,}  (removed {original_count - filtered_count:,})"
    )

    return Dataset.from_pandas(df, preserve_index=False)

def get_bnb_config(bits: int) -> BitsAndBytesConfig | None:
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None  # bits == 16 → no quantization

PROMPT_TEMPLATE = "### Instruction: {} \n ### Answer: {}"
INSTRUCTIONS_LIST = ["Write a post for the gab.com social network", "Write a post you would publish on the gab.com website",
                     "Write a post on gab.com, sharing your opinions", "You have an account on gab.com. Write a post"]

def format_prompt(row, content_field):
    instruction = random.choice(INSTRUCTIONS_LIST)
    return {"text": PROMPT_TEMPLATE.format(instruction, row[content_field])}

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

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = load_and_filter(args.csv_path, args.text_column)
    dataset = dataset.map(format_prompt, batched=False, fn_kwargs={"content_field": args.text_column})

    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_dataset, eval_dataset = split["train"], split["test"]
        logger.info(f"Train size: {len(train_dataset):,} | Eval size: {len(eval_dataset):,}")
    else:
        train_dataset, eval_dataset = dataset, None
        logger.info(f"Train size: {len(train_dataset):,} | No evaluation split")

    logger.info(f"Loading tokenizer from {args.model_id}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = get_bnb_config(args.bits)
    logger.info(f"Loading model ({args.bits}-bit) from {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16 if args.bits == 16 else None,
        attn_implementation="flash_attention_2",  # remove if not supported
    )
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    text = "Write 5 posts that you would publish on gab.com"
    encoded_text = gpt2_tokenizer.encode(text)
    print(model(**encoded_text))

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["c_attn", "c_proj"],  # "q_proj", "v_proj"
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    mixed_precision = "bf16" if args.bf16 else ("fp16" if args.fp16 else "no")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        bf16=(mixed_precision == "bf16"),
        fp16=(mixed_precision == "fp16"),
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_steps=args.save_steps if eval_dataset else None,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        report_to="none",  # swap for "wandb" or "tensorboard" if desired
        seed=args.seed,
        dataloader_num_workers=4,
        optim="paged_adamw_8bit" if args.bits in (4, 8) else "adamw_torch",
        gradient_checkpointing=True,
        group_by_length=True,  # speeds up training by minimising padding
        ddp_find_unused_parameters=False,
    )

    print(training_args)
    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        completion_only_loss=True,
        packing=False,  # must be False when using completion-only collator
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training …")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final_lora_adapter")
    logger.info(f"Saving LoRA adapter to {final_path}")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Done! ✓")

    # ── Usage hint ────────────────────────────────────────────────────────────
    logger.info(
        "\n──────────────────────────────────────────────\n"
        "To load and run inference:\n\n"
        "  from peft import PeftModel\n"
        "  from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        f"  base = AutoModelForCausalLM.from_pretrained('{args.model_id}', device_map='auto')\n"
        f"  model = PeftModel.from_pretrained(base, '{final_path}')\n"
        f"  tokenizer = AutoTokenizer.from_pretrained('{final_path}')\n"
        "──────────────────────────────────────────────"
    )


if __name__ == "__main__":
    main()
