import argparse
import logging
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Must match the template used during fine-tuning
PROMPT = "### Instruction: Write a post for the gab.com social network.\n### Answer:"


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate posts with fine-tuned LoRA model")

    # Paths
    parser.add_argument("--adapter_path", type=str, default="irix-12b-lora/final_lora_adapter/adapter_model.safetensors",
                        help="Path to the saved LoRA adapter (final_lora_adapter directory)")
    parser.add_argument("--base_model", type=str, default="DreadPoor/Irix-12B-Model_Stock",
                        help="Base model ID (must match the one used for fine-tuning)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save generated posts as a text file")

    # Quantization
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8, 16],
                        help="Quantization bits (should match what was used during fine-tuning)")

    # Generation
    parser.add_argument("--n", type=int, default=1, help="Number of posts to generate")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Maximum new tokens to generate per post")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (higher = more creative)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Penalise repeated tokens (1.0 = disabled, >1.0 = penalise)")
    parser.add_argument("--gpu_memory", type=str, default="10GiB",
                        help="Max VRAM to use, e.g. '38GiB' for a 40GB card or '78GiB' for an 80GB card")
    parser.add_argument("--cpu_memory", type=str, default="64GiB",
                        help="Max CPU RAM to use as overflow when model doesn't fit in VRAM")

    # Mode
    parser.add_argument("--interactive", action="store_true",
                        help="Start an interactive REPL loop (overrides --n)")

    return parser.parse_args()


# ── Quantization config ───────────────────────────────────────────────────────
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
    return None


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(args: argparse.Namespace):
    logger.info(f"Loading base model: {args.base_model}")
    bnb_config = get_bnb_config(args.bits)
    max_memory = {
        0: args.gpu_memory,
        "cpu": args.cpu_memory,
    }

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    base.config.use_cache = True  # enable KV cache for faster inference

    logger.info(f"Loading LoRA adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    logger.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ── Generation ────────────────────────────────────────────────────────────────
@torch.inference_mode()
def generate(model, tokenizer, args: argparse.Namespace, n: int = 1) -> list[str]:
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs,
        num_return_sequences=n,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens, skipping the prompt
    posts = [
        tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
        for seq in outputs
    ]
    return posts


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    model, tokenizer = load_model(args)

    if args.interactive:
        print("\nInteractive mode — press Ctrl+C to quit.\n")
        while True:
            try:
                input("Press Enter to generate a post (Ctrl+C to quit) … ")
                posts = generate(model, tokenizer, args, n=1)
                print(f"\n── Generated post ──\n{posts[0]}\n")
            except KeyboardInterrupt:
                print("\nBye!")
                break

    else:
        logger.info(f"Generating {args.n} post(s) …")
        posts = generate(model, tokenizer, args, n=args.n)

        for i, post in enumerate(posts, 1):
            print(f"\n── Post {i} ──\n{post}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("\n\n---\n\n".join(posts))
            logger.info(f"Saved {len(posts)} post(s) to {args.output}")


if __name__ == "__main__":
    main()