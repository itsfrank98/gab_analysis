import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

HOME_DIR = "/lustrehome/benedettifrancescophd/"

MODEL_NAME = os.path.join(HOME_DIR, "models/roberta-xlmt")
DATA_PATH = os.path.join(HOME_DIR, "gab_analysis/downstream_task/real_posts_capped_64.tsv")
OUTPUT_DIR = os.path.join(HOME_DIR, "gab_analysis/xlmt_finetuned", "model_attention_pooling_network_labeling_real")
LABELS_DF_NAME = os.path.join(HOME_DIR, "gab_analysis/downstream_task/real_users_capped.tsv")
PREDICT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "test_predictions.tsv")
TEXT_COLUMN = "content"
POST_LABEL_COLUMN = "exact_level_found"
ACCOUNT_LABEL_COLUMN = "label"
PREDICTION_COLUMN = "predicted_by_xlmt"
NUM_LABELS = 6
MAX_LENGTH = 512
MAX_POSTS_PER_USER = 64
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.7, 0.15, 0.15
SEED = 42
N_FOLDS = 10


def is_main_process() -> bool:
    """True on the single process (single-GPU) or rank-0 process (torchrun DDP)."""
    return int(os.environ.get("RANK", "0")) == 0


def report_token_lengths(df: pd.DataFrame, tokenizer) -> None:
    lengths = [len(ids) for ids in tokenizer(df[TEXT_COLUMN].tolist(), truncation=False)["input_ids"]]
    lengths = np.array(lengths)
    logger.info(
        "Token lengths (no truncation) - max: %d, mean: %.1f, p95: %.1f, p99: %.1f, over %d: %d/%d",
        lengths.max(),
        lengths.mean(),
        np.percentile(lengths, 95),
        np.percentile(lengths, 99),
        MAX_LENGTH,
        (lengths > MAX_LENGTH).sum(),
        len(lengths),
    )


def report_posts_per_user(bags_df: pd.DataFrame) -> None:
    counts = bags_df["posts"].map(len)
    logger.info(
        "Posts per user - mean: %.1f, median: %.0f, p95: %.0f, p99: %.0f, max: %d (capped at %d)",
        counts.mean(),
        counts.median(),
        np.percentile(counts, 95),
        np.percentile(counts, 99),
        counts.max(),
        MAX_POSTS_PER_USER,
    )


def _cap_posts(group: pd.DataFrame, cap: int, label: int, rng: np.random.Generator, aggregator: str = "max") -> list:
    if len(group) <= cap:
        return group[TEXT_COLUMN].astype(str).tolist()

    if aggregator == "max":
        # always keep a post that actually justifies the max-based label, then fill the rest
        # randomly - a naive uniform sample could drop the one post the label depends on
        max_idx = group.index[group[POST_LABEL_COLUMN] == label].to_numpy()
        keep_idx = rng.choice(max_idx, size=1, replace=False)
        remaining_idx = np.setdiff1d(group.index.to_numpy(), keep_idx)
        extra_idx = rng.choice(remaining_idx, size=cap - 1, replace=False)
        chosen_idx = np.concatenate([keep_idx, extra_idx])
    else:
        chosen_idx = rng.choice(group.index.to_numpy(), size=cap, replace=False)

    return group.loc[chosen_idx, TEXT_COLUMN].astype(str).tolist()


def load_user_bags(path: str, labels_df: pd.DataFrame = None, max_posts_per_user: int = MAX_POSTS_PER_USER, seed: int = SEED, aggregator: str = "max") -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["account_id", TEXT_COLUMN, POST_LABEL_COLUMN])
    labels_accounts = labels_df["account_id"].tolist()
    df = df[df["account_id"].isin(labels_accounts)]
    df[POST_LABEL_COLUMN] = df[POST_LABEL_COLUMN].astype(int)

    labels_dict = None
    if labels_df is not None:
        labels_df = labels_df.set_index("account_id")
        labels_dict = labels_df.to_dict()[ACCOUNT_LABEL_COLUMN]
    rng = np.random.default_rng(seed)
    rows = []
    dropped = 0
    for account_id, group in df.groupby("account_id"):
        label = int(group[POST_LABEL_COLUMN].max()) if not labels_dict else labels_dict[account_id]
        posts = _cap_posts(group, max_posts_per_user, label, rng, aggregator)
        if not posts:
            dropped += 1
            continue
        rows.append({"account_id": account_id, ACCOUNT_LABEL_COLUMN: label, "posts": posts})

    if dropped:
        logger.info("Dropped %d users with no valid posts", dropped)

    bags_df = pd.DataFrame(rows)
    logger.info("Built %d user bags from %d posts", len(bags_df), len(df))
    return bags_df


def split_bags(bags_df: pd.DataFrame, test_account_ids: list = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits bags into train/val/test.

    If test_account_ids is given, that exact set of accounts becomes the test split (so
    the test set stays fixed across runs/strategies) and only train/val get split here.
    """
    if test_account_ids is not None:
        test_mask = bags_df["account_id"].isin(test_account_ids)
        test_df = bags_df[test_mask]
        train_val_df = bags_df[~test_mask]
        try:
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=VAL_FRAC / (TRAIN_FRAC + VAL_FRAC),
                stratify=train_val_df[ACCOUNT_LABEL_COLUMN],
                random_state=SEED,
            )
        except ValueError:
            logger.warning("Stratified split failed (a class is too small) - falling back to unstratified split")
            train_df, val_df = train_test_split(
                train_val_df, test_size=VAL_FRAC / (TRAIN_FRAC + VAL_FRAC), random_state=SEED
            )
    else:
        try:
            train_val_df, test_df = train_test_split(
                bags_df, test_size=TEST_FRAC, stratify=bags_df[ACCOUNT_LABEL_COLUMN], random_state=SEED
            )
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=VAL_FRAC / (TRAIN_FRAC + VAL_FRAC),
                stratify=train_val_df[ACCOUNT_LABEL_COLUMN],
                random_state=SEED,
            )
        except ValueError:
            logger.warning("Stratified split failed (a class is too small) - falling back to unstratified split")
            train_val_df, test_df = train_test_split(bags_df, test_size=TEST_FRAC, random_state=SEED)
            train_df, val_df = train_test_split(
                train_val_df, test_size=VAL_FRAC / (TRAIN_FRAC + VAL_FRAC), random_state=SEED
            )

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info("%s split: %d users", name, len(split))

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def to_bag_dataset(bags_df: pd.DataFrame, tokenizer, include_labels: bool = True) -> Dataset:
    data = {"posts": bags_df["posts"].tolist()}
    if include_labels:
        data["labels"] = bags_df[ACCOUNT_LABEL_COLUMN].tolist()
    ds = Dataset.from_dict(data)

    def tokenize_batch(examples):
        all_ids, all_mask = [], []
        for posts in examples["posts"]:
            encoded = tokenizer(posts, truncation=True, max_length=MAX_LENGTH)
            all_ids.append(encoded["input_ids"])
            all_mask.append(encoded["attention_mask"])
        return {"input_ids": all_ids, "attention_mask": all_mask}

    # do not call ds.set_format("torch") - input_ids/attention_mask are ragged
    # (variable posts x variable tokens per user), only UserBagCollator can pad them
    return ds.map(tokenize_batch, batched=True, remove_columns=["posts"])


class UserBagCollator:
    """Pads a batch of per-user post bags into (batch, max_posts, max_tokens) tensors.

    Needed because each example is ragged in two dimensions (posts per user, tokens per
    post), which the built-in HF collators (single flat sequence per example) can't handle.
    """

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list) -> dict:
        batch_size = len(features)
        max_posts = max(len(f["input_ids"]) for f in features)
        max_tokens = max(len(post_ids) for f in features for post_ids in f["input_ids"])

        input_ids = torch.full((batch_size, max_posts, max_tokens), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_posts, max_tokens), dtype=torch.long)
        # every post slot (including filler ones) keeps one unmasked token so the encoder's
        # attention softmax never sees an all-masked row - that produces NaNs *before*
        # post_mask gets a chance to zero the filler post's contribution out
        attention_mask[:, :, 0] = 1
        post_mask = torch.zeros((batch_size, max_posts), dtype=torch.long)

        for i, f in enumerate(features):
            post_mask[i, : len(f["input_ids"])] = 1
            for j, (post_ids, post_attn) in enumerate(zip(f["input_ids"], f["attention_mask"])):
                length = len(post_ids)
                input_ids[i, j, :length] = torch.tensor(post_ids, dtype=torch.long)
                attention_mask[i, j, :length] = torch.tensor(post_attn, dtype=torch.long)

        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "post_mask": post_mask}
        if "labels" in features[0]:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return batch


class UserAttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = NUM_LABELS,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        # add_pooling_layer=False drops RoBERTa's built-in CLS-token pooler head
        # (outputs.pooler_output), which forward() never reads - only last_hidden_state
        # feeds the mean/attention pooling below. Left enabled, that head's params get no
        # gradient, which DDP flags as "unused parameters" (error or slower find_unused search)
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden_size = self.encoder.config.hidden_size
        self.attn_score = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        # only applied during eval (see forward) - training still uses the unweighted loss
        self.register_buffer("class_weights", class_weights)

    def encode(self, input_ids, attention_mask, post_mask):
        """Pools a batch of per-user post bags into one embedding per user (B, H)."""
        batch_size, num_posts, num_tokens = input_ids.shape
        flat_ids = input_ids.view(batch_size * num_posts, num_tokens)
        flat_mask = attention_mask.view(batch_size * num_posts, num_tokens)

        token_embeds = self.encoder(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
        token_mask = flat_mask.unsqueeze(-1).float()
        post_embeds = (token_embeds * token_mask).sum(1) / token_mask.sum(1).clamp(min=1e-9)
        post_embeds = post_embeds.view(batch_size, num_posts, -1)

        scores = self.attn_score(post_embeds).squeeze(-1)
        scores = scores.masked_fill(post_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        user_embeds = (weights * post_embeds).sum(1)
        return user_embeds

    def forward(self, input_ids, attention_mask, post_mask, labels=None):
        user_embeds = self.encode(input_ids, attention_mask, post_mask)
        logits = self.classifier(user_embeds)
        loss = None
        if labels is not None:
            weight = None if self.training else self.class_weights
            loss = nn.functional.cross_entropy(logits, labels, weight=weight)
        return {"loss": loss, "logits": logits}


def build_model(class_weights: torch.Tensor | None = None) -> UserAttentionPoolingClassifier:
    return UserAttentionPoolingClassifier(class_weights=class_weights)


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }


def model_exists(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "model_state_dict.pt"))


def load_trainer(model_path: str, data_collator: UserBagCollator) -> Trainer:
    logger.info("Loading fine-tuned model from %s", model_path)
    model = build_model()
    state_dict = torch.load(os.path.join(model_path, "model_state_dict.pt"), map_location="cpu")
    model.load_state_dict(state_dict)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=8,
        report_to="none",
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
    )
    return Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


def _training_args(output_dir: str) -> TrainingArguments:
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="no",  # explicit save below - Trainer's own checkpointing for a non-PreTrainedModel falls back to undocumented behavior
        train_sampling_strategy ="group_by_length",
        length_column_name="length",
        num_train_epochs=5,
        per_device_train_batch_size=4,  # lower than a flat-sequence setup: cost now scales with batch_size x posts_per_bag x tokens_per_post
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        seed=SEED,
        report_to="none",
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # encoder has no unused params now (see add_pooling_layer=False above), so DDP
        # doesn't need to search for them each step - cuts DDP overhead per backward pass
        ddp_find_unused_parameters=False,
    )


def train(train_dataset: Dataset, val_dataset: Dataset, data_collator: UserBagCollator, tokenizer) -> Trainer:
    model = build_model()

    train_dataset = train_dataset.add_column("length", [len(ids) for ids in train_dataset["input_ids"]])

    training_args = _training_args(OUTPUT_DIR)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # under multi-GPU DDP every rank runs this function identically - only rank 0 should
    # touch disk, otherwise ranks race to write the same files
    if trainer.is_world_process_zero():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_state_dict.pt"))
        tokenizer.save_pretrained(OUTPUT_DIR)
        with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
            json.dump(
                {
                    "model_name": MODEL_NAME,
                    "num_labels": NUM_LABELS,
                    "max_posts_per_user": MAX_POSTS_PER_USER,
                    "max_length": MAX_LENGTH,
                },
                f,
            )
        logger.info("Saved fine-tuned model to %s", OUTPUT_DIR)

    return trainer


def _balanced_class_weights(labels: np.ndarray, num_labels: int = NUM_LABELS) -> torch.Tensor:
    weights = compute_class_weight("balanced", classes=np.arange(num_labels), y=labels)
    return torch.tensor(weights, dtype=torch.float)


def _average_fold_metrics(fold_metrics: list) -> dict:
    keys = fold_metrics[0].keys()
    summary = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics])
        summary[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary


def cross_validate(bags_df: pd.DataFrame, tokenizer, data_collator: UserBagCollator, n_folds: int = N_FOLDS) -> list:
    """Repeats the main() train/eval pipeline over n_folds stratified folds.

    Same model, tokenization and training args as train() for each fold. The only
    difference from that pipeline is that validation loss is computed with balanced
    class weights (derived from the fold's own train split) to account for label
    imbalance, since the plain unweighted loss used by train()/evaluate_on_test()
    would let majority classes dominate the reported eval loss.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    labels = bags_df[ACCOUNT_LABEL_COLUMN].to_numpy()

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(bags_df, labels), start=1):
        logger.info("Cross-validation fold %d/%d", fold, n_folds)
        train_df = bags_df.iloc[train_idx].reset_index(drop=True)
        val_df = bags_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = to_bag_dataset(train_df, tokenizer)
        val_dataset = to_bag_dataset(val_df, tokenizer)
        train_dataset = train_dataset.add_column("length", [len(ids) for ids in train_dataset["input_ids"]])

        class_weights = _balanced_class_weights(train_df[ACCOUNT_LABEL_COLUMN].to_numpy())
        model = build_model(class_weights=class_weights)

        fold_output_dir = os.path.join(OUTPUT_DIR, "cross_validation", f"fold_{fold}")
        trainer = Trainer(
            model=model,
            args=_training_args(fold_output_dir),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        metrics = trainer.evaluate()
        logger.info("Fold %d metrics: %s", fold, metrics)
        fold_metrics.append(metrics)

        # extract_user_embeddings/save run a plain (non-DDP) forward pass - restrict to
        # rank 0 to avoid every rank redoing the same inference and racing on the same files
        if trainer.is_world_process_zero():
            train_ids, train_embeddings = extract_user_embeddings(trainer.model, train_df, tokenizer, data_collator)
            save_user_embeddings(train_ids, train_embeddings, os.path.join(fold_output_dir, "train_user_embeddings.pt"))

            test_ids, test_embeddings = extract_user_embeddings(trainer.model, val_df, tokenizer, data_collator)
            save_user_embeddings(test_ids, test_embeddings, os.path.join(fold_output_dir, "test_user_embeddings.pt"))
            torch.save(trainer.model.state_dict(), os.path.join(fold_output_dir, "model_state_dict.pt"))
            tokenizer.save_pretrained(fold_output_dir)

    summary = _average_fold_metrics(fold_metrics)
    logger.info("Cross-validation summary over %d folds: %s", n_folds, summary)

    return fold_metrics


def extract_user_embeddings(
    model: UserAttentionPoolingClassifier,
    bags_df: pd.DataFrame,
    tokenizer,
    data_collator: UserBagCollator,
    batch_size: int = 8,
) -> tuple[list, torch.Tensor]:
    """Runs the encoder + attention pooling (no classifier head) over a set of user bags.

    Returns (account_ids, embeddings) with embeddings[i] corresponding to account_ids[i].
    Uses a plain non-shuffling DataLoader over a single process, so - unlike Trainer.predict()
    under distributed evaluation - row order is guaranteed to match bags_df's order exactly.
    """
    device = next(model.parameters()).device
    model.eval()

    dataset = to_bag_dataset(bags_df, tokenizer, include_labels=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            embeddings.append(model.encode(batch["input_ids"], batch["attention_mask"], batch["post_mask"]).cpu())

    return bags_df["account_id"].tolist(), torch.cat(embeddings, dim=0)


def save_user_embeddings(account_ids: list, embeddings: torch.Tensor, path: str) -> None:
    torch.save({"account_id": account_ids, "embeddings": embeddings}, path)
    logger.info("Saved %d user embeddings (dim %d) to %s", len(account_ids), embeddings.shape[1], path)


def evaluate_on_test(trainer: Trainer, test_dataset: Dataset, test_df: pd.DataFrame) -> None:
    # trainer.predict is a collective call under DDP (every rank feeds its shard and the
    # results get gathered) - it must run unconditionally on all ranks; only the reporting
    # and file-writing below is restricted to rank 0
    predictions = trainer.predict(test_dataset)
    if not trainer.is_world_process_zero():
        return

    preds = np.argmax(predictions.predictions, axis=-1)
    logger.info("Test set metrics: %s", predictions.metrics)

    true_labels = test_df[ACCOUNT_LABEL_COLUMN].to_numpy()
    binary_true_labels = np.array([1 if e > 2 else 0 for e in true_labels])
    binary_preds = np.array([1 if e > 2 else 0 for e in preds])
    p, r, f, _ = precision_recall_fscore_support(true_labels, preds, average="macro", zero_division=0)
    logger.info("Classification report")
    logger.info(classification_report(true_labels, preds))

    logger.info("Binary classification report")
    logger.info(classification_report(binary_true_labels, binary_preds))

    out_df = test_df.copy()
    out_df[PREDICTION_COLUMN] = preds
    os.makedirs(os.path.dirname(PREDICT_OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(PREDICT_OUTPUT_PATH, sep="\t", index=False)
    logger.info("Wrote %d test predictions to %s", len(out_df), PREDICT_OUTPUT_PATH)


def main() -> None:
    CROSS_VAL = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = UserBagCollator(tokenizer)
    aggregator = "attention"

    if is_main_process():
        report_token_lengths(pd.read_csv(DATA_PATH, sep="\t").dropna(subset=[TEXT_COLUMN]), tokenizer)

    labels_df = pd.read_csv(LABELS_DF_NAME, sep="\t")
    bags_df = load_user_bags(DATA_PATH, labels_df=labels_df, aggregator=aggregator)

    if not CROSS_VAL:
        test_ids_path = os.path.join(OUTPUT_DIR, "test_account_ids.tsv")
        if os.path.exists(test_ids_path):
            test_account_ids = pd.read_csv(test_ids_path, sep="\t")["account_id"].tolist()
            train_df, val_df, test_df = split_bags(bags_df, test_account_ids=test_account_ids)
        else:
            train_df, val_df, test_df = split_bags(bags_df)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            test_df[["account_id"]].to_csv(test_ids_path, sep="\t", index=False)
        test_dataset = to_bag_dataset(test_df, tokenizer)

        if model_exists(OUTPUT_DIR):
            logger.info("Model exists, I am loading it")
            trainer = load_trainer(OUTPUT_DIR, data_collator)
        else:
            logger.info("Model doesn't exist, I am training it")
            train_dataset = to_bag_dataset(train_df, tokenizer)
            val_dataset = to_bag_dataset(val_df, tokenizer)
            trainer = train(train_dataset, val_dataset, data_collator, tokenizer)

        evaluate_on_test(trainer, test_dataset, test_df)

        if trainer.is_world_process_zero():
            train_ids, train_embeddings = extract_user_embeddings(trainer.model, train_df, tokenizer, data_collator)
            save_user_embeddings(train_ids, train_embeddings, os.path.join(OUTPUT_DIR, "train_user_embeddings.pt"))

            test_ids, test_embeddings = extract_user_embeddings(trainer.model, test_df, tokenizer, data_collator)
            save_user_embeddings(test_ids, test_embeddings, os.path.join(OUTPUT_DIR, "test_user_embeddings.pt"))
    else:
        cross_validate(bags_df, tokenizer, data_collator)


if __name__ == "__main__":
    main()
