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
DATA_PATH = os.path.join(HOME_DIR, "gab_analysis/for_immense/real_posts_newer.tsv")
OUTPUT_DIR = os.path.join(HOME_DIR, "gab_analysis/xlmt_finetuned", "model_attention_pooling")
PREDICT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "test_predictions.tsv")
TEXT_COLUMN = "content"
LABEL_COLUMN = "exact_level_found"
PREDICTION_COLUMN = "predicted_by_xlmt"
NUM_LABELS = 6
MAX_LENGTH = 512
MAX_POSTS_PER_USER = 500
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.7, 0.15, 0.15
SEED = 42
N_FOLDS = 10


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


def _cap_posts(group: pd.DataFrame, cap: int, label: int, rng: np.random.Generator) -> list:
    if len(group) <= cap:
        return group[TEXT_COLUMN].astype(str).tolist()

    # always keep a post that actually justifies the max-based label, then fill the rest
    # randomly - a naive uniform sample could drop the one post the label depends on
    max_idx = group.index[group[LABEL_COLUMN] == label].to_numpy()
    keep_idx = rng.choice(max_idx, size=1, replace=False)
    remaining_idx = np.setdiff1d(group.index.to_numpy(), keep_idx)
    extra_idx = rng.choice(remaining_idx, size=cap - 1, replace=False)
    chosen_idx = np.concatenate([keep_idx, extra_idx])
    return group.loc[chosen_idx, TEXT_COLUMN].astype(str).tolist()


def load_user_bags(path: str, labels_dict: {}, max_posts_per_user: int = MAX_POSTS_PER_USER, seed: int = SEED) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["account_id", TEXT_COLUMN, LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    rng = np.random.default_rng(seed)
    rows = []
    dropped = 0
    for account_id, group in df.groupby("account_id"):
        label = int(group[LABEL_COLUMN].max()) if not labels_dict else labels_dict[account_id]
        posts = _cap_posts(group, max_posts_per_user, label, rng)
        if not posts:
            dropped += 1
            continue
        rows.append({"account_id": account_id, "label": label, "posts": posts})

    if dropped:
        logger.info("Dropped %d users with no valid posts", dropped)

    bags_df = pd.DataFrame(rows)
    logger.info("Built %d user bags from %d posts", len(bags_df), len(df))
    return bags_df


def split_bags(bags_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        train_val_df, test_df = train_test_split(
            bags_df, test_size=TEST_FRAC, stratify=bags_df["label"], random_state=SEED
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=VAL_FRAC / (TRAIN_FRAC + VAL_FRAC),
            stratify=train_val_df["label"],
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
        data["labels"] = bags_df["label"].tolist()
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
        self.encoder = AutoModel.from_pretrained(model_name)
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

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=8,
        report_to="none",
    )
    return Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


def _training_args(output_dir: str) -> TrainingArguments:
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
    labels = bags_df["label"].to_numpy()

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(bags_df, labels), start=1):
        logger.info("Cross-validation fold %d/%d", fold, n_folds)
        train_df = bags_df.iloc[train_idx].reset_index(drop=True)
        val_df = bags_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = to_bag_dataset(train_df, tokenizer)
        val_dataset = to_bag_dataset(val_df, tokenizer)
        train_dataset = train_dataset.add_column("length", [len(ids) for ids in train_dataset["input_ids"]])

        class_weights = _balanced_class_weights(train_df["label"].to_numpy())
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
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    logger.info("Test set metrics: %s", predictions.metrics)
    #
    true_labels = test_df["label"].to_numpy()
    p, r, f, _ = precision_recall_fscore_support(true_labels, preds, average="macro", zero_division=0)
    logger.info("Classification report")
    logger.info(classification_report(true_labels, preds))
    print(classification_report(true_labels, preds))

    out_df = test_df[["account_id", "label"]].rename(columns={"label": LABEL_COLUMN})
    out_df[PREDICTION_COLUMN] = preds
    os.makedirs(os.path.dirname(PREDICT_OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(PREDICT_OUTPUT_PATH, sep="\t", index=False)
    logger.info("Wrote %d test predictions to %s", len(out_df), PREDICT_OUTPUT_PATH)


def main() -> None:
    CROSS_VAL = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = UserBagCollator(tokenizer)

    report_token_lengths(pd.read_csv(DATA_PATH, sep="\t").dropna(subset=[TEXT_COLUMN]), tokenizer)

    bags_df = load_user_bags(DATA_PATH)

    if not CROSS_VAL:
        train_df, val_df, test_df = split_bags(bags_df)
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

        train_ids, train_embeddings = extract_user_embeddings(trainer.model, train_df, tokenizer, data_collator)
        save_user_embeddings(train_ids, train_embeddings, os.path.join(OUTPUT_DIR, "train_user_embeddings.pt"))

        test_ids, test_embeddings = extract_user_embeddings(trainer.model, test_df, tokenizer, data_collator)
        save_user_embeddings(test_ids, test_embeddings, os.path.join(OUTPUT_DIR, "test_user_embeddings.pt"))
    else:
        cross_validate(bags_df, tokenizer, data_collator)


if __name__ == "__main__":
    main()
