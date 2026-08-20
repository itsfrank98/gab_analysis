import logging
import os
import yaml
import numpy as np
import pandas as pd
from datasets import Dataset
from pydantic.v1.schema import model_process_schema
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
SEED = 42

def load_data(path: str, text_column, label_column) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df[[text_column, label_column]].dropna()
    df[label_column] = df[label_column].astype(int)
    df = df.reset_index(drop=True)
    logger.info("Loaded %d posts", len(df))
    return df


def to_dataset(df: pd.DataFrame, tokenizer, text_column, label_column, max_length, include_labels: bool = True) -> Dataset:
    if include_labels:
        cols = df.rename(columns={label_column: "label"})[[text_column, "label"]]
    else:
        cols = df[[text_column]]
    ds = Dataset.from_pandas(cols, preserve_index=False)
    return ds.map(
        lambda batch: tokenizer(batch[text_column], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=[text_column],
    )


def report_token_lengths(df: pd.DataFrame, tokenizer, text_column, max_length) -> None:
    lengths = [len(ids) for ids in tokenizer(df[text_column].tolist(), truncation=False)["input_ids"]]
    lengths = np.array(lengths)
    logger.info(
        "Token lengths (no truncation) - max: %d, mean: %.1f, p95: %.1f, p99: %.1f, over %d: %d/%d",
        lengths.max(),
        lengths.mean(),
        np.percentile(lengths, 95),
        np.percentile(lengths, 99),
        max_length,
        (lengths > max_length).sum(),
        len(lengths),
    )


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    mae = mean_absolute_error(labels, preds)
    rmse = root_mean_squared_error(labels, preds)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "mae": mae,
        "rmse": rmse,
    }


def _average_fold_metrics(fold_metrics: list) -> dict:
    keys = fold_metrics[0].keys()
    summary = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics if key in m])
        summary[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary


def _average_classification_reports(fold_reports: list) -> dict:
    """Averages sklearn classification_report(output_dict=True) dicts across folds.

    Every fold's report must cover the same set of row keys (pass
    labels=np.arange(NUM_LABELS) to classification_report so folds where a class has
    zero support still get a row - otherwise per-class rows would go missing/misaligned
    across folds and this would KeyError).
    """
    summary = {}
    for key, value in fold_reports[0].items():
        if isinstance(value, dict):
            summary[key] = {
                metric: {
                    "mean": float(np.mean([r[key][metric] for r in fold_reports])),
                    "std": float(np.std([r[key][metric] for r in fold_reports])),
                }
                for metric in value
            }
        else:  # "accuracy" is a bare float, not a nested dict
            values = np.array([r[key] for r in fold_reports])
            summary[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary


def _format_classification_report_summary(summary: dict) -> str:
    rows = [k for k in summary if k != "accuracy"]
    lines = [f"{'':>15}{'precision':>16}{'recall':>16}{'f1-score':>16}{'support':>16}"]
    for row in rows:
        cells = "".join(
            f"{summary[row][metric]['mean']:>9.3f} ± {summary[row][metric]['std']:<5.3f}"
            for metric in ("precision", "recall", "f1-score", "support")
        )
        lines.append(f"{row:>15}{cells}")
    if "accuracy" in summary:
        acc = summary["accuracy"]
        lines.append(f"\n{'accuracy':>15}{acc['mean']:>9.3f} ± {acc['std']:<5.3f}")
    return "\n".join(lines)

def build_model(model_name, num_labels) -> PreTrainedModel:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, device_map="auto")
    print(model.device)
    return model


def model_exists(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def load_trainer(model_path: str, tokenizer, data_collator, output_dir) -> Trainer:
    logger.info("Loading fine-tuned model from %s", model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map="auto")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=32,
        report_to="none",
    )
    return Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        processing_class=tokenizer,
    )


def run_kfold_cv(df: pd.DataFrame, tokenizer, data_collator, n_folds, add_synthetic, output_dir, text_column,
                 label_column, max_length, model_name, num_labels, synthetic_posts_path=None, ) -> None:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_metrics = []
    fold_reports = []
    fold_binary_reports = []

    oof_preds = np.empty(len(df), dtype=int)
    oof_labels = np.empty(len(df), dtype=int)
    if add_synthetic:
        synthetic_posts = pd.read_csv(synthetic_posts_path, sep="\t")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y=df[label_column]), start=1):
        logger.info(f"CURRENT FOLD: {fold}")
        os.makedirs(os.path.join(output_dir, "cross_validation", f"fold_{fold}"), exist_ok=True)
        train_df_name = os.path.join(output_dir, "cross_validation", f"fold_{fold}", "train_df.tsv")
        val_df_name = os.path.join(output_dir, "cross_validation", f"fold_{fold}", "val_df.tsv")
        if os.path.exists(train_df_name) and os.path.exists(val_df_name):
            train_df = pd.read_csv(train_df_name, sep="\t")
            if add_synthetic:
                train_df = pd.concat([train_df, synthetic_posts])
            val_df = pd.read_csv(val_df_name, sep="\t")
        else:
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            train_df.to_csv(train_df_name, sep="\t", index=False)
            val_df.to_csv(val_df_name, sep="\t", index=False)

        train_dataset = to_dataset(train_df, tokenizer, text_column=text_column, label_column=label_column, max_length=max_length)
        val_dataset = to_dataset(val_df, tokenizer, text_column=text_column, label_column=label_column, max_length=max_length)

        model = build_model(model_name, num_labels)

        training_args = TrainingArguments(
            output_dir=f"{output_dir}/fold_{fold}",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=50,
            seed=SEED,
            report_to="none",
        )
        fold_models_dir = os.path.join(output_dir, f"fold_{fold}")
        logger.info("\n\n")
        logger.info(fold_models_dir)
        prefix = "checkpoint-"
        matches = [name for name in os.listdir(fold_models_dir) if name.startswith(prefix) and
                   os.path.isdir(os.path.join(fold_models_dir, name))]
        logger.info(matches)
        if matches:
            ckp_value = max([int(m.split("-")[1]) for m in matches])
            print(f"LOADING THE MODEL FROM CHECKPOINT {ckp_value}")
            model_path = os.path.join(fold_models_dir, f"checkpoint-{ckp_value}")
            trainer = load_trainer(output_dir=output_dir, tokenizer=tokenizer, data_collator=data_collator, model_path=model_path)
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                processing_class=tokenizer,
                compute_metrics=compute_metrics,
            )
            trainer.train()

        predictions = trainer.predict(val_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids

        oof_preds[val_idx] = preds
        oof_labels[val_idx] = labels

        logger.info("Fold %d report: %s", fold, predictions.metrics)
        fold_metrics.append(predictions.metrics)

        report = classification_report(predictions.label_ids, preds, labels=np.arange(num_labels), output_dict=True,
                                       zero_division=0)
        logger.info("Fold %d classification report:\n%s", fold, report)
        fold_reports.append(report)

        binary_preds = np.array([0 if p < 3 else 1 for p in preds])
        binary_true_values = np.array([0 if p < 3 else 1 for p in labels])
        binary_report = classification_report(binary_true_values, binary_preds, labels=np.array([0, 1]),
                                              output_dict=True, zero_division=0)
        logger.info("Fold %d binary classification report:\n%s", fold, binary_report)
        fold_binary_reports.append(binary_report)

    metric_names = [k for k in fold_metrics[0] if k.startswith("test_")]
    with open(f"fold_{n_folds}_metrics.txt", "w") as f:
        f.write(f"=== Cross-validation summary ({n_folds} folds) ===")
        for name in metric_names:
            values = [m[name] for m in fold_metrics]
            f.write(f"\t{name} = {np.mean(values)} +/- {np.std(values)}\n")
        f.close()

    print("Out-of-fold classification report")
    print(classification_report(oof_labels, oof_preds, digits=4))

    summary = _average_fold_metrics(fold_metrics)
    logger.info("Cross-validation summary over %d folds: %s", n_folds, summary)

    if fold_reports:
        report_summary = _average_classification_reports(fold_reports)
        logger.info(
            "Cross-validation mean classification report over %d folds:\n%s",
            n_folds,
            _format_classification_report_summary(report_summary),
        )

    if fold_binary_reports:
        binary_report_summary = _average_classification_reports(fold_binary_reports)
        logger.info(
            "Cross-validation mean classification report over %d folds:\n%s",
            n_folds,
            _format_classification_report_summary(binary_report_summary),
        )


def train(train_df: pd.DataFrame, tokenizer, data_collator, text_column, label_column, output_dir, model_name, num_labels) -> Trainer:
    train_dataset = to_dataset(train_df, tokenizer, text_column, label_column, num_labels)

    model = build_model(model_name, num_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved fine-tuned model to %s", output_dir)

    return trainer


def predict_and_annotate(trainer: Trainer, tokenizer, predict_path: str, output_path: str, text_column, label_column, max_length) -> None:
    predict_df = pd.read_csv(predict_path, sep="\t")
    print(predict_path)
    print(predict_df.columns)
    predict_df = predict_df.dropna(subset=[text_column]).reset_index(drop=True)
    predict_dataset = to_dataset(predict_df, tokenizer, include_labels=False, text_column=text_column,
                                 label_column=label_column, max_length=max_length)

    predictions = trainer.predict(predict_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    predict_df["predicted_by_xlmt"] = preds
    predict_df.to_csv(output_path, sep="\t", index=False)
    logger.info("Wrote %d predictions to %s", len(predict_df), output_path)


def main(model_name, data_poth, cross_validate, output_dir, predict_data_path, predict_output_path, text_column,
         label_column, max_length, n_folds, add_synthetic, synthetic_posts_path, num_labels, model_path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_df = load_data(data_poth, text_column=text_column, label_column=label_column)

    report_token_lengths(load_data(data_poth, text_column=text_column, label_column=label_column), tokenizer,
                         text_column=text_column, max_length=max_length)
    if cross_validate:
        run_kfold_cv(train_df, tokenizer, data_collator, n_folds=n_folds, add_synthetic=add_synthetic, output_dir=output_dir,
                     text_column=text_column, label_column=label_column, max_length=max_length, model_name=model_name,
                     num_labels=num_labels, synthetic_posts_path=synthetic_posts_path)
    else:
        if model_exists(output_dir):
            print("LOADING THE MODEL")
            trainer = load_trainer(output_dir=output_dir, tokenizer=tokenizer, data_collator=data_collator, model_path=model_path)
        else:
            trainer = train(train_df, tokenizer, data_collator, text_column=text_column, label_column=label_column,
                            output_dir=output_dir, model_name=model_name, num_labels=num_labels)

        predict_and_annotate(trainer, tokenizer, predict_data_path, predict_output_path, text_column=text_column,
                             label_column=label_column, max_length=max_length)


if __name__ == "__main__":
    PARAMETERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xlmt_non_aggregated_parameters.yaml")
    with open(PARAMETERS_PATH, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["MODEL_NAME"]
    data_path = config["DATA_PATH"]
    synthetic_posts_path = config["SYNTHETIC_POSTS_PATH"]
    output_dir = config["OUTPUT_DIR"]
    text_column = config["TEXT_COLUMN"]
    label_column = config["LABEL_COLUMN"]
    prediction = config["PREDICTION_COLUMN"]
    num_labels = config["NUM_LABELS"]
    max_length = config["MAX_LENGTH"]
    n_folds = config["N_FOLDS"]
    cross_validate = config["CROSS_VALIDATE"]
    add_synthetic = config["ADD_SYNTHETIC"]
    # use if CROSS_VALIDATE:= False
    prdatapath = config["PREDICT_DATA_PATH"]
    proutput_path = config["PREDICT_OUTPUT_PATH"]
    main(model_name=model_name, data_poth=data_path, synthetic_posts_path=synthetic_posts_path, output_dir=output_dir,
         text_column=text_column, label_column=label_column, max_length=max_length, num_labels=num_labels, model_path=model_name,
         cross_validate=cross_validate, predict_data_path=prdatapath, predict_output_path=proutput_path, n_folds=n_folds,
         add_synthetic=add_synthetic)