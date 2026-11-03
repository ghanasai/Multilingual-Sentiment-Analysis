import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODEL_DIR = ROOT / "models" / "mbert_improved"

RESULTS.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "bert-base-multilingual-cased"
SEED = 42
MAX_LENGTH = 96

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

print("=" * 60)
print("PHASE 3 - IMPROVED mBERT CPU FINE-TUNING")
print("=" * 60)

# Load and clean data
train = pd.read_csv(DATA / "train_clean.csv")
val = pd.read_csv(DATA / "validation_clean.csv")

label_map = {"negative": 0, "positive": 1}

def clean_dataframe(df):
    df = df.copy()
    df["label"] = (
        df["label"].astype("string").str.lower().str.strip()
    )
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin(label_map)].copy()
    df["label_id"] = df["label"].map(label_map).astype(int)
    return df

train = clean_dataframe(train)
val = clean_dataframe(val)

# Balanced samples from five languages
samples = []

for language, group in train.groupby("language"):
    for sentiment, subset in group.groupby("label"):
        samples.append(
            subset.sample(
                n=min(500, len(subset)),
                random_state=SEED
            )
        )

train_small = pd.concat(samples).sample(
    frac=1, random_state=SEED
).reset_index(drop=True)

print("Training samples:", len(train_small))
print("Validation samples:", len(val))

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class SentimentDataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenizer(
            df["text"].astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
        self.labels = df["label_id"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(
            self.labels[index], dtype=torch.long
        )
        return item

train_dataset = SentimentDataset(train_small)
val_dataset = SentimentDataset(val)

print("\nLoading pretrained mBERT...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id=label_map
)

# Freeze entire BERT encoder
for parameter in model.bert.parameters():
    parameter.requires_grad = False

# Unfreeze final two transformer layers
for layer in model.bert.encoder.layer[-2:]:
    for parameter in layer.parameters():
        parameter.requires_grad = True

# Classification head remains trainable
trainable = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

total = sum(p.numel() for p in model.parameters())

print(
    f"Trainable parameters: {trainable:,} / {total:,}"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

args = TrainingArguments(
    output_dir=str(MODEL_DIR / "checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=100,
    report_to="none",
    no_cuda=True,
    dataloader_num_workers=0,
    seed=SEED
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)

print("\nStarting partial fine-tuning...")
trainer.train()

print("\nEvaluating best model...")
metrics = trainer.evaluate()

print("\nFINAL VALIDATION METRICS")
for key, value in metrics.items():
    if key.startswith("eval_"):
        print(f"{key}: {value}")

output = trainer.predict(val_dataset)
predictions = np.argmax(output.predictions, axis=1)

print("\nClassification report:")
print(classification_report(
    val["label_id"],
    predictions,
    target_names=["negative", "positive"],
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(
    val["label_id"],
    predictions,
    labels=[0, 1]
))

results = val[["language", "text", "label"]].copy()
results["prediction"] = [
    "positive" if p == 1 else "negative"
    for p in predictions
]

results.to_csv(
    RESULTS / "mbert_improved_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nLANGUAGE-WISE RESULTS")
for language, group in results.groupby("language"):
    accuracy = accuracy_score(
        group["label"], group["prediction"]
    )
    _, _, f1, _ = precision_recall_fscore_support(
        group["label"],
        group["prediction"],
        average="macro",
        zero_division=0
    )
    print(
        f"{language}: Accuracy={accuracy:.4f}, "
        f"Macro F1={f1:.4f}"
    )

trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

print("\nImproved model saved to:", MODEL_DIR)
print("\nIMPROVED mBERT EXPERIMENT COMPLETED!")