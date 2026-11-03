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
    TrainingArguments
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "mbert_cpu"
RESULTS = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(exist_ok=True)

MODEL_NAME = "bert-base-multilingual-cased"
SEED = 42
MAX_LENGTH = 96

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

print("=" * 60)
print("PHASE 3 - mBERT CPU TRAINING")
print("=" * 60)

print("Device: CPU")

# Load data
train = pd.read_csv(DATA / "train_clean.csv")
val = pd.read_csv(DATA / "validation_clean.csv")

labels = {"negative": 0, "positive": 1}

for df in [train, val]:
    df["label"] = df["label"].astype(str).str.lower()
    df.dropna(subset=["text"], inplace=True)
    df.drop(
        df[~df["label"].isin(labels)].index,
        inplace=True
    )
    df["label_id"] = df["label"].map(labels).astype(int)

# Use a balanced, reproducible CPU experiment:
# 1,000 examples per language, approximately balanced by label.
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
    frac=1,
    random_state=SEED
).reset_index(drop=True)

print("Training samples:", len(train_small))
print("Validation samples:", len(val))

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class SentimentDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["label_id"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item

train_dataset = SentimentDataset(train_small)
val_dataset = SentimentDataset(val)

print("\nLoading pretrained mBERT...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id=labels
)

# First CPU run: train classification head only.
# This makes the experiment much faster.
for parameter in model.bert.parameters():
    parameter.requires_grad = False

def compute_metrics(eval_pred):
    logits, true_labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy_score(
            true_labels, predictions
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

args = TrainingArguments(
    output_dir=str(MODEL_DIR / "checkpoints"),
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
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
    compute_metrics=compute_metrics
)

print("\nStarting CPU training...")
trainer.train()

print("\nEvaluating validation dataset...")
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
    RESULTS / "mbert_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nLANGUAGE-WISE RESULTS")
for language, group in results.groupby("language"):
    accuracy = accuracy_score(
        group["label"],
        group["prediction"]
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

print("\nModel saved to:", MODEL_DIR)
print("\nPHASE 3 CPU EXPERIMENT COMPLETED!")