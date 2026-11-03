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
    f1_score,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
SOURCE_MODEL = ROOT / "models" / "mbert_improved"
OUTPUT_MODEL = ROOT / "models" / "mbert_multilingual_v2"
RESULTS = ROOT / "results"

OUTPUT_MODEL.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(exist_ok=True)

SEED = 42
MAX_LENGTH = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

print("=" * 60)
print("MULTILINGUAL mBERT IMPROVEMENT")
print("=" * 60)

label_map = {"negative": 0, "positive": 1}

def load_data(filename):
    df = pd.read_csv(DATA / filename)
    df["label"] = (
        df["label"].astype("string")
        .str.strip().str.lower()
    )
    df = df.dropna(subset=["text", "label", "language"])
    df = df[df["label"].isin(label_map)].copy()
    df["label_id"] = df["label"].map(label_map).astype(int)
    return df

train = load_data("train_clean.csv")
val = load_data("validation_clean.csv")

# Balanced multilingual subset:
# 200 positive + 200 negative per language
samples = []

for language in ["en", "hi", "te", "ta", "ml"]:
    for label in ["positive", "negative"]:
        subset = train[
            (train["language"] == language) &
            (train["label"] == label)
        ]

        n = min(200, len(subset))

        samples.append(
            subset.sample(n=n, random_state=SEED)
        )

train_small = pd.concat(samples).sample(
    frac=1, random_state=SEED
).reset_index(drop=True)

print("Training samples:", len(train_small))
print("Validation samples:", len(val))
print("\nLanguage distribution:")
print(train_small["language"].value_counts())

tokenizer = AutoTokenizer.from_pretrained(
    str(SOURCE_MODEL)
)

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

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(
            self.labels[idx], dtype=torch.long
        )
        return item

train_dataset = SentimentDataset(train_small)
val_dataset = SentimentDataset(val)

print("\nLoading existing improved mBERT...")
model = AutoModelForSequenceClassification.from_pretrained(
    str(SOURCE_MODEL)
)

# Freeze all transformer layers
for parameter in model.bert.parameters():
    parameter.requires_grad = False

# Fine-tune only the final transformer layer
for parameter in model.bert.encoder.layer[-1].parameters():
    parameter.requires_grad = True

# Classification head remains trainable
print("Trainable parameters:", sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    }

args = TrainingArguments(
    output_dir=str(OUTPUT_MODEL / "checkpoints"),
    max_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=20,
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

print("\nStarting short multilingual fine-tuning...")
trainer.train()

print("\nEvaluating best checkpoint...")
metrics = trainer.evaluate()

print("\nFINAL RESULTS")
print("Accuracy:", metrics["eval_accuracy"])
print("Macro F1:", metrics["eval_f1"])

output = trainer.predict(val_dataset)
predictions = np.argmax(output.predictions, axis=1)

print("\nClassification report:")
print(classification_report(
    val["label_id"],
    predictions,
    target_names=["negative", "positive"],
    zero_division=0
))

print("\nLANGUAGE-WISE RESULTS")
for language in sorted(val["language"].unique()):
    mask = val["language"] == language
    accuracy = accuracy_score(
        val.loc[mask, "label_id"],
        predictions[mask.to_numpy()]
    )
    f1 = f1_score(
        val.loc[mask, "label_id"],
        predictions[mask.to_numpy()],
        average="macro",
        zero_division=0
    )
    print(
        f"{language}: Accuracy={accuracy:.4f}, "
        f"Macro F1={f1:.4f}"
    )

trainer.save_model(str(OUTPUT_MODEL))
tokenizer.save_pretrained(str(OUTPUT_MODEL))

results = val[["language", "text", "label"]].copy()
results["prediction"] = [
    "positive" if p == 1 else "negative"
    for p in predictions
]

results.to_csv(
    RESULTS / "mbert_multilingual_v2_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved model:", OUTPUT_MODEL)
print("MULTILINGUAL IMPROVEMENT EXPERIMENT COMPLETED!")