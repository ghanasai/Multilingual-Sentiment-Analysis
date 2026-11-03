from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MBERT_PATH = ROOT / "models" / "mbert_full_dataset"
BASELINE_PATH = (
    ROOT / "models" / "baseline_logistic_regression.joblib"
)

torch.set_num_threads(4)

LABEL_MAP = {"negative": 0, "positive": 1}
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "ml": "Malayalam"
}

print("=" * 60)
print("PHASE 5 - FINAL MODEL EVALUATION")
print("=" * 60)

# --------------------------------------------------
# LOAD VALIDATION DATA
# --------------------------------------------------

df = pd.read_csv(DATA / "validation_clean.csv")

df["label"] = (
    df["label"].astype("string")
    .str.strip()
    .str.lower()
)

df = df.dropna(subset=["text", "label", "language"])
df = df[df["label"].isin(LABEL_MAP)].copy()
df["label_id"] = df["label"].map(LABEL_MAP).astype(int)

df = df.reset_index(drop=True)

texts = df["text"].astype(str).tolist()
y_true = df["label_id"].to_numpy()

print("Validation samples:", len(df))
print("Languages:", df["language"].unique())

# --------------------------------------------------
# LOAD LOGISTIC REGRESSION
# --------------------------------------------------

print("\nLoading Logistic Regression...")

baseline = joblib.load(BASELINE_PATH)

baseline_raw = baseline.predict(texts)

def normalize_prediction(value):
    value = str(value).strip().lower()

    if value in ("positive", "1"):
        return 1
    if value in ("negative", "0"):
        return 0

    raise ValueError(f"Unknown prediction label: {value}")

baseline_pred = np.array([
    normalize_prediction(value)
    for value in baseline_raw
])

# --------------------------------------------------
# LOAD FULL-DATASET mBERT
# --------------------------------------------------

print("\nLoading Full-Dataset mBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MBERT_PATH),
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MBERT_PATH),
    local_files_only=True
)

model.to("cpu")
model.eval()

# --------------------------------------------------
# BATCH mBERT PREDICTIONS
# --------------------------------------------------

print("\nRunning mBERT validation predictions...")

BATCH_SIZE = 16
mbert_pred = []

with torch.inference_mode():
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        logits = model(**inputs).logits
        predictions = torch.argmax(
            logits, dim=-1
        ).tolist()

        mbert_pred.extend(predictions)

        print(
            f"Processed {min(start + BATCH_SIZE, len(texts))}"
            f"/{len(texts)}",
            end="\r"
        )

mbert_pred = np.array(mbert_pred)

print("\nPredictions completed.")

# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(y_actual, y_predicted):
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_actual,
            y_predicted,
            average="macro",
            zero_division=0
        )
    )

    return {
        "Accuracy (%)": round(
            accuracy_score(y_actual, y_predicted) * 100, 2
        ),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "Macro F1 (%)": round(f1 * 100, 2)
    }

models = {
    "Logistic Regression": baseline_pred,
    "Full-Dataset mBERT": mbert_pred
}

comparison_rows = []

for name, predictions in models.items():
    scores = calculate_metrics(y_true, predictions)
    comparison_rows.append({
        "Model": name,
        **scores
    })

    print(f"\n{name}")
    print(scores)

    print(classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        zero_division=0
    ))

    cm = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{name} - Validation Confusion Matrix")
    plt.tight_layout()

    filename = (
        "baseline_confusion_matrix.png"
        if name == "Logistic Regression"
        else "mbert_full_confusion_matrix.png"
    )

    plt.savefig(RESULTS / filename, dpi=300)
    plt.close()

# --------------------------------------------------
# LANGUAGE-WISE PERFORMANCE
# --------------------------------------------------

language_rows = []

for code, group in df.groupby("language"):
    indices = group.index.to_numpy()
    actual = y_true[indices]

    for name, predictions in models.items():
        scores = calculate_metrics(
            actual, predictions[indices]
        )

        language_rows.append({
            "Language": LANGUAGE_NAMES.get(code, code),
            "Model": name,
            "Samples": len(indices),
            **scores
        })

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

comparison = pd.DataFrame(comparison_rows)
language_results = pd.DataFrame(language_rows)

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

print("\nLANGUAGE-WISE PERFORMANCE")
print(language_results.to_string(index=False))

comparison.to_csv(
    RESULTS / "final_model_comparison.csv",
    index=False
)

language_results.to_csv(
    RESULTS / "final_language_results.csv",
    index=False
)

prediction_results = df[
    ["language", "text", "label"]
].copy()

prediction_results["baseline_prediction"] = [
    "positive" if x == 1 else "negative"
    for x in baseline_pred
]

prediction_results["mbert_prediction"] = [
    "positive" if x == 1 else "negative"
    for x in mbert_pred
]

prediction_results.to_csv(
    RESULTS / "final_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nAll evaluation results saved in:", RESULTS)
print("\nPHASE 5 VALIDATION EVALUATION COMPLETED!")


# ==========================================================
# PHASE 5 - RESULTS VISUALIZATION
# ==========================================================

import matplotlib.pyplot as plt
import numpy as np

print("\nGenerating final evaluation visualizations...")

# 1. OVERALL ACCURACY COMPARISON

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    comparison["Model"],
    comparison["Accuracy (%)"],
    color=["#3498db", "#2ecc71"]
)

ax.set_title("Overall Model Accuracy Comparison")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{bar.get_height():.2f}%",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig(RESULTS / "01_accuracy_comparison.png", dpi=300)
plt.show()


# 2. PRECISION, RECALL AND F1 COMPARISON

metrics_to_plot = [
    "Precision (%)",
    "Recall (%)",
    "Macro F1 (%)"
]

x = np.arange(len(metrics_to_plot))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

for i, (_, row) in enumerate(comparison.iterrows()):
    values = [row[m] for m in metrics_to_plot]

    ax.bar(
        x + (i - 0.5) * width,
        values,
        width,
        label=row["Model"]
    )

ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot)
ax.set_ylim(0, 100)
ax.set_ylabel("Score (%)")
ax.set_title("Precision, Recall and Macro F1 Comparison")
ax.legend()

plt.tight_layout()
plt.savefig(RESULTS / "02_metrics_comparison.png", dpi=300)
plt.show()


# 3. LANGUAGE-WISE ACCURACY COMPARISON

language_pivot = language_results.pivot(
    index="Language",
    columns="Model",
    values="Accuracy (%)"
)

ax = language_pivot.plot(
    kind="bar",
    figsize=(11, 6),
    width=0.7
)

ax.set_title("Language-Wise Model Accuracy")
ax.set_xlabel("Language")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)

plt.xticks(rotation=0)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(RESULTS / "03_language_accuracy.png", dpi=300)
plt.show()


# 4. LOGISTIC REGRESSION CONFUSION MATRIX

from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(
    y_true,
    baseline_pred,
    labels=[0, 1],
    display_labels=["Negative", "Positive"],
    cmap="Blues",
    values_format="d",
    ax=ax
)

ax.set_title("Logistic Regression Confusion Matrix")

plt.tight_layout()
plt.savefig(RESULTS / "04_baseline_confusion.png", dpi=300)
plt.show()


# 5. FULL-DATASET mBERT CONFUSION MATRIX

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(
    y_true,
    mbert_pred,
    labels=[0, 1],
    display_labels=["Negative", "Positive"],
    cmap="Greens",
    values_format="d",
    ax=ax
)

ax.set_title("Full-Dataset mBERT Confusion Matrix")

plt.tight_layout()
plt.savefig(RESULTS / "05_mbert_confusion.png", dpi=300)
plt.show()

print("\nAll 5 visualizations generated successfully!")
print("Saved in:", RESULTS)