import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("PHASE 2 - TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)

# Load training and validation datasets
train = pd.read_csv(DATA_DIR / "train_clean.csv")
val = pd.read_csv(DATA_DIR / "validation_clean.csv")

# Keep valid binary labels
valid_labels = ["positive", "negative"]

train = train[
    train["label"].isin(valid_labels)
].dropna(subset=["text", "label"])

val = val[
    val["label"].isin(valid_labels)
].dropna(subset=["text", "label"])

X_train = train["text"].astype(str)
y_train = train["label"]

X_val = val["text"].astype(str)
y_val = val["label"]

print("\nTraining samples:", len(train))
print("Validation samples:", len(val))

# Character n-grams work across different scripts.
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

print("\nTraining model...")
model.fit(X_train, y_train)

print("Training completed!")

# Predict validation sentiments
predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )
)

print("\nVALIDATION RESULTS")
print("-" * 60)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification report:")
print(classification_report(
    y_val, predictions, zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_val,
    predictions,
    labels=["negative", "positive"]
))

# Save model
model_path = MODEL_DIR / "baseline_logistic_regression.joblib"

joblib.dump(model, model_path)

print("\nModel saved to:", model_path)

# Save predictions
results = val[["language", "text", "label"]].copy()
results["prediction"] = predictions

results.to_csv(
    RESULTS_DIR / "baseline_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

# Evaluate each language separately
print("\nLANGUAGE-WISE VALIDATION RESULTS")
print("-" * 60)

for language in sorted(results["language"].unique()):
    subset = results[results["language"] == language]

    lang_accuracy = accuracy_score(
        subset["label"], subset["prediction"]
    )

    _, _, lang_f1, _ = precision_recall_fscore_support(
        subset["label"],
        subset["prediction"],
        average="macro",
        zero_division=0
    )

    print(
        f"{language}: Accuracy={lang_accuracy:.4f}, "
        f"Macro F1={lang_f1:.4f}"
    )

print("\nPHASE 2 BASELINE MODEL COMPLETED!")