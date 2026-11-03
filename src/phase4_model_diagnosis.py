from pathlib import Path
import pandas as pd
import joblib
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "mbert_improved"

torch.set_num_threads(4)

# Load existing models
baseline = joblib.load(
    ROOT / "models" / "baseline_logistic_regression.joblib"
)

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR)
)
model.eval()

# Telugu examples with known sentiment
examples = [
    ("ఈ సినిమా చాలా బాగుంది.", "positive"),
    ("ఈ సినిమా అద్భుతంగా ఉంది.", "positive"),
    ("నాకు ఈ సినిమా చాలా నచ్చింది.", "positive"),
    ("ఈ సినిమా చాలా చెత్తగా ఉంది.", "negative"),
    ("ఈ సినిమా అసలు బాగాలేదు.", "negative"),
    ("ఈ సినిమా చాలా నిరాశపరిచింది.", "negative"),
    ("ఈ సినిమా చాలా బాగుంది కానీ ముగింపు బాలేదు.", "negative"),
    ("ఈ సినిమా బాగాలేదు అనుకున్నాను కానీ చాలా నచ్చింది.", "positive"),
]

texts = [x[0] for x in examples]

# Logistic Regression
baseline_predictions = [
    str(x).lower()
    for x in baseline.predict(texts)
]

# mBERT batch prediction
encoded = tokenizer(
    texts,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=96
)

with torch.inference_mode():
    logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1)
    ids = probs.argmax(dim=-1).tolist()

mbert_predictions = [
    str(model.config.id2label[i]).lower()
    for i in ids
]

rows = []

for i, (text, expected) in enumerate(examples):
    rows.append({
        "text": text,
        "expected": expected,
        "baseline": baseline_predictions[i],
        "mbert": mbert_predictions[i],
        "mbert_confidence": round(
            probs[i, ids[i]].item(), 4
        ),
        "mbert_correct": (
            mbert_predictions[i] == expected
        )
    })

results = pd.DataFrame(rows)

print("\nTELUGU DIAGNOSTIC RESULTS\n")
print(results.to_string(index=False))

print("\nBaseline correct:",
      (results["baseline"] == results["expected"]).sum(),
      "/", len(results))

print("mBERT correct:",
      results["mbert_correct"].sum(),
      "/", len(results))

# Check the training corpus for Telugu sentiment terms
train = pd.read_csv(DATA / "train_clean.csv")
telugu = train[train["language"] == "te"].copy()

print("\nTELUGU TRAINING DATA CHECK")
print("Total Telugu records:", len(telugu))

for term in ["బాగుంది", "బాగాలేదు", "నచ్చింది", "చెత్తగా"]:
    matches = telugu[
        telugu["text"].astype(str).str.contains(
            term, regex=False, na=False
        )
    ]
    print(f"\nTerm: {term}")
    print("Matches:", len(matches))
    print(
        matches["label"].value_counts().to_string()
        if len(matches) else "No matches"
    )

output = ROOT / "results" / "telugu_diagnosis.csv"
output.parent.mkdir(exist_ok=True)

results.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:", output)