import pandas as pd
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "raw" / "training_dataset"
    / "Multilingual_sentiment_analysis.csv"
)

df = pd.read_csv(INPUT_FILE)

# Remove rows with missing sentences or labels
df = df.dropna(subset=["Sentence", "Label"]).copy()

df["Sentence"] = df["Sentence"].astype(str).str.strip()
df["Label"] = df["Label"].astype(str).str.strip().str.lower()

df = df[df["Sentence"] != ""]

def detect_script(text):
    counts = Counter()

    for char in text:
        code = ord(char)

        if 0x0900 <= code <= 0x097F:
            counts["Devanagari"] += 1
        elif 0x0C00 <= code <= 0x0C7F:
            counts["Telugu"] += 1
        elif 0x0B80 <= code <= 0x0BFF:
            counts["Tamil"] += 1
        elif 0x0D00 <= code <= 0x0D7F:
            counts["Malayalam"] += 1
        elif 0x0041 <= code <= 0x005A or 0x0061 <= code <= 0x007A:
            counts["Latin"] += 1
        elif char.isalpha():
            counts["Other script"] += 1

    if not counts:
        return "Unknown"

    return counts.most_common(1)[0][0]

df["detected_script"] = df["Sentence"].apply(detect_script)

print("=" * 60)
print("PHASE 1.6 - TRAINING LANGUAGE INSPECTION")
print("=" * 60)

print("\nTotal valid records:", len(df))

print("\nScript distribution:")
print(df["detected_script"].value_counts())

print("\nSentiment distribution by script:")
print(pd.crosstab(df["detected_script"], df["Label"]))

print("\nExample sentences for each script:")

for script in df["detected_script"].unique():
    print(f"\n--- {script} ---")

    examples = df[df["detected_script"] == script].head(3)

    for sentence in examples["Sentence"]:
        print(sentence[:200])

print("\nPHASE 1.6 INSPECTION COMPLETED!")