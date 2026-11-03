import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "results" /
    "phase1_dataset_summary.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 1.4 - DATASET CLEANING")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nOriginal records:", len(df))

# Convert labels to lowercase safely
df["label"] = (
    df["label"]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Keep only valid sentiment labels
valid_labels = ["positive", "negative"]

invalid_count = (~df["label"].isin(valid_labels)).sum()

df = df[df["label"].isin(valid_labels)].copy()

print("Records with invalid/missing labels:", invalid_count)

# Remove missing or empty reviews
df = df.dropna(subset=["text"])

df["text"] = df["text"].astype(str).str.strip()

df = df[df["text"] != ""].copy()

# Inspect duplicates without silently removing
# repeated records from official evaluation splits.
duplicate_count = df.duplicated(
    subset=["language", "split", "text"]
).sum()

print("Duplicate records within language/split:",
      duplicate_count)

# Check potential cross-split overlap
overlap_count = 0

for language in sorted(df["language"].unique()):
    subset = df[df["language"] == language]

    val_texts = set(
        subset[subset["split"] == "validation"]["text"]
    )

    test_texts = set(
        subset[subset["split"] == "test"]["text"]
    )

    overlap = len(val_texts.intersection(test_texts))
    overlap_count += overlap

    print(f"{language} validation/test overlap: {overlap}")

# Save cleaned splits separately
for split in ["validation", "test"]:
    split_df = df[df["split"] == split].copy()

    output_path = OUTPUT_DIR / f"{split}_clean.csv"

    split_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved {split}: {len(split_df)} records")

print("\nFinal total records:", len(df))

print("\nSentiment distribution:")
print(df["label"].value_counts())

print("\nLanguage-wise distribution:")
print(pd.crosstab(df["language"], df["label"]))

print("\nPHASE 1.4 COMPLETED SUCCESSFULLY!")