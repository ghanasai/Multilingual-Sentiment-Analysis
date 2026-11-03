import pandas as pd
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT / "results"
PROCESSED_DIR = ROOT / "data" / "processed"

TRAIN_FILE = RESULTS_DIR / "phase1_language_predictions.csv"
VAL_FILE = PROCESSED_DIR / "validation_clean.csv"
TEST_FILE = PROCESSED_DIR / "test_clean.csv"

train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

print("=" * 60)
print("PHASE 1.8 - DATA LEAKAGE DETECTION")
print("=" * 60)

# Keep only positive and negative training examples.
train["Label"] = (
    train["Label"].astype("string").str.strip().str.lower()
)

train = train[
    train["Label"].isin(["positive", "negative"])
].copy()

print("\nBinary training candidates:", len(train))

# Normalize text for exact-overlap detection.
def normalize_text(text):
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()

    return text

train["normalized_text"] = train["Sentence"].apply(
    normalize_text
)

# Collect both English and Indic evaluation reviews.
evaluation_texts = set()

for df in [val, test]:
    for column in ["text", "english_text"]:
        evaluation_texts.update(
            df[column].dropna().apply(normalize_text)
        )

evaluation_texts.discard("")

print("Unique evaluation texts:", len(evaluation_texts))

# Find exact normalized overlaps.
train["overlaps_evaluation"] = (
    train["normalized_text"].isin(evaluation_texts)
)

print("\nExact evaluation overlaps:",
      train["overlaps_evaluation"].sum())

print("\nOverlaps by language:")
print(
    train.groupby("language")["overlaps_evaluation"]
    .sum()
)

# Identify duplicates within training candidates.
duplicate_count = train.duplicated(
    subset=["normalized_text"]
).sum()

print("\nDuplicate training sentences:", duplicate_count)

# Keep exact-overlap matches in an audit file.
overlaps = train[train["overlaps_evaluation"]].copy()

overlaps.to_csv(
    RESULTS_DIR / "phase1_exact_overlaps.csv",
    index=False,
    encoding="utf-8-sig"
)

# Remove exact evaluation overlaps.
safe_candidates = train[
    ~train["overlaps_evaluation"]
].copy()

# Remove repeated normalized sentences.
safe_candidates = safe_candidates.drop_duplicates(
    subset=["normalized_text"],
    keep="first"
)

safe_candidates = safe_candidates[
    safe_candidates["normalized_text"] != ""
].copy()

safe_candidates.to_csv(
    RESULTS_DIR / "phase1_screened_candidates.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nRemaining screened candidates:",
      len(safe_candidates))

print("\nScreened language distribution:")
print(safe_candidates["language"].value_counts())

print("\nScreened sentiment distribution:")
print(pd.crosstab(
    safe_candidates["language"],
    safe_candidates["Label"]
))

print("\nPHASE 1.8 EXACT-MATCH SCREENING COMPLETED!")