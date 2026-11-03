import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "data" / "processed" / "train_clean.csv"

train = pd.read_csv(
    RESULTS / "phase1_screened_candidates.csv"
)

raw = pd.read_csv(
    RESULTS / "phase1_language_predictions.csv"
)

near = pd.read_csv(
    RESULTS / "phase1_near_duplicate_review.csv"
)

def normalize(series):
    return (
        series.astype(str)
        .str.normalize("NFKC")
        .str.casefold()
        .str.split()
        .str.join(" ")
    )

raw["norm"] = normalize(raw["Sentence"])
raw["Label"] = raw["Label"].str.lower()

raw = raw[raw["Label"].isin(["positive", "negative"])]

conflicts = raw.groupby("norm")["Label"].nunique()
conflict_texts = set(conflicts[conflicts > 1].index)

train["norm"] = normalize(train["Sentence"])

near_texts = (
    set(near["norm"])
    if "norm" in near.columns
    else set()
)

excluded = conflict_texts | near_texts

train = train[~train["norm"].isin(excluded)].copy()

train = train.rename(
    columns={"Sentence": "text", "Label": "label"}
)

train[["text", "label", "language"]].to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("Final training records:", len(train))
print("\nLanguage distribution:")
print(train["language"].value_counts())
print("\nSaved:", OUTPUT)
print("\nTRAINING DATA PREPARATION COMPLETED!")