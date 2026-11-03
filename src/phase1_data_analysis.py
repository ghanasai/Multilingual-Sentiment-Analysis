
import json
from pathlib import Path
from collections import Counter

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "indic_sentiment"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

LANGUAGES = {
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "ml": "Malayalam"
}

all_records = []

print("=" * 65)
print("PHASE 1.3 - DATASET ANALYSIS")
print("=" * 65)

for split in ["validation", "test"]:

    print(f"\n{'=' * 20} {split.upper()} {'=' * 20}")

    for code, language in LANGUAGES.items():

        path = DATA_DIR / split / f"{code}.json"

        with open(path, "r", encoding="utf-8") as file:
            records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        df = pd.DataFrame(records)

        print(f"\nLanguage: {language}")
        print("Total records:", len(df))

        print("\nSentiment distribution:")
        print(df["LABEL"].value_counts(dropna=False))

        print("\nMissing values:")
        print(
            df[["INDIC REVIEW", "ENGLISH REVIEW", "LABEL"]]
            .isnull()
            .sum()
        )

        print(
            "Duplicate Indic reviews:",
            df["INDIC REVIEW"].duplicated().sum()
        )

        for record in records:
            all_records.append({
                "text": record["INDIC REVIEW"],
                "english_text": record["ENGLISH REVIEW"],
                "language": code,
                "label": str(record["LABEL"]).strip().lower(),
                "split": split,
                "source": "IndicSentiment"
            })

combined_df = pd.DataFrame(all_records)

print("\n" + "=" * 65)
print("COMBINED DATASET SUMMARY")
print("=" * 65)

print("\nTotal records:", len(combined_df))

print("\nOverall sentiment distribution:")
print(combined_df["label"].value_counts(dropna=False))

print("\nLanguage-wise sentiment distribution:")
print(
    pd.crosstab(
        combined_df["language"],
        combined_df["label"]
    )
)

print("\nSplit-wise sentiment distribution:")
print(
    pd.crosstab(
        combined_df["split"],
        combined_df["label"]
    )
)

print("\nUnique sentiment labels:")
print(sorted(combined_df["label"].unique()))

output_path = RESULTS_DIR / "phase1_dataset_summary.csv"

combined_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved dataset summary to:")
print(output_path)

print("\nPHASE 1.3 ANALYSIS COMPLETED SUCCESSFULLY!")
