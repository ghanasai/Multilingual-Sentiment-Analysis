import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "raw" / "training_dataset"

csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV file found!")
    raise SystemExit

for file in csv_files:
    print("=" * 60)
    print("FILE:", file.name)
    print("=" * 60)

    df = pd.read_csv(file, low_memory=False)

    print("\nTotal records:", len(df))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head(5).to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum())

    for column in df.columns:
        if any(word in column.lower()
               for word in ["language", "sentiment", "label", "source"]):

            print(f"\nValue counts for {column}:")
            print(df[column].value_counts(dropna=False).head(25))

print("\nTRAINING DATASET INSPECTION COMPLETED!")