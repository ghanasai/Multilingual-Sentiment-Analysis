
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "indic_sentiment"

languages = {
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "ml": "Malayalam"
}

print("=" * 60)
print("PHASE 1 - DATASET INSPECTION")
print("=" * 60)

for split in ["validation", "test"]:
    print(f"\n--- {split.upper()} DATA ---")

    for code, language in languages.items():
        path = DATA_DIR / split / f"{code}.json"

        print(f"\nLanguage: {language}")

        if not path.exists():
            print("FILE NOT FOUND:", path)
            continue

        records = []

        try:
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        records.append(json.loads(line))

            print("Total records:", len(records))

            if records:
                print("Available columns:", list(records[0].keys()))
                print("First record:")
                print(json.dumps(
                    records[0],
                    ensure_ascii=False,
                    indent=2
                ))

        except Exception as error:
            print("Error:", error)

print("\nPHASE 1 DATASET INSPECTION COMPLETED")
