"""Remove exact feature-vector/target duplicates before a train/validation split."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/polish_5year.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/polish_5year_deduplicated.csv"))
    parser.add_argument("--audit", type=Path, default=Path("data/processed/duplicate_record_audit.csv"))
    args = parser.parse_args()

    seen: dict[tuple[str, ...], int] = {}
    retained: list[dict[str, str]] = []
    duplicate_rows: list[dict[str, str]] = []
    with args.input.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        feature_fields = [name for name in fields if name != "default_next_year"]
        for source_row, row in enumerate(reader, start=2):
            signature = tuple(row[name] for name in feature_fields) + (row["default_next_year"],)
            if signature in seen:
                duplicate_rows.append({
                    "duplicate_source_row": str(source_row),
                    "first_seen_source_row": str(seen[signature]),
                    "default_next_year": row["default_next_year"],
                })
            else:
                seen[signature] = source_row
                retained.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained)
    with args.audit.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["duplicate_source_row", "first_seen_source_row", "default_next_year"])
        writer.writeheader()
        writer.writerows(duplicate_rows)
    removed_events = sum(row["default_next_year"] == "1" for row in duplicate_rows)
    print(f"Retained: {len(retained):,}; removed exact duplicates: {len(duplicate_rows):,}; duplicate events: {removed_events:,}")
    print(f"Wrote {args.output} and {args.audit}")


if __name__ == "__main__":
    main()
