"""Convert the selected UCI ARFF subset to an auditable CSV.

Missing ARFF values (?) are emitted as empty CSV cells. The class label is
renamed to default_next_year. This converter does not impute, cap, or scale data.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ATTRIBUTE = re.compile(r"^@attribute\s+([^\s]+)\s+", re.IGNORECASE)


def convert(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    attributes: list[str] = []
    rows: list[list[str]] = []
    in_data = False
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            if not in_data:
                match = ATTRIBUTE.match(line)
                if match:
                    attributes.append(match.group(1).strip("'\""))
                elif line.lower() == "@data":
                    in_data = True
                continue
            values = next(csv.reader([line]))
            values = [value.strip() for value in values]
            if len(values) != len(attributes):
                raise ValueError(f"Expected {len(attributes)} fields, found {len(values)}")
            if values[-1] not in {"0", "1"}:
                raise ValueError(f"Unexpected target value: {values[-1]!r}")
            rows.append(["" if value == "?" else value for value in values[:-1]] + [values[-1]])

    if not in_data or not attributes or attributes[-1].lower() != "class":
        raise ValueError("ARFF data section or final class attribute is missing")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = attributes[:-1] + ["default_next_year"]
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)
    positives = sum(row[-1] == "1" for row in rows)
    missing_features = sum(value == "" for row in rows for value in row[:-1])
    return len(rows), positives, missing_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/polish_companies_5year.arff"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/polish_5year.csv"))
    args = parser.parse_args()
    rows, positives, missing = convert(args.input, args.output)
    print(f"Rows: {rows:,}; one-year bankruptcy events: {positives:,}; missing feature cells: {missing:,}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
