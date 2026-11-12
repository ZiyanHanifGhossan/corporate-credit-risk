from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
TARGET = "default_next_year"
FEATURES = ["Attr27", "Attr21"]

data = pd.read_csv(
    PROJECT_DIR
    / "data"
    / "processed"
    / "polish_5year_deduplicated.csv"
)

print(f"Proporsi bangkrut seluruh data: {data[TARGET].mean():.3%}")

all_results = []

for feature in FEATURES:
    # Pisahkan baris yang punya nilai rasio dari baris yang missing
    observed = data.loc[data[feature].notna(), [feature, TARGET]].copy()
    missing = data.loc[data[feature].isna(), [feature, TARGET]].copy()

    # Bagi nilai yang tersedia menjadi lima kelompok, dari nilai rendah ke tinggi
    quantile_numbers = pd.qcut(
        observed[feature],
        q=5,
        labels=False,
        duplicates="drop",
    )

    observed["kelompok_nilai"] = (
        "Q" + (quantile_numbers.astype(int) + 1).astype(str)
    )
    missing["kelompok_nilai"] = "Missing"

    # Gabungkan kembali agar missingness tampil sebagai kelompok tersendiri
    combined = pd.concat([observed, missing], ignore_index=True)

    summary = (
        combined.groupby("kelompok_nilai", observed=True)
        .agg(
            jumlah_perusahaan=(TARGET, "size"),
            jumlah_bangkrut=(TARGET, "sum"),
            tingkat_bangkrut=(TARGET, "mean"),
            median_rasio=(feature, "median"),
        )
        .reset_index()
    )

    # Urutkan kelompok nilai Q1 sampai Q5, lalu tampilkan Missing
    desired_order = ["Q1", "Q2", "Q3", "Q4", "Q5", "Missing"]
    summary["urutan"] = summary["kelompok_nilai"].map(
        {name: position for position, name in enumerate(desired_order)}
    )
    summary = summary.sort_values("urutan").drop(columns="urutan")
    summary.insert(0, "fitur", feature)

    print(f"\n=== {feature} ===")
    print(summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    all_results.append(summary)

# Simpan kedua tabel ke satu file
results = pd.concat(all_results, ignore_index=True)
output_path = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "top_feature_risk_bands.csv"
)
results.to_csv(output_path, index=False)
print(f"\nRingkasan disimpan di: {output_path}")