import pandas as pd

file_path = "data/processed/polish_5year_deduplicated.csv"
df = pd.read_csv(file_path)

print("Bentuk data (baris, kolom):", df.shape)

print("\nJumlah untuk masing-masing target:")
print(df["default_next_year"].value_counts().sort_index())

print(
    "\nPersentase kasus bangkrut:",
    round(df["default_next_year"].mean() * 100, 2),
    "%",
)

missing = df.isna().sum().sort_values(ascending=False)
print("\nKolom dengan missing terbanyak:")
print(missing[missing > 0].head(10))

print("\nContoh tiga baris:")
print(df.head(3).to_string(index=False))

features = [column for column in df.columns if column != "default_next_year"]
has_missing = df[features].isna().any(axis=1)

missing_summary = pd.DataFrame({
    "target": df["default_next_year"],
    "has_missing": has_missing,
})

summary = missing_summary.groupby("target")["has_missing"].agg(
    ["sum", "mean", "count"]
)
summary["mean"] = (summary["mean"] * 100).round(1)

print("\nBaris dengan minimal satu missing, per target:")
print(summary.rename(columns={
    "sum": "jumlah_baris_missing",
    "mean": "persentase_baris_missing",
    "count": "total_baris",
}))

for column in ["Attr37", "Attr27", "Attr21"]:
    print(f"\nMissing pada {column}:")
    for target, label in [(0, "non-bangkrut"), (1, "bangkrut")]:
        subset = df[df["default_next_year"] == target]
        percentage = subset[column].isna().mean() * 100
        print(f"  {label}: {percentage:.1f}%")