from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
TARGET = "default_next_year"
FEATURES = ["Attr27", "Attr21"]

# Membaca data yang sudah dibersihkan dari duplikat
data = pd.read_csv(
    PROJECT_DIR
    / "data"
    / "processed"
    / "polish_5year_deduplicated.csv"
)

rows = []

# Membuat ringkasan terpisah untuk perusahaan nonbangkrut dan bangkrut
for feature in FEATURES:
    for target_value, group_name in [
        (0, "non-bangkrut"),
        (1, "bangkrut"),
    ]:
        values = data.loc[data[TARGET] == target_value, feature]
        observed_values = values.dropna()

        rows.append({
            "fitur": feature,
            "kelompok": group_name,
            "jumlah_baris": len(values),
            "jumlah_missing": int(values.isna().sum()),
            "persen_missing": values.isna().mean() * 100,
            "jumlah_nilai_tersedia": len(observed_values),
            "persentil_05": observed_values.quantile(0.05),
            "kuartil_25": observed_values.quantile(0.25),
            "median": observed_values.median(),
            "kuartil_75": observed_values.quantile(0.75),
            "persentil_95": observed_values.quantile(0.95),
            "nilai_minimum": observed_values.min(),
            "nilai_maksimum": observed_values.max(),
        })

summary = pd.DataFrame(rows)

print(summary.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}",
))

# Menyimpan ringkasan agar dapat dipakai lagi dalam laporan
output_path = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "top_feature_profile.csv"
)
summary.to_csv(output_path, index=False)
print(f"\nRingkasan disimpan di: {output_path}")