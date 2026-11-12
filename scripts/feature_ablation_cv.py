from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
TARGET = "default_next_year"
TOP_FEATURES = ["Attr27", "Attr21"]

data = pd.read_csv(
    PROJECT_DIR
    / "data"
    / "processed"
    / "polish_5year_deduplicated.csv"
)

X = data.drop(columns=[TARGET])
y = data[TARGET]

# Menyiapkan lima versi fitur untuk membandingkan sumber sinyal
X_without_top = X.drop(columns=TOP_FEATURES)
X_top_values = X[TOP_FEATURES]
X_top_missing = X_top_values.isna().astype(int)

# Semua model memakai konfigurasi yang sama agar perbandingan fitur lebih adil
def make_hgb():
    return HistGradientBoostingClassifier(
        l2_regularization=0.0,
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        random_state=42,
    )


model_variants = {
    "Semua 64 fitur": (make_hgb(), X),
    "Tanpa Attr27 dan Attr21": (make_hgb(), X_without_top),
    "Tanpa Attr27": (make_hgb(), X.drop(columns=["Attr27"])),
    "Tanpa Attr21": (make_hgb(), X.drop(columns=["Attr21"])),
    "Hanya Attr27 dan Attr21, missing native": (make_hgb(), X_top_values),
    "Hanya indikator missing Attr27 dan Attr21": (make_hgb(), X_top_missing),
    "Hanya nilai Attr27 dan Attr21, diisi median": (
        make_pipeline(
            SimpleImputer(strategy="median"),
            make_hgb(),
        ),
        X_top_values,
    ),
}

# Setiap versi diuji pada pembagian fold yang sama
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

results = []

for variant_name, (model, feature_data) in model_variants.items():
    print(f"\nMenghitung: {variant_name}")

    scores = cross_validate(
        estimator=model,
        X=feature_data,
        y=y,
        cv=cv,
        scoring={
            "average_precision": "average_precision",
            "roc_auc": "roc_auc",
        },
        n_jobs=1,
    )

    ap_scores = scores["test_average_precision"]
    auc_scores = scores["test_roc_auc"]

    print(f"AP per fold: {ap_scores.round(3)}")
    print(f"Rata-rata AP: {ap_scores.mean():.3f}")
    print(f"ROC-AUC per fold: {auc_scores.round(3)}")
    print(f"Rata-rata ROC-AUC: {auc_scores.mean():.3f}")

    results.append({
        "versi_fitur": variant_name,
        "mean_average_precision": ap_scores.mean(),
        "std_average_precision": ap_scores.std(),
        "mean_roc_auc": auc_scores.mean(),
        "std_roc_auc": auc_scores.std(),
    })

results_df = pd.DataFrame(results)

output_path = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "feature_ablation_results.csv"
)
results_df.to_csv(output_path, index=False)

print(f"\nRingkasan disimpan di: {output_path}")