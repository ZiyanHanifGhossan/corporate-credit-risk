from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


PROJECT_DIR = Path(__file__).resolve().parents[1]

# Membaca hasil simulasi kapasitas dan data bersih
capacity = pd.read_csv(
    PROJECT_DIR
    / "data"
    / "processed"
    / "attr21_stress_capacity.csv"
)
data = pd.read_csv(
    PROJECT_DIR
    / "data"
    / "processed"
    / "polish_5year_deduplicated.csv"
)

TARGET = "default_next_year"

# Menghitung jumlah dan tingkat bangkrut untuk Attr21 tersedia atau missing
availability_groups = [
    ("Tersedia", data[data["Attr21"].notna()]),
    ("Missing", data[data["Attr21"].isna()]),
]

availability_summary = []

for group_name, group_data in availability_groups:
    availability_summary.append({
        "kelompok": group_name,
        "jumlah": len(group_data),
        "bangkrut": int(group_data[TARGET].sum()),
        "tingkat_bangkrut": group_data[TARGET].mean(),
    })

availability = pd.DataFrame(availability_summary)

# Membuat bidang gambar dengan tiga panel
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle(
    "Corporate Bankruptcy Risk: Capacity and Attr21 Sensitivity",
    fontsize=16,
    fontweight="bold",
)

model_labels = {
    "full": "Full model",
    "without_attr21": "Without Attr21",
}
model_colors = {
    "full": "#1769AA",
    "without_attr21": "#E68619",
}

# Panel 1: bagian kasus bangkrut yang tertangkap pada tiap kapasitas
for model_name, label in model_labels.items():
    subset = capacity[capacity["model"] == model_name].sort_values(
        "kapasitas"
    )

    axes[0].plot(
        subset["kapasitas"] * 100,
        subset["recall"],
        marker="o",
        linewidth=2.5,
        label=label,
        color=model_colors[model_name],
    )

    for _, row in subset.iterrows():
        label_offset = (0, 9) if model_name == "full" else (0, -16)
        label_alignment = "bottom" if model_name == "full" else "top"

        axes[0].annotate(
            f'{int(row["kasus_tertangkap"])} cases',
            (row["kapasitas"] * 100, row["recall"]),
            xytext=label_offset,
            textcoords="offset points",
            ha="center",
            va=label_alignment,
            fontsize=8,
        )

axes[0].set_title("Bankrupt cases captured")
axes[0].set_xlabel("Review capacity")
axes[0].set_ylabel("Share of all bankrupt cases")
axes[0].set_xticks([5, 10, 20])
axes[0].set_xticklabels(["5%", "10%", "20%"])
axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
axes[0].set_ylim(0, 1.05)
axes[0].grid(axis="y", alpha=0.25)
axes[0].legend(frameon=False)

# Panel 2: tingkat bangkrut di dalam daftar yang ditinjau
for model_name, label in model_labels.items():
    subset = capacity[capacity["model"] == model_name].sort_values(
        "kapasitas"
    )

    axes[1].plot(
        subset["kapasitas"] * 100,
        subset["tingkat_bangkrut_dalam_daftar"],
        marker="o",
        linewidth=2.5,
        label=label,
        color=model_colors[model_name],
    )

axes[1].set_title("Bankruptcy rate in reviewed group")
axes[1].set_xlabel("Review capacity")
axes[1].set_ylabel("Share of reviewed records that went bankrupt")
axes[1].set_xticks([5, 10, 20])
axes[1].set_xticklabels(["5%", "10%", "20%"])
axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
axes[1].set_ylim(0, 1.05)
axes[1].grid(axis="y", alpha=0.25)
axes[1].legend(frameon=False)

# Panel 3: tingkat bangkrut berdasarkan ketersediaan Attr21
bar_labels = [
    f'{row["kelompok"]}\nN={int(row["jumlah"])}'
    for _, row in availability.iterrows()
]
bar_rates = availability["tingkat_bangkrut"].tolist()
bar_colors = ["#7A8793", "#B54747"]

bars = axes[2].bar(
    bar_labels,
    bar_rates,
    color=bar_colors,
    width=0.6,
)

axes[2].set_title("Bankruptcy rate by Attr21 availability")
axes[2].set_ylabel("Observed bankruptcy rate")
axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
axes[2].set_ylim(0, 1.08)
axes[2].grid(axis="y", alpha=0.25)

for bar, (_, row) in zip(bars, availability.iterrows()):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.025,
        f'{int(row["bangkrut"])}/{int(row["jumlah"])} '
        f'({row["tingkat_bangkrut"]:.1%})',
        ha="center",
        va="bottom",
        fontsize=9,
    )

# Catatan kaki membedakan data grafik kapasitas dan data missingness
fig.text(
    0.01,
    0.015,
    "Capacity panels use the exploratory holdout. "
    "Attr21 availability panel uses the full deduplicated dataset. "
    "Descriptive analysis; not deployment validation.",
    fontsize=9,
)

fig.tight_layout(rect=[0, 0.07, 1, 0.92])

# Menyimpan gambar di folder docs
output_path = (
    PROJECT_DIR
    / "docs"
    / "model_capacity_and_attr21.png"
)
fig.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Grafik disimpan di: {output_path}")
