from __future__ import annotations

import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def analyze_engagement(csv_path: str) -> None:
    # Колонки (без header в файле)
    columns = [
        "timestamp",
        "track_id",
        "name",
        "label",
        "score",
        "attentive_ratio",
        "class_avg_attention",
        "class_avg_score",
    ]

    # Чтение CSV без header
    df = pd.read_csv(csv_path)


    df = df[df["name"].notna()]
    df = df[df["name"] != "name"]
    # Определяем attentive (1) / not attentive (0)
    df["is_attentive"] = df["label"].apply(lambda x: 1 if x == "attentive" else 0)

    # Группировка по ученикам
    grouped = df.groupby("name").agg(
        total_frames=("is_attentive", "count"),
        attentive_frames=("is_attentive", "sum"),
    )

    # Процент внимания
    grouped["attention_percent"] = (
        grouped["attentive_frames"] / grouped["total_frames"] * 100
    )

    # Сортировка
    grouped = grouped.sort_values(by="attention_percent", ascending=False)

    # ===== PLOT =====
    plt.figure()

    grouped["attention_percent"].plot(kind="bar")

    plt.xlabel("Students")
    plt.ylabel("Attention (%)")
    plt.title("Student Attention Overview")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_dir, f"engagement_report_{timestamp}.png")

    plt.savefig(output_file)
    print(f"Report saved to: {output_file}")
    
    # Также можно вывести таблицу
    print("\nAttention summary:\n")
    print(grouped[["attention_percent"]])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.analyze_engagement <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    analyze_engagement(csv_path)