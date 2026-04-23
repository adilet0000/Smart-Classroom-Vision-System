from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/smart_classroom_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/smart_classroom_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ATTENTIVE_LABEL = "attentive"
DEFAULT_BIN_SECONDS = 30


def analyze_engagement(
    csv_path: str,
    output_dir: str = "reports",
    bin_seconds: int = DEFAULT_BIN_SECONDS,
    db_path: str = "data/faces.db",
) -> None:
    source_path = Path(csv_path)
    df = _read_log(source_path)
    roster = _load_roster(Path(db_path))

    report_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if df.empty:
        if roster:
            absent = pd.DataFrame(
                {
                    "name": roster,
                    "attendance_status": "absent",
                    "first_seen": "",
                    "last_seen": "",
                    "observations": 0,
                    "observed_minutes": 0.0,
                    "attentive_minutes": 0.0,
                    "attention_percent": 0.0,
                    "average_score": 0.0,
                }
            )
            summary_file = output_path / f"engagement_summary_{report_id}.csv"
            absent.to_csv(summary_file, index=False)
        else:
            summary_file = None

        empty_report = output_path / f"engagement_report_{report_id}.md"
        content = (
            "# Engagement Report\n\n"
            f"Source: `{source_path}`\n\n"
            "No recognized student records were found in this log.\n"
        )
        if summary_file is not None:
            content += f"\nRoster attendance summary saved to: `{summary_file}`\n"
            content += "\n## Attendance Summary\n\n"
            content += _to_markdown_table(absent)
            content += "\n"
        empty_report.write_text(content, encoding="utf-8")
        print("No recognized students in this log.")
        print(f"Empty report saved to: {empty_report}")
        if summary_file is not None:
            print(f"Roster attendance summary saved to: {summary_file}")
        return

    df = _prepare_time_columns(df)
    summary = _build_student_summary(df, roster)
    label_distribution = _build_label_distribution(df)
    timeline = _build_timeline(df, bin_seconds)

    summary_file = output_path / f"engagement_summary_{report_id}.csv"
    labels_file = output_path / f"engagement_labels_{report_id}.csv"
    timeline_file = output_path / f"engagement_timeline_{report_id}.csv"
    figure_file = output_path / f"engagement_report_{report_id}.png"
    markdown_file = output_path / f"engagement_report_{report_id}.md"

    summary.to_csv(summary_file, index=False)
    label_distribution.to_csv(labels_file)
    timeline.to_csv(timeline_file, index=False)
    _plot_dashboard(summary, label_distribution, timeline, figure_file)
    _write_markdown_report(
        source_path=source_path,
        output_file=markdown_file,
        df=df,
        summary=summary,
        label_distribution=label_distribution,
        timeline=timeline,
        figure_file=figure_file,
        summary_file=summary_file,
        labels_file=labels_file,
        timeline_file=timeline_file,
    )

    print(f"Dashboard saved to: {figure_file}")
    print(f"Summary CSV saved to: {summary_file}")
    print(f"Timeline CSV saved to: {timeline_file}")
    print(f"Label distribution CSV saved to: {labels_file}")
    print(f"Markdown report saved to: {markdown_file}")
    print("\nAttendance and attention summary:\n")
    print(summary.to_string(index=False))


def _read_log(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "session_id" not in df.columns:
        df["session_id"] = "legacy"

    for column in [
        "score",
        "class_avg_attention",
        "class_avg_score",
        "person_confidence",
        "face_confidence",
        "yaw",
        "pitch",
        "roll",
        "eye_aspect_ratio",
        "gaze_x",
        "gaze_y",
        "gaze_score",
        "body_tilt",
    ]:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["student_code", "reason", "eyes_closed", "body_visible"]:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[df["name"].notna()]
    df = df[df["name"] != "name"]
    df = df[df["timestamp"].notna()]

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()]
    df = df.sort_values(["name", "timestamp"]).reset_index(drop=True)
    return df


def _prepare_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_attentive"] = (df["label"] == ATTENTIVE_LABEL).astype(int)

    next_ts = df.groupby("name")["timestamp"].shift(-1)
    delta = (next_ts - df["timestamp"]).dt.total_seconds()
    positive_delta = delta[delta > 0]
    median_delta = float(positive_delta.median()) if not positive_delta.empty else 1.0
    median_delta = max(median_delta, 1.0)

    df["sample_seconds"] = delta.fillna(median_delta)
    df.loc[df["sample_seconds"] <= 0, "sample_seconds"] = median_delta
    df["sample_seconds"] = df["sample_seconds"].clip(upper=median_delta * 3)
    df["attentive_seconds"] = df["sample_seconds"] * df["is_attentive"]
    return df


def _load_roster(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT full_name
                FROM students
                ORDER BY full_name
                """
            ).fetchall()
    except sqlite3.Error:
        return []

    return [str(row[0]) for row in rows if row and row[0]]


def _build_student_summary(df: pd.DataFrame, roster: list[str]) -> pd.DataFrame:
    grouped = df.groupby("name", as_index=False).agg(
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        observations=("timestamp", "count"),
        observed_seconds=("sample_seconds", "sum"),
        attentive_seconds=("attentive_seconds", "sum"),
        average_score=("score", "mean"),
        avg_yaw=("yaw", "mean"),
        avg_pitch=("pitch", "mean"),
        avg_gaze_score=("gaze_score", "mean"),
        avg_person_confidence=("person_confidence", "mean"),
        avg_face_confidence=("face_confidence", "mean"),
    )

    grouped["attention_percent"] = _safe_percent(
        grouped["attentive_seconds"],
        grouped["observed_seconds"],
    )
    grouped["observed_minutes"] = grouped["observed_seconds"] / 60.0
    grouped["attentive_minutes"] = grouped["attentive_seconds"] / 60.0
    grouped["attendance_status"] = "present"
    grouped["first_seen"] = grouped["first_seen"].dt.strftime("%H:%M:%S")
    grouped["last_seen"] = grouped["last_seen"].dt.strftime("%H:%M:%S")

    if roster:
        present_names = set(grouped["name"])
        absent_names = [name for name in roster if name not in present_names]
        if absent_names:
            absent = pd.DataFrame(
                {
                    "name": absent_names,
                    "attendance_status": "absent",
                    "first_seen": "",
                    "last_seen": "",
                    "observations": 0,
                    "observed_minutes": 0.0,
                    "attentive_minutes": 0.0,
                    "attention_percent": 0.0,
                    "average_score": 0.0,
                    "avg_yaw": pd.NA,
                    "avg_pitch": pd.NA,
                    "avg_gaze_score": pd.NA,
                    "avg_person_confidence": pd.NA,
                    "avg_face_confidence": pd.NA,
                }
            )
            grouped = pd.concat([grouped, absent], ignore_index=True)

    columns = [
        "name",
        "attendance_status",
        "first_seen",
        "last_seen",
        "observations",
        "observed_minutes",
        "attentive_minutes",
        "attention_percent",
        "average_score",
        "avg_yaw",
        "avg_pitch",
        "avg_gaze_score",
        "avg_person_confidence",
        "avg_face_confidence",
    ]
    grouped["_present_rank"] = (grouped["attendance_status"] == "present").astype(int)
    return (
        grouped[columns + ["_present_rank"]]
        .sort_values(["_present_rank", "attention_percent", "average_score"], ascending=False)
        .drop(columns="_present_rank")
    )


def _build_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    weighted = (
        df.groupby(["name", "label"])["sample_seconds"]
        .sum()
        .reset_index(name="seconds")
    )
    pivot = weighted.pivot_table(
        index="name",
        columns="label",
        values="seconds",
        aggfunc="sum",
        fill_value=0.0,
    )
    total = pivot.sum(axis=1).replace(0, pd.NA)
    return (pivot.div(total, axis=0) * 100).fillna(0.0).sort_index()


def _build_timeline(df: pd.DataFrame, bin_seconds: int) -> pd.DataFrame:
    df = df.copy()
    bin_seconds = max(1, int(bin_seconds))
    df["time_bin"] = df["timestamp"].dt.floor(f"{bin_seconds}s")

    student_timeline = (
        df.groupby(["time_bin", "name"], as_index=False)
        .apply(_weighted_timeline_row, include_groups=False)
        .reset_index(drop=True)
    )

    class_timeline = (
        df.groupby("time_bin", as_index=False)
        .apply(_weighted_timeline_row, include_groups=False)
        .reset_index(drop=True)
    )
    class_timeline["name"] = "__class_average__"

    timeline = pd.concat([student_timeline, class_timeline], ignore_index=True)
    timeline["time_bin"] = timeline["time_bin"].dt.strftime("%H:%M:%S")
    return timeline.sort_values(["time_bin", "name"])


def _weighted_timeline_row(group: pd.DataFrame) -> pd.Series:
    seconds = group["sample_seconds"].sum()
    if seconds <= 0:
        attention_percent = 0.0
        average_score = float(group["score"].mean())
    else:
        attention_percent = float((group["attentive_seconds"].sum() / seconds) * 100)
        average_score = float((group["score"].fillna(0.0) * group["sample_seconds"]).sum() / seconds)

    return pd.Series(
        {
            "attention_percent": attention_percent,
            "average_score": average_score,
            "observed_seconds": float(seconds),
        }
    )


def _plot_dashboard(
    summary: pd.DataFrame,
    label_distribution: pd.DataFrame,
    timeline: pd.DataFrame,
    output_file: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Smart Classroom Engagement Report", fontsize=16)

    top = summary.sort_values("attention_percent", ascending=True)
    axes[0, 0].barh(top["name"], top["attention_percent"], color="#2d8f6f")
    axes[0, 0].set_title("Attention by Student")
    axes[0, 0].set_xlabel("Attention (%)")
    axes[0, 0].set_xlim(0, 100)

    axes[0, 1].barh(top["name"], top["observed_minutes"], color="#4f7cac")
    axes[0, 1].set_title("Observed Time by Student")
    axes[0, 1].set_xlabel("Minutes")

    class_timeline = timeline[timeline["name"] == "__class_average__"].copy()
    if not class_timeline.empty:
        axes[1, 0].plot(
            class_timeline["time_bin"],
            class_timeline["attention_percent"],
            marker="o",
            color="#2d8f6f",
            label="attention %",
        )
        axes[1, 0].plot(
            class_timeline["time_bin"],
            class_timeline["average_score"] * 100,
            marker="o",
            color="#c65d32",
            label="score x100",
        )
    axes[1, 0].set_title("Class Timeline")
    axes[1, 0].set_ylabel("Percent")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].legend(loc="lower left")

    label_distribution.plot(kind="bar", stacked=True, ax=axes[1, 1], colormap="tab20")
    axes[1, 1].set_title("Behavior Distribution by Student")
    axes[1, 1].set_ylabel("Time (%)")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def _write_markdown_report(
    source_path: Path,
    output_file: Path,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    label_distribution: pd.DataFrame,
    timeline: pd.DataFrame,
    figure_file: Path,
    summary_file: Path,
    labels_file: Path,
    timeline_file: Path,
) -> None:
    session_id = str(df["session_id"].iloc[0])
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    duration_minutes = max((end - start).total_seconds(), 0.0) / 60.0
    class_rows = timeline[timeline["name"] == "__class_average__"]
    class_attention = float(class_rows["attention_percent"].mean()) if not class_rows.empty else 0.0
    class_score = float(class_rows["average_score"].mean()) if not class_rows.empty else 0.0

    best = summary.iloc[0]
    needs_attention = summary.sort_values(["attention_percent", "average_score"]).head(3)

    lines = [
        "# Engagement Report",
        "",
        f"Source: `{source_path}`",
        f"Session ID: `{session_id}`",
        f"Time range: {start.strftime('%Y-%m-%d %H:%M:%S')} - {end.strftime('%H:%M:%S')}",
        f"Logged duration: {duration_minutes:.1f} min",
        f"Recognized students: {summary['name'].nunique()}",
        f"Class attention: {class_attention:.1f}%",
        f"Class average score: {class_score:.2f}",
        "",
        "## Files",
        "",
        f"- Dashboard: `{figure_file}`",
        f"- Student summary: `{summary_file}`",
        f"- Timeline: `{timeline_file}`",
        f"- Label distribution: `{labels_file}`",
        "",
        "## Top Student",
        "",
        (
            f"{best['name']}: {best['attention_percent']:.1f}% attention, "
            f"{best['observed_minutes']:.1f} observed minutes, score {best['average_score']:.2f}."
        ),
        "",
        "## Students Needing Review",
        "",
    ]

    for _, row in needs_attention.iterrows():
        lines.append(
            f"- {row['name']}: {row['attention_percent']:.1f}% attention, "
            f"{row['observed_minutes']:.1f} observed minutes, score {row['average_score']:.2f}"
        )

    lines.extend(
        [
            "",
            "## Student Summary",
            "",
            _to_markdown_table(summary),
            "",
            "## Label Distribution (%)",
            "",
            _to_markdown_table(label_distribution.reset_index()),
            "",
        ]
    )

    output_file.write_text("\n".join(lines), encoding="utf-8")


def _to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )

    headers = [str(column) for column in display.columns]
    rows = display.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")

    return "\n".join(lines)


def _safe_percent(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.replace(0, pd.NA)
    return ((numerator / safe_denominator) * 100).fillna(0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to session CSV log")
    parser.add_argument("--output-dir", default="reports", help="Directory for generated reports")
    parser.add_argument(
        "--bin-seconds",
        type=int,
        default=DEFAULT_BIN_SECONDS,
        help="Timeline bucket size in seconds",
    )
    parser.add_argument(
        "--db-path",
        default="data/faces.db",
        help="SQLite face database path for roster-based attendance summary",
    )
    args = parser.parse_args()

    analyze_engagement(args.csv_path, args.output_dir, args.bin_seconds, args.db_path)


if __name__ == "__main__":
    main()
