import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mscoco_label_map import category_index
from utils.timing import time_to_frame


TARGET_CLASSES = ("person", "dog", "cat", "horse", "cow")


def print_label_map_entries(label_map_path: Path) -> None:
    source_lines = label_map_path.read_text(encoding="utf-8").splitlines()

    for class_name in TARGET_CLASSES:
        class_id = next(
            class_id
            for class_id, entry in category_index.items()
            if entry["name"] == class_name
        )
        source_line_number = next(
            line_number
            for line_number, line in enumerate(source_lines, start=1)
            if f"'id': {class_id}" in line and f"'name': '{class_name}'" in line
        )
        print(
            f"{class_id}: {class_name} "
            f"({label_map_path.relative_to(PROJECT_ROOT)}:{source_line_number})"
        )


def print_rows(title: str, rows: pd.DataFrame) -> None:
    print(title)
    if rows.empty:
        print("No rows")
    else:
        print(rows.to_string(index=False))


def run_diagnostics(csv_path: Path, current_time: float, fps: float) -> None:
    video_info_df = pd.read_csv(csv_path)
    required_columns = {"class_str", "frame", "score"}
    missing_columns = required_columns.difference(video_info_df.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path}: missing required columns: {', '.join(sorted(missing_columns))}"
        )

    current_frame = time_to_frame(current_time, fps)
    print(f"CSV: {csv_path}")
    print(f"Time: {current_time} seconds; FPS: {fps}; frame: {current_frame}")
    print_rows(
        f"Rows at frame {current_frame}:",
        video_info_df[video_info_df["frame"] == current_frame]
    )
    print_rows(
        f"Rows at frames {current_frame - 3} through {current_frame + 3}:",
        video_info_df[video_info_df["frame"].between(current_frame - 3, current_frame + 3)]
    )

    print("Class counts:")
    print(video_info_df["class_str"].value_counts().to_string())

    print("Lowest and highest score per class:")
    print(video_info_df.groupby("class_str")["score"].agg(["min", "max"]).to_string())

    print("Class presence:")
    for class_name in TARGET_CLASSES:
        present = video_info_df["class_str"].eq(class_name).any()
        print(f"{class_name}: {'present' if present else 'absent'}")

    label_map_path = PROJECT_ROOT / "utils" / "mscoco_label_map.py"
    print("Label-map ID-to-name entries:")
    print_label_map_entries(label_map_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--time", required=True, type=float, help="Playback time in seconds.")
    parser.add_argument("--fps", default=24.0, type=float)
    args = parser.parse_args()

    run_diagnostics(args.csv, args.time, args.fps)


if __name__ == "__main__":
    main()
