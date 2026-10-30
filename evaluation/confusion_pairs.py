import argparse
from itertools import combinations
import math
from pathlib import Path

import pandas as pd


THRESHOLD = 0.2
CONFUSABLE_PAIRS = [
    ("horse", "cow"),
    ("dog", "cat"),
    ("sheep", "goat"),
    ("deer", "horse"),
]
REQUIRED_COLUMNS = {"class_str", "frame", "score"}


def analyze_detections(detections: pd.DataFrame, source: str = "<data>") -> dict:
    missing = REQUIRED_COLUMNS.difference(detections.columns)
    if missing:
        raise ValueError(f"{source}: missing required columns: {', '.join(sorted(missing))}")

    data = detections.copy()
    data["frame"] = pd.to_numeric(data["frame"], errors="coerce")
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    if data["frame"].isna().any() or data["score"].isna().any():
        raise ValueError(f"{source}: frame and score must contain numeric values")

    pair_frame_counts = {}
    candidates = []
    for frame, frame_rows in data.groupby("frame", sort=True):
        try:
            frame_value = float(str(frame))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source}: frame values must be integers") from error
        if not math.isfinite(frame_value) or not frame_value.is_integer():
            raise ValueError(f"{source}: frame values must be integers")
        frame_number = int(frame_value)

        classes = set(frame_rows["class_str"].astype(str))
        for pair in combinations(sorted(classes), 2):
            pair_frame_counts[pair] = pair_frame_counts.get(pair, 0) + 1

        high_rows = frame_rows.loc[frame_rows["score"] > THRESHOLD]
        high_classes = set(high_rows["class_str"].astype(str))
        for class_a, class_b in CONFUSABLE_PAIRS:
            if class_a in classes and class_b in classes:
                above_threshold = sorted({class_a, class_b} & high_classes)
                if above_threshold:
                    candidates.append(
                        {
                            "file": source,
                            "frame": frame_number,
                            "pair": f"{class_a}/{class_b}",
                            "classes_present": ";".join(sorted({class_a, class_b})),
                            "high_confidence_classes": ";".join(above_threshold),
                        }
                    )

    return {
        "class_counts": data["class_str"].value_counts(),
        "pair_frame_counts": pair_frame_counts,
        "candidates": pd.DataFrame(
            candidates,
            columns=[
                "file",
                "frame",
                "pair",
                "classes_present",
                "high_confidence_classes",
            ],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find same-frame class co-occurrences for candidate review."
    )
    parser.add_argument("detection_csvs", nargs="+", type=Path)
    args = parser.parse_args()

    for csv_path in args.detection_csvs:
        result = analyze_detections(pd.read_csv(csv_path), str(csv_path))
        print(f"\n{csv_path}")
        print("Class counts:")
        print(result["class_counts"].to_string() or "No detections")
        print("Class-pair shared-frame counts:")
        if result["pair_frame_counts"]:
            for pair, count in sorted(result["pair_frame_counts"].items()):
                print(f"  {pair[0]}/{pair[1]}: {count}")
        else:
            print("  No class pairs share a frame.")
        print(f"Confusable-pair candidates (either class score > {THRESHOLD:g}; review only):")
        if result["candidates"].empty:
            print("  No candidate frames.")
        else:
            print(result["candidates"].to_string(index=False))


if __name__ == "__main__":
    main()
