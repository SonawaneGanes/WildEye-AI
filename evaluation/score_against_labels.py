import argparse
import math
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LABEL_COLUMNS = {"video", "frame", "true_classes", "notes"}
DETECTION_COLUMNS = {"class_str", "frame", "score"}
REPORT_COLUMNS = [
    "video",
    "frame",
    "true_classes",
    "detected_classes",
    "missed_true_classes",
    "wrong_class_predictions",
    "false_positives",
    "missed_count",
    "wrong_class_count",
    "false_positive_count",
]


def _validate_columns(df: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{source}: missing required columns: {', '.join(sorted(missing))}")


def score_against_labels(
    labels_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    threshold: float = 0.2,
) -> pd.DataFrame:
    """Score labeled frames against detections above the strict confidence threshold."""
    _validate_columns(labels_df, LABEL_COLUMNS, "labels")
    _validate_columns(detections_df, DETECTION_COLUMNS, "detections")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    detections = detections_df.copy()
    detections["frame"] = pd.to_numeric(detections["frame"], errors="coerce")
    detections["score"] = pd.to_numeric(detections["score"], errors="coerce")
    if detections["frame"].isna().any() or detections["score"].isna().any():
        raise ValueError("detections: frame and score must contain numeric values")

    has_true_classes = labels_df["true_classes"].map(
        lambda value: pd.notna(value) and bool(str(value).strip())
    )
    labeled_frames = labels_df.loc[labels_df["frame"].notna() & has_true_classes]
    report = []
    for row in labeled_frames.itertuples(index=False):
        try:
            frame_value = float(str(row.frame))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"labels: frame must be an integer value, got {row.frame!r}"
            ) from error
        if not math.isfinite(frame_value) or not frame_value.is_integer():
            raise ValueError(f"labels: frame must be an integer value, got {row.frame!r}")
        frame = int(frame_value)
        true_classes = {
            class_name.strip()
            for class_name in str(row.true_classes).split(";")
            if class_name.strip()
        }
        frame_detections = detections.loc[
            (detections["frame"] == frame) & (detections["score"] > threshold)
        ]
        detected_names: list[str] = [
            str(class_name) for class_name in frame_detections["class_str"].tolist()
        ]
        predicted_classes = set(detected_names)
        missed = sorted(true_classes - predicted_classes)
        detection_scores: list[float] = [
            float(str(score)) for score in frame_detections["score"].tolist()
        ]
        false_positives: list[tuple[str, float]] = []
        for class_name, score in zip(detected_names, detection_scores):
            if class_name not in true_classes:
                false_positives.append((class_name, score))
        wrong_predictions = false_positives if missed else []
        report.append(
            {
                "video": row.video,
                "frame": frame,
                "true_classes": ";".join(sorted(true_classes)),
                "detected_classes": ";".join(sorted(predicted_classes)),
                "missed_true_classes": ";".join(missed),
                "wrong_class_predictions": ";".join(
                    class_name for class_name, _ in wrong_predictions
                ),
                "false_positives": ";".join(
                    f"{class_name}@{score:g}" for class_name, score in false_positives
                ),
                "missed_count": len(missed),
                "wrong_class_count": len(wrong_predictions),
                "false_positive_count": len(false_positives),
            }
        )
    return pd.DataFrame(report, columns=REPORT_COLUMNS)


def _labels_for_video(labels_df: pd.DataFrame, video_path: Path, selector: str | None):
    video_names = {video_path.name, video_path.stem}
    if selector:
        video_names.add(selector)
    return labels_df.loc[labels_df["video"].astype(str).isin(video_names)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure detection results against manually labeled frames."
    )
    parser.add_argument("detection_csv", type=Path)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).parent / "labels" / "labels_template.csv",
    )
    parser.add_argument("--video", help="Video value in the labels file.")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--out", type=Path, help="Optional path for the report CSV.")
    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels)
    selected_labels = _labels_for_video(labels_df, args.detection_csv, args.video)
    result = score_against_labels(
        selected_labels, pd.read_csv(args.detection_csv), threshold=args.threshold
    )
    print(f"Threshold used: {args.threshold:g}")
    if args.out:
        result.to_csv(args.out, index=False)
        if result.empty:
            print(f"No labeled frames for this video; empty report written to {args.out}.")
        else:
            print(f"Scored {len(result)} labeled frame(s); report written to {args.out}.")
    elif result.empty:
        print("No labeled frames for this video.")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
