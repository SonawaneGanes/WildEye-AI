import pandas as pd


def keep_detection(
    class_name: str,
    score: float,
    class_thresholds: dict[str, float] | None,
    default_threshold: float,
) -> bool:
    """Return whether a detection strictly exceeds its class-specific threshold."""
    thresholds = class_thresholds or {}
    return score > thresholds.get(class_name, default_threshold)


def filter_frame(
    df: pd.DataFrame,
    frame: int,
    threshold: float,
    class_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return detections for one frame with scores strictly above threshold."""
    frame_rows = df.loc[df["frame"] == frame]
    if class_thresholds is None:
        return frame_rows.loc[frame_rows["score"] > threshold]

    keep = frame_rows.apply(
        lambda row: keep_detection(
            row["class_str"], row["score"], class_thresholds, threshold
        ),
        axis=1,
    )
    return frame_rows.loc[keep]


def top_n_detections(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Return up to n detections ordered by descending score."""
    return df.sort_values("score", ascending=False).head(n)


def count_classes(df: pd.DataFrame) -> pd.Series:
    """Return detection counts grouped by class name."""
    return df["class_str"].value_counts()
