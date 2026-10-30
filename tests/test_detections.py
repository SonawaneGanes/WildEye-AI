import pandas as pd

from utils.detections import (
    count_classes,
    filter_frame,
    keep_detection,
    top_n_detections,
)


def test_filter_frame_returns_empty_frame_when_no_rows_match():
    df = pd.DataFrame(
        {"frame": [1], "score": [0.9], "class_str": ["dog"]}
    )

    assert filter_frame(df, frame=2, threshold=0.2).empty


def test_filter_frame_excludes_score_equal_to_threshold():
    df = pd.DataFrame(
        {"frame": [1, 1], "score": [0.2, 0.3], "class_str": ["dog", "cat"]}
    )

    result = filter_frame(df, frame=1, threshold=0.2)

    assert result["class_str"].tolist() == ["cat"]


def test_empty_class_thresholds_match_default_filtering():
    df = pd.DataFrame(
        {"frame": [1, 1, 1], "score": [0.2, 0.3, 0.9], "class_str": ["dog", "cat", "dog"]}
    )

    assert filter_frame(df, frame=1, threshold=0.2).equals(
        filter_frame(df, frame=1, threshold=0.2, class_thresholds={})
    )


def test_class_threshold_override_changes_only_that_class():
    df = pd.DataFrame(
        {"frame": [1, 1], "score": [0.3, 0.3], "class_str": ["dog", "cat"]}
    )

    result = filter_frame(
        df, frame=1, threshold=0.2, class_thresholds={"dog": 0.4}
    )

    assert result["class_str"].tolist() == ["cat"]
    assert keep_detection("dog", 0.3, {"dog": 0.4}, 0.2) is False
    assert keep_detection("cat", 0.3, {"dog": 0.4}, 0.2) is True


def test_top_n_detections_caps_results_at_eight():
    df = pd.DataFrame({"score": range(10)})

    assert len(top_n_detections(df)) == 8


def test_top_n_detections_sorts_by_descending_score():
    df = pd.DataFrame({"score": [0.2, 0.9, 0.5]})

    assert top_n_detections(df)["score"].tolist() == [0.9, 0.5, 0.2]


def test_count_classes_returns_class_counts():
    df = pd.DataFrame({"class_str": ["dog", "cat", "dog"]})

    assert count_classes(df).to_dict() == {"dog": 2, "cat": 1}
