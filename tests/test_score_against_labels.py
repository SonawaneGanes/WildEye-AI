import pandas as pd

from evaluation.score_against_labels import score_against_labels


def test_reports_misses_wrong_classes_and_false_positives():
    labels = pd.DataFrame(
        [{"video": "clip", "frame": 4, "true_classes": "horse; cow", "notes": ""}]
    )
    detections = pd.DataFrame(
        {
            "frame": [4, 4, 4],
            "score": [0.8, 0.9, 0.2],
            "class_str": ["dog", "horse", "cat"],
        }
    )

    result = score_against_labels(labels, detections).iloc[0]

    assert result["missed_true_classes"] == "cow"
    assert result["wrong_class_predictions"] == "dog"
    assert result["false_positives"] == "dog@0.8"
    assert result["missed_count"] == 1
    assert result["wrong_class_count"] == 1
    assert result["false_positive_count"] == 1


def test_does_not_report_wrong_class_when_all_true_classes_are_detected():
    labels = pd.DataFrame(
        [{"video": "clip", "frame": 1, "true_classes": "horse", "notes": ""}]
    )
    detections = pd.DataFrame(
        {"frame": [1, 1], "score": [0.8, 0.9], "class_str": ["horse", "dog"]}
    )

    result = score_against_labels(labels, detections).iloc[0]

    assert result["wrong_class_predictions"] == ""
    assert result["false_positives"] == "dog@0.9"


def test_empty_labels_return_empty_report():
    labels = pd.DataFrame(columns=["video", "frame", "true_classes", "notes"])
    detections = pd.DataFrame(columns=["frame", "score", "class_str"])

    assert score_against_labels(labels, detections).empty
