import pandas as pd

from evaluation.confusion_pairs import analyze_detections


def test_counts_class_cooccurrence_once_per_frame_and_reports_candidate():
    detections = pd.DataFrame(
        {
            "frame": [1, 1, 1, 2],
            "class_str": ["horse", "horse", "cow", "dog"],
            "score": [0.3, 0.7, 0.1, 0.9],
        }
    )

    result = analyze_detections(detections, "sample.csv")

    assert result["class_counts"].to_dict() == {"horse": 2, "cow": 1, "dog": 1}
    assert result["pair_frame_counts"] == {("cow", "horse"): 1}
    assert result["candidates"][["frame", "pair"]].values.tolist() == [
        [1, "horse/cow"]
    ]
    assert result["candidates"].iloc[0]["high_confidence_classes"] == "horse"


def test_pair_without_a_score_above_threshold_is_not_a_candidate():
    detections = pd.DataFrame(
        {"frame": [5, 5], "class_str": ["dog", "cat"], "score": [0.2, 0.2]}
    )

    result = analyze_detections(detections)

    assert result["pair_frame_counts"] == {("cat", "dog"): 1}
    assert result["candidates"].empty
