# Evaluation

Evaluation results should document:

- Dataset size and the confirmed class list.
- The labeling method.
- The train/validation/test split method. Split by video or camera to avoid
  near-duplicate frames crossing splits.
- Per-class precision, recall, and F1.
- A confusion matrix that includes horse/cow and dog/cat.

## Before and after

| Field | Before | After |
|---|---|---|
| Change made |  |  |
| Threshold used |  |  |
| Labeled frame count |  |  |
| Missed count |  |  |
| Wrong-class count |  |  |
| False-positive count |  |  |

## Known open questions

- Frame offset: TODO — verify the relationship between video frames and CSV frame IDs.
- Label-map check: TODO — verify whether the label map matches the model's output IDs.
- Tracking: no `track_id` column or tracker was found in the current code or data CSVs;
  temporal voting requires tracking.
- Detection thresholds: TODO — tune optional per-class values using labeled frames.
