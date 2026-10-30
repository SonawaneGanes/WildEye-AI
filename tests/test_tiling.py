import numpy as np

from utils.generate_video_data import non_max_suppression, run_tiled_detection


def test_tiled_detection_shifts_tile_boxes_into_full_image_coordinates():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    boxes, scores, classes = run_tiled_detection(
        image,
        lambda tile: (
            np.array([[0.2, 0.5, 0.4, 0.75]]),
            np.array([0.8]),
            np.array([3]),
        ),
        grid=(1, 2),
        overlap=0.2,
    )

    assert boxes.shape == (2, 4)
    np.testing.assert_allclose(
        boxes,
        [[0.2, 0.28, 0.4, 0.42], [0.2, 0.72, 0.4, 0.86]],
    )
    np.testing.assert_allclose(scores, [0.8, 0.8])
    np.testing.assert_array_equal(classes, [3, 3])


def test_nms_suppresses_overlapping_boxes_of_same_class_only():
    boxes = np.array(
        [
            [0.1, 0.1, 0.5, 0.5],
            [0.12, 0.12, 0.48, 0.48],
            [0.12, 0.12, 0.48, 0.48],
        ]
    )
    scores = np.array([0.9, 0.8, 0.7])
    classes = np.array([1, 1, 2])

    assert non_max_suppression(boxes, scores, classes).tolist() == [0, 2]


def test_no_tile_detections_return_empty_arrays():
    boxes, scores, classes = run_tiled_detection(
        np.zeros((20, 20, 3)), lambda tile: ([], [], [])
    )

    assert boxes.shape == (0, 4)
    assert scores.size == 0
    assert classes.size == 0
