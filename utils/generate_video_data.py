# coding: utf-8
import numpy as np

# Generate the base64 string of each frame, not recommended
ENCODE_B64 = False
# Prints information about training in console
VERBOSE = True
# Show video being processed in window
SHOW_PROCESS = True
# Create a video with the bounding boxes
WRITE_VIDEO_OUT = True
# Minimum score threshold for a bounding box to be recorded in data
THRESHOLD = 0.2
OUTPUT_FPS = 24.0
# Change name of video being processed
VIDEO_FILE_NAME = "../videos/DroneCarFestival3"
VIDEO_EXTENSION = ".mp4"
FROZEN_GRAPH_PATH = "frozen_inference_graph.pb"
BASE64_CSV_PATH = "video_frames_b64.csv"
OUTPUT_VIDEO_SIZE = (1280, 720)
VIDEO_CODEC = "MP4V"
BOUNDING_BOX_VIDEO_SUFFIX = "WithBoundingBoxes.mp4"
ORIGINAL_VIDEO_SUFFIX = "Original.mp4"
DETECTION_CSV_SUFFIX = "DetectionData.csv"
BASE64_IMAGE_EXTENSION = ".png"
PROCESS_WINDOW_NAME = "Object detection"


def _tile_starts(length, tile_count, overlap):
    if tile_count == 1:
        return [(0, length)]
    tile_length = min(
        length,
        int(np.ceil(length / (tile_count - (tile_count - 1) * overlap))),
    )
    starts = np.linspace(0, length - tile_length, tile_count).round().astype(int)
    return [(int(start), int(start + tile_length)) for start in starts]


def non_max_suppression(boxes, scores, classes, iou_threshold=0.5):
    """Return score-ordered indices after class-aware NMS for normalized yxyx boxes."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    classes = np.asarray(classes).reshape(-1)
    if not (len(boxes) == len(scores) == len(classes)):
        raise ValueError("boxes, scores, and classes must have matching lengths")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")

    order = np.argsort(-scores, kind="stable")
    kept = []
    while order.size:
        current = order[0]
        kept.append(current)
        remaining = order[1:]
        if not remaining.size:
            break

        top = boxes[current]
        candidates = boxes[remaining]
        intersection_ymin = np.maximum(top[0], candidates[:, 0])
        intersection_xmin = np.maximum(top[1], candidates[:, 1])
        intersection_ymax = np.minimum(top[2], candidates[:, 2])
        intersection_xmax = np.minimum(top[3], candidates[:, 3])
        intersection = np.maximum(0, intersection_ymax - intersection_ymin) * np.maximum(
            0, intersection_xmax - intersection_xmin
        )
        top_area = max(0, top[2] - top[0]) * max(0, top[3] - top[1])
        candidate_areas = np.maximum(0, candidates[:, 2] - candidates[:, 0]) * np.maximum(
            0, candidates[:, 3] - candidates[:, 1]
        )
        union = top_area + candidate_areas - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        suppress = (classes[remaining] == classes[current]) & (iou > iou_threshold)
        order = remaining[~suppress]
    return np.asarray(kept, dtype=int)


def run_tiled_detection(image, detect_fn, grid=(2, 2), overlap=0.2):
    """Run a detector on overlapping tiles and return merged normalized yxyx boxes.

    ``detect_fn`` receives an image tile and returns ``(boxes, scores, classes)``;
    boxes use the detector's normalized ``[ymin, xmin, ymax, xmax]`` format.
    """
    if len(grid) != 2 or any(not isinstance(value, int) or value < 1 for value in grid):
        raise ValueError("grid must contain two positive integers")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in the range [0, 1)")
    image = np.asarray(image)
    if image.ndim < 2 or not image.shape[0] or not image.shape[1]:
        raise ValueError("image must have non-empty height and width")

    height, width = image.shape[:2]
    boxes_by_tile = []
    scores_by_tile = []
    classes_by_tile = []
    for top, bottom in _tile_starts(height, grid[0], overlap):
        for left, right in _tile_starts(width, grid[1], overlap):
            tile = image[top:bottom, left:right]
            boxes, scores, classes = detect_fn(tile)
            boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
            scores = np.asarray(scores, dtype=float).reshape(-1)
            classes = np.asarray(classes).reshape(-1)
            if not (len(boxes) == len(scores) == len(classes)):
                raise ValueError("detect_fn must return matching boxes, scores, and classes")
            if not len(boxes):
                continue

            tile_height, tile_width = tile.shape[:2]
            absolute_boxes = boxes.copy()
            absolute_boxes[:, [0, 2]] = absolute_boxes[:, [0, 2]] * tile_height + top
            absolute_boxes[:, [1, 3]] = absolute_boxes[:, [1, 3]] * tile_width + left
            absolute_boxes[:, [0, 2]] = np.clip(absolute_boxes[:, [0, 2]], 0, height) / height
            absolute_boxes[:, [1, 3]] = np.clip(absolute_boxes[:, [1, 3]], 0, width) / width
            boxes_by_tile.append(absolute_boxes)
            scores_by_tile.append(scores)
            classes_by_tile.append(classes)

    if not boxes_by_tile:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    boxes = np.concatenate(boxes_by_tile)
    scores = np.concatenate(scores_by_tile)
    classes = np.concatenate(classes_by_tile)
    kept = non_max_suppression(boxes, scores, classes)
    return boxes[kept], scores[kept], classes[kept]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tiles",
        action="store_true",
        help="Experimentally run detection on overlapping image tiles."
    )
    parser.add_argument(
        "--log-raw-classes",
        action="store_true",
        help="Print raw model class IDs and names for each frame."
    )
    args = parser.parse_args()

    import base64
    import time

    import cv2 as cv
    import pandas as pd
    import tensorflow as tf

    from utils.mscoco_label_map import category_index
    from utils.visualization_utils import visualize_boxes_and_labels_on_image_array

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FROZEN_GRAPH_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Loading the videocapture objects
    cap = cv.VideoCapture(f'{VIDEO_FILE_NAME}{VIDEO_EXTENSION}')

    if WRITE_VIDEO_OUT:
        # Setup the video creation process
        fourcc = cv.VideoWriter_fourcc(*VIDEO_CODEC)
        out = cv.VideoWriter(
            f'{VIDEO_FILE_NAME}{BOUNDING_BOX_VIDEO_SUFFIX}',
            fourcc,
            OUTPUT_FPS,
            OUTPUT_VIDEO_SIZE
        )
        out_orig = cv.VideoWriter(
            f'{VIDEO_FILE_NAME}{ORIGINAL_VIDEO_SUFFIX}',
            fourcc,
            OUTPUT_FPS,
            OUTPUT_VIDEO_SIZE
        )

    # Start the session
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image that a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown in the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            frame_base64_ls = []  # The list containing all the frame in base64 and their timestamp
            frame_info_ls = []  # The list containing the information about the frames

            counter = 0
            while cap.isOpened():
                ret, image = cap.read()

                if ret:
                    # Retrieve timestamp
                    curr_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))

                    # Convert image into an np array
                    image_np = np.array(image)
                    image_np_expanded = np.expand_dims(image_np, axis=0)

                    t1 = time.time()

                    if args.tiles:
                        def detect_tile(tile):
                            tile_output = sess.run(
                                [
                                    detection_boxes,
                                    detection_scores,
                                    detection_classes,
                                    num_detections,
                                ],
                                feed_dict={image_tensor: np.expand_dims(tile, axis=0)},
                            )
                            tile_count = int(np.asarray(tile_output[3]).reshape(-1)[0])
                            return (
                                np.asarray(tile_output[0]).reshape(-1, 4)[:tile_count],
                                np.asarray(tile_output[1]).reshape(-1)[:tile_count],
                                np.asarray(tile_output[2]).reshape(-1).astype(np.int32)[
                                    :tile_count
                                ],
                            )

                        boxes, scores, classes = run_tiled_detection(image_np, detect_tile)
                        num = len(scores)
                    else:
                        # Run the algorithm, retrieve the boxes, score and classes
                        (boxes, scores, classes, num) = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})

                    t2 = time.time()

                    # Remove the leading 1 dimension
                    if not args.tiles:
                        boxes = np.squeeze(boxes)
                        classes = np.squeeze(classes).astype(np.int32)
                        scores = np.squeeze(scores)

                    if args.log_raw_classes:
                        raw_classes = [
                            f"{int(class_id)}:{category_index.get(int(class_id), {}).get('name', 'unmapped')}"
                            for class_id in classes[:int(num)]
                        ]
                        print(f"Raw classes at frame {curr_frame}: {', '.join(raw_classes)}")

                    # Draw the bounding boxes with information about the predictions
                    visualize_boxes_and_labels_on_image_array(
                        image_np,
                        boxes,
                        classes,
                        scores,
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=2
                    )

                    # Encode the image into base64
                    if ENCODE_B64:
                        retval, buffer = cv.imencode(BASE64_IMAGE_EXTENSION, image_np)
                        img_str = base64.b64encode(buffer)
                        image_b64 = 'data:image/png;base64,{}'.format(img_str.decode('ascii'))

                        # Append the image along with timestamp
                        frame_base64_ls.append([curr_frame, image_b64])

                    # Update the output video
                    if WRITE_VIDEO_OUT:
                        out.write(image_np)
                        out_orig.write(image)  # Writes the original image

                    # Process the information about the video at that exact timestamp
                    timestamp_df = pd.DataFrame([curr_frame for _ in range(int(num))], columns=["frame"])
                    boxes_df = pd.DataFrame(boxes, columns=['y', 'x', 'bottom', 'right'])
                    classes_df = pd.DataFrame(classes, columns=['class'])
                    score_df = pd.DataFrame(scores, columns=['score'])
                    # Maps a np array of integer to their coco index
                    coco_map = np.vectorize(lambda i: category_index[i]['name'])
                    classes_str_df = pd.DataFrame(coco_map(classes), columns=['class_str'])

                    # Concatenate all the information
                    info_df = pd.concat([timestamp_df, boxes_df, classes_df, classes_str_df, score_df], axis=1)

                    # Only keep the entries with a score over the threshold
                    narrow_info_df = info_df[info_df['score'] > THRESHOLD]

                    # Append it the list of information of all the frames
                    frame_info_ls.append(narrow_info_df)

                    t3 = time.time()

                    counter += 1
                    if VERBOSE:
                        print(f"Algorithm runtime at frame {counter}: {t2-t1:.2f}")

                    if SHOW_PROCESS:
                        cv.imshow(PROCESS_WINDOW_NAME, image_np)

                        if cv.waitKey(1) & 0xFF == ord('q'):
                            break

                else:
                    break

            if ENCODE_B64:
                # Save the frames in base64
                frame_base64_df = pd.DataFrame(frame_base64_ls, columns=['frame', 'source'])
                frame_base64_df.to_csv(BASE64_CSV_PATH, index=False)

            frame_info_df = pd.concat(frame_info_ls)
            frame_info_df.to_csv(f"{VIDEO_FILE_NAME}{DETECTION_CSV_SUFFIX}", index=False)

    # Release processes
    cap.release()

    if WRITE_VIDEO_OUT:
        out.release()
        out_orig.release()

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
