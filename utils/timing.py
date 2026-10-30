def time_to_frame(current_time: float, fps: float, frame_offset: int = 0) -> int:
    """Convert playback time in seconds to a frame number."""
    if current_time < 0:
        raise ValueError("current_time must be non-negative")
    return round(current_time * fps) + frame_offset


# TODO: Verify offset against a real video. generate_video_data.py appears to store frame numbers after reading, so IDs may start at 1 (see evaluation/README.md).
