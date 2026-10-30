import pytest

from utils.timing import time_to_frame


def test_time_zero_returns_zero():
    assert time_to_frame(0.0, 24.0) == 0


def test_five_seconds_at_24_fps_returns_120():
    assert time_to_frame(5.0, 24.0) == 120


def test_frame_offset_is_added():
    assert time_to_frame(5.0, 24.0, frame_offset=1) == 121


def test_negative_time_raises_value_error():
    with pytest.raises(ValueError):
        time_to_frame(-0.1, 24.0)
