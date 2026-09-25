"""Shared camera selection helpers for the demo scripts."""

from __future__ import annotations

import os

import cv2


def default_camera_index() -> int:
    value = os.environ.get("FACELOCKING_CAMERA_INDEX", "1")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            "FACELOCKING_CAMERA_INDEX must be an integer camera index."
        ) from exc


def open_camera(
    index: int | None = None,
    width: int | None = None,
    height: int | None = None,
):
    """Open a camera by index, preferring DirectShow on Windows."""

    camera_index = default_camera_index() if index is None else int(index)

    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    except (AttributeError, TypeError):
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Run `python -m src.camera --scan` to find the external camera, "
            "then set FACELOCKING_CAMERA_INDEX to its index."
        )

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
