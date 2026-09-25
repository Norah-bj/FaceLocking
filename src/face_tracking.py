# src/face_tracking.py

import argparse
import time

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.align import align_face_5pt
from src.camera_utils import default_camera_index, open_camera
from src.face_signals import FaceSignalExtractor
from src.recognize import (
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    HaarFaceMesh5pt,
    load_db_npz,
)


class LockState(Enum):
    SEARCHING = auto()
    LOCKED = auto()
    LOST = auto()


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter / float(area_a + area_b - inter)


def center(box):
    x1, y1, x2, y2 = box

    return np.array(
        [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        dtype=np.float32,
    )


@dataclass
class TrackingSignal:
    error_x: float
    error_y: float
    horizontal: str
    vertical: str
    nose_x: int
    nose_y: int
    distance_px: float


class LockedFaceTracker:
    def __init__(
        self,
        target_name: Optional[str],
        detector,
        embedder,
        matcher,
        verify_every: int = 5,
        lost_timeout: float = 2.0,
        ema_alpha: float = 0.22,
        dead_zone: float = 0.08,
    ):
        self.target_name = target_name
        self.auto_select = target_name is None
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.verify_every = verify_every
        self.lost_timeout = float(lost_timeout)
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone
        self.enter_zone = 0.16
        # Stay LOCKED through detection gaps this long.
        self.coast_s = self.lost_timeout
        # Similarity to the enrolled face.
        # Acquire is stricter. A weak frame does not unlock.
        self.acquire_sim = 0.58
        self.keep_sim = 0.50
        self.reject_sim = 0.38
        self.verify_fail_limit = 8

        self.state = LockState.SEARCHING
        self.last_box = None
        self.last_face = None
        self.smooth_box = None
        self.smooth_center = None
        self.smooth_nose = None
        self.horizontal = "CENTER"
        self.vertical = "CENTER"

        self.miss_since = None
        self.verify_fails = 0
        self.frame_index = 0
        self.detected_faces = []
        self.detected_labels = []
        self._cached_detection_labels = []

    @staticmethod
    def box(face):
        return (face.x1, face.y1, face.x2, face.y2)

    def face_embedding(self, frame, face) -> np.ndarray:
        aligned, _ = align_face_5pt(
            frame,
            face.kps,
            out_size=(112, 112),
        )
        return self.embedder.embed(aligned).reshape(-1)

    def similarity_to_target(self, frame, face) -> float:
        template = self.matcher.db.get(self.target_name)

        if template is None:
            return 0.0

        emb = self.face_embedding(frame, face)
        ref = np.asarray(template, dtype=np.float32).reshape(-1)

        return float(np.dot(emb, ref))

    def acquire(self, frame, faces):
        best = None
        best_similarity = self.acquire_sim
        best_name = self.target_name

        for face in faces:
            # Approximate boxes are for holding a lock, not starting one.
            if face.score < 1.0:
                continue

            if self.auto_select and self.target_name is None:
                match = self.matcher.match(
                    self.face_embedding(frame, face)
                )
                if not match.accepted:
                    continue
                name = match.name
                similarity = match.similarity
            else:
                name = self.target_name
                similarity = self.similarity_to_target(frame, face)

            if name is not None and similarity > best_similarity:
                best = face
                best_similarity = similarity
                best_name = name

        if best is not None and self.auto_select:
            self.target_name = best_name

        return best

    def label_detections(self, frame, faces):
        """Classify detected faces, rechecking embeddings every few frames."""

        if not faces:
            self.detected_labels = []
            self._cached_detection_labels = []
            return

        refresh = (
            not self._cached_detection_labels
            or self.frame_index % self.verify_every == 0
        )

        if refresh:
            labeled = []
            for face in faces:
                name = "UNKNOWN"
                if face.score >= 1.0:
                    result = self.matcher.match(
                        self.face_embedding(frame, face)
                    )
                    if result.accepted and result.name is not None:
                        name = result.name
                labeled.append((self.box(face), name))
            self._cached_detection_labels = labeled
            self.detected_labels = [name for _box, name in labeled]
            return

        previous = list(self._cached_detection_labels)
        labeled = []
        for face in faces:
            current_box = self.box(face)
            name = "UNKNOWN"
            if previous:
                best_index = max(
                    range(len(previous)),
                    key=lambda i: iou(current_box, previous[i][0]),
                )
                overlap = iou(current_box, previous[best_index][0])
                if overlap >= 0.10:
                    _old_box, name = previous.pop(best_index)
            labeled.append((current_box, name))

        self._cached_detection_labels = labeled
        self.detected_labels = [name for _box, name in labeled]

    def associate(self, faces):
        if self.smooth_box is None or not faces:
            return None

        ref = tuple(float(v) for v in self.smooth_box.tolist())
        last_center = center(ref)

        last_diag = max(
            np.linalg.norm(
                np.array(
                    [ref[2] - ref[0], ref[3] - ref[1]],
                    dtype=np.float32,
                )
            ),
            1.0,
        )

        ranked = []

        for face in faces:
            box = self.box(face)
            overlap = iou(ref, box)
            displacement = (
                np.linalg.norm(center(box) - last_center)
                / last_diag
            )

            # Real landmarks beat a Haar-only box at the same place.
            score = overlap - 0.25 * displacement + 0.15 * face.score
            ranked.append((score, overlap, displacement, face))

        _score, overlap, displacement, candidate = max(
            ranked,
            key=lambda item: item[0],
        )

        if overlap > 0.05 or displacement < 0.75:
            return candidate

        return None

    def _remember(self, face):
        raw = np.array(self.box(face), dtype=np.float32)

        if self.smooth_box is None:
            self.smooth_box = raw
        else:
            width = max(1.0, float(raw[2] - raw[0]))
            jump = float(
                np.linalg.norm(
                    center(raw) - center(self.smooth_box)
                )
            )

            if jump > 0.9 * width:
                self.smooth_box = raw
            else:
                a = self.ema_alpha
                self.smooth_box = (
                    a * raw + (1.0 - a) * self.smooth_box
                )

        self.last_face = face
        self.last_box = self.box(face)
        self.smooth_center = center(self.smooth_box)
        nose = np.asarray(face.kps[2], dtype=np.float32)
        if self.smooth_nose is None:
            self.smooth_nose = nose
        else:
            a = self.ema_alpha
            self.smooth_nose = a * nose + (1.0 - a) * self.smooth_nose

    def display_box(self):
        if self.smooth_box is None:
            return self.last_box

        x1, y1, x2, y2 = self.smooth_box.tolist()

        return (
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
        )

    def _drop_lock(self):
        self.state = LockState.SEARCHING
        if self.auto_select:
            self.target_name = None
        self.last_box = None
        self.last_face = None
        self.smooth_box = None
        self.smooth_center = None
        self.smooth_nose = None
        self.horizontal = "CENTER"
        self.vertical = "CENTER"
        self.verify_fails = 0
        self.miss_since = None

    def update(self, frame):
        self.frame_index += 1

        faces = self.detector.detect(
            frame,
            max_faces=3,
            coarse=True,
        )
        self.detected_faces = faces
        self.label_detections(frame, faces)

        if self.state == LockState.SEARCHING:
            candidate = self.acquire(frame, faces)
        else:
            candidate = self.associate(faces)

        # A bad embedding no longer clears the lock.
        # Only a long run of "this is someone else" does.
        if (
            candidate is not None
            and candidate.score >= 1.0
            and self.state in (LockState.LOCKED, LockState.LOST)
            and self.frame_index % self.verify_every == 0
        ):
            similarity = self.similarity_to_target(
                frame,
                candidate,
            )

            if similarity >= self.keep_sim:
                self.verify_fails = 0
            else:
                self.verify_fails += 1

            if self.verify_fails >= self.verify_fail_limit:
                self._drop_lock()
                return None, None

        if candidate is None:
            now = time.monotonic()

            if self.last_face is not None:
                if self.miss_since is None:
                    self.miss_since = now

                if (now - self.miss_since) < self.coast_s:
                    self.state = LockState.LOST
                    return (
                        self.last_face,
                        self.position_signal(frame.shape),
                    )

            self._drop_lock()
            return None, None

        self.miss_since = None
        self.state = LockState.LOCKED
        self._remember(candidate)

        return candidate, self.position_signal(frame.shape)

    def _axis(self, value, state, negative, positive):
        if state == negative:
            if value < -self.dead_zone:
                return negative
            return "CENTER"

        if state == positive:
            if value > self.dead_zone:
                return positive
            return "CENTER"

        if value < -self.enter_zone:
            return negative

        if value > self.enter_zone:
            return positive

        return "CENTER"

    def position_signal(self, shape) -> TrackingSignal:
        height, width = shape[:2]
        nose_x, nose_y = self.smooth_nose.tolist()

        ex = float(
            (nose_x - width / 2.0)
            / (width / 2.0)
        )

        ey = float(
            (nose_y - height / 2.0)
            / (height / 2.0)
        )

        self.horizontal = self._axis(
            ex,
            self.horizontal,
            "LEFT",
            "RIGHT",
        )
        self.vertical = self._axis(
            ey,
            self.vertical,
            "UP",
            "DOWN",
        )

        return TrackingSignal(
            ex,
            ey,
            self.horizontal,
            self.vertical,
            int(round(nose_x)),
            int(round(nose_y)),
            float(np.hypot(nose_x - width / 2.0, nose_y - height / 2.0)),
        )


def draw_label(
    frame,
    text,
    xy,
    color,
    scale=0.62,
):
    cv2.putText(
        frame,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        default=None,
        help="optional enrolled identity; omit to lock any recognized person",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=default_camera_index(),
        help="camera device index (defaults to FACELOCKING_CAMERA_INDEX)",
    )

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--lost-timeout",
        type=float,
        default=2.0,
        help="seconds to keep the target lock while the face is missing",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.34,
    )

    args = parser.parse_args()

    detector = HaarFaceMesh5pt(
        min_size=(48, 48),
        debug=False,
    )

    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )

    matcher = FaceDBMatcher(
        load_db_npz(Path("data/db/face_db.npz")),
        dist_thresh=args.threshold,
    )
    if not matcher.db:
        raise RuntimeError(
            "The face database is empty. Run `python -m src.enroll` first."
        )

    tracker = LockedFaceTracker(
        args.target,
        detector,
        embedder,
        matcher,
        lost_timeout=args.lost_timeout,
    )

    signals = FaceSignalExtractor()

    cap = open_camera(
        args.camera,
        width=args.width,
        height=args.height,
    )

    blink_total = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            locked_face, position = tracker.update(frame)

            view = frame.copy()
            locked_name = tracker.target_name or "ANY KNOWN PERSON"

            state_text = (
                f"FACE MISSING - REACQUIRING: {locked_name}"
                if tracker.state == LockState.LOST
                else f"{tracker.state.name}: {locked_name}"
            )

            state_color = (0, 0, 255) if tracker.state == LockState.LOST else (
                (0, 180, 0) if locked_face is not None else (0, 140, 255)
            )

            draw_label(
                view,
                state_text,
                (12, 28),
                state_color,
                0.72,
            )

            if (
                locked_face is not None
                and position is not None
                and tracker.state == LockState.LOCKED
            ):
                box = tracker.display_box()
                x1, y1, x2, y2 = box

                cv2.rectangle(
                    view,
                    (x1, y1),
                    (x2, y2),
                    (255, 170, 0),
                    3,
                )

                face_state = signals.analyze(
                    frame,
                    box,
                )

                if face_state is not None:
                    if face_state.blink:
                        blink_total += 1

                    if not signals.expression_ready:
                        expression = "CALIBRATING - HOLD NEUTRAL"
                    else:
                        active_expressions = []
                        if face_state.smiling:
                            active_expressions.append("SMILE")
                        if face_state.frowning:
                            active_expressions.append("FROWN")
                        if face_state.sad:
                            active_expressions.append("SAD CUES")
                        if face_state.grimacing:
                            active_expressions.append("GRIMACE")
                        expression = (
                            " + ".join(active_expressions)
                            if active_expressions
                            else "NEUTRAL"
                        )

                    eye_text = (
                        "EYES CLOSED"
                        if face_state.eyes_closed
                        else "EYES OPEN"
                    )

                    draw_label(
                        view,
                        expression,
                        (x1, max(55, y1 - 50)),
                        (0, 170, 255)
                        if not signals.expression_ready
                        else (0, 255, 255),
                    )

                    draw_label(
                        view,
                        f"{eye_text} blinks={blink_total}",
                        (x1, max(78, y1 - 25)),
                        (255, 255, 0),
                    )

                    draw_label(
                        view,
                        f"EAR={face_state.ear:.3f} "
                        f"smile={face_state.smile_score:.3f}",
                        (12, view.shape[0] - 18),
                        (255, 255, 255),
                        0.52,
                    )

                    cv2.circle(
                        view,
                        (position.nose_x, position.nose_y),
                        5,
                        (0, 255, 255),
                        -1,
                    )

                    draw_label(
                        view,
                        f"H={position.horizontal} "
                        f"V={position.vertical} "
                        f"nose offset=({position.error_x:+.2f},"
                        f"{position.error_y:+.2f})",
                        (12, 56),
                        (255, 170, 0),
                        0.60,
                    )
                    draw_label(
                        view,
                        f"nose=({position.nose_x},{position.nose_y}) "
                        f"from center={position.distance_px:.0f}px",
                        (12, 82),
                        (255, 170, 0),
                        0.55,
                    )
            else:
                if tracker.state == LockState.SEARCHING:
                    signals.reset()

                h, w = view.shape[:2]
                dz = tracker.dead_zone

                cv2.rectangle(
                    view,
                    (
                        int(w * (0.5 - dz / 2)),
                        int(h * (0.5 - dz / 2)),
                    ),
                    (
                        int(w * (0.5 + dz / 2)),
                        int(h * (0.5 + dz / 2)),
                    ),
                    (120, 120, 120),
                    1,
                )

            target_box = (
                tracker.display_box()
                if tracker.last_face is not None
                else None
            )
            for index, other in enumerate(tracker.detected_faces):
                other_box = tracker.box(other)
                if target_box is not None and iou(target_box, other_box) > 0.25:
                    continue
                identity = (
                    tracker.detected_labels[index]
                    if index < len(tracker.detected_labels)
                    else "UNKNOWN"
                )
                cv2.rectangle(
                    view,
                    (other.x1, other.y1),
                    (other.x2, other.y2),
                    (150, 150, 150),
                    1,
                )
                draw_label(
                    view,
                    identity,
                    (other.x1, max(110, other.y1 - 8)),
                    (0, 0, 255) if identity == "UNKNOWN" else (180, 180, 180),
                    0.48,
                )

            cv2.imshow(
                "Locked Face Tracking",
                view,
            )

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cap.release()
        signals.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
