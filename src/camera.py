"""Preview cameras and find the index assigned to the external camera."""

import argparse

import cv2

from .camera_utils import default_camera_index, open_camera


def scan_cameras(max_index: int) -> None:
    available = []

    for index in range(max_index + 1):
        try:
            cap = open_camera(index)
        except RuntimeError:
            print(f"Camera index {index}: unavailable")
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"Camera index {index}: opened but no frame received")
            cap.release()
            continue

        available.append(index)
        height, width = frame.shape[:2]
        print(
            f"Camera index {index}: {width}x{height}. "
            "Check the preview to identify the external camera."
        )

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            cv2.putText(
                frame,
                f"Camera index {index} - press N for next, Q to quit",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Camera index scanner", frame)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("n"), ord("N")):
                break
            if key in (ord("q"), ord("Q"), 27):
                cap.release()
                cv2.destroyAllWindows()
                print(f"Available camera indexes: {available}")
                return

        cap.release()
        cv2.destroyAllWindows()

    print(f"Available camera indexes: {available}")
    print(
        "Set the external camera index for this PowerShell session with: "
        "$env:FACELOCKING_CAMERA_INDEX = \"N\""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        type=int,
        default=default_camera_index(),
        help="camera index to preview",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="preview camera indexes one at a time to identify the external camera",
    )
    parser.add_argument("--max-index", type=int, default=5)
    args = parser.parse_args()

    if args.scan:
        scan_cameras(args.max_index)
        return

    cap = open_camera(args.camera)
    print(f"Camera preview using index {args.camera}. Press q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the camera.")
                break
            cv2.imshow("Camera Test", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
