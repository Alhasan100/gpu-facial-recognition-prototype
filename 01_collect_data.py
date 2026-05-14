"""
Webcam-based face image collection utility.

Version: 1.0.0
Author: Project Maintainers
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


WINDOW_NAME = "Face Data Collection"
DEFAULT_OUTPUT_DIR = Path("dataset")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the collection utility.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed command-line options.
    """
    parser = argparse.ArgumentParser(description="Collect webcam face images for training.")
    parser.add_argument("--name", required=True, help="Person name used as the dataset folder name.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Dataset output directory.")
    return parser.parse_args()


def create_person_directory(output_dir: Path, person_name: str) -> Path:
    """
    Create and return the image directory for a person.

    Args:
        output_dir (Path): Base dataset directory.
        person_name (str): Person name used for the folder.

    Returns:
        Path: Directory where captured images will be saved.
    """
    safe_name = person_name.strip().replace(" ", "_")
    person_dir = output_dir / safe_name
    person_dir.mkdir(parents=True, exist_ok=True)
    return person_dir


def build_image_path(person_dir: Path, person_name: str, count: int) -> Path:
    """
    Build a unique image path for a captured frame.

    Args:
        person_dir (Path): Directory for the current person.
        person_name (str): Original person name.
        count (int): Current capture counter.

    Returns:
        Path: Full path for the next image file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = person_name.strip().replace(" ", "_")
    return person_dir / f"{safe_name}_{timestamp}_{count:04d}.jpg"


def draw_status(frame, person_name: str, capture_count: int) -> None:
    """
    Draw collection status text on the preview frame.

    Args:
        frame: Current OpenCV frame.
        person_name (str): Name of the person being collected.
        capture_count (int): Number of saved images in this session.

    Returns:
        None.
    """
    status = f"{person_name} | Saved: {capture_count} | C: capture | Q/ESC: quit"
    cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 220, 30), 2)


def should_quit() -> bool:
    """
    Check whether the user requested the collection loop to stop.

    Args:
        None.

    Returns:
        bool: True when the process should quit.
    """
    key = cv2.waitKeyEx(1)
    if key in (ord("q"), ord("Q"), 27):
        return True

    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def collect_images(camera_index: int, person_name: str, person_dir: Path) -> int:
    """
    Open the webcam and save frames when the user presses C.

    Args:
        camera_index (int): OpenCV camera index.
        person_name (str): Person name used in file names.
        person_dir (Path): Output directory for captured images.

    Returns:
        int: Number of images captured during the session.
    """
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    capture_count = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read from the webcam.")

            preview = frame.copy()
            draw_status(preview, person_name, capture_count)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKeyEx(1)
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord("c"):
                capture_count += 1
                image_path = build_image_path(person_dir, person_name, capture_count)
                cv2.imwrite(str(image_path), frame)
                print(f"Saved {image_path}")
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return capture_count


def main() -> None:
    """
    Run the image collection workflow.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    person_dir = create_person_directory(args.output_dir, args.name)
    total = collect_images(args.camera_index, args.name, person_dir)
    print(f"Collection complete. Saved {total} image(s) to {person_dir}.")


if __name__ == "__main__":
    main()
