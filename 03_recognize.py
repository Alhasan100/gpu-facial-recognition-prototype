"""
Real-time GPU-accelerated facial recognition.

Version: 1.0.0
Author: Project Maintainers
"""

from __future__ import annotations

import argparse
import os
import pickle
import site
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np


WINDOW_NAME = "Face Recognition"
DEFAULT_ENCODINGS_PATH = Path("encodings") / "face_encodings.pickle"
STATUS_BAR_HEIGHT = 72


def configure_windows_cuda_dll_paths() -> None:
    """
    Register NVIDIA DLL paths used by CUDA-enabled dlib on Windows.

    Args:
        None.

    Returns:
        None.
    """
    if os.name != "nt":
        return

    site_package_dirs = {Path(sys.prefix) / "Lib" / "site-packages"}
    site_package_dirs.update(Path(path) for path in site.getsitepackages())

    candidate_dirs = []
    for package_dir in site_package_dirs:
        candidate_dirs.extend(
            [
                package_dir / "nvidia" / "cudnn" / "bin",
                package_dir / "nvidia" / "cu13" / "bin" / "x86_64",
            ]
        )

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidate_dirs.append(Path(cuda_path) / "bin")

    candidate_dirs.extend(Path(path) for path in os.environ.get("PATH", "").split(os.pathsep) if path)

    existing_dirs = [path for path in candidate_dirs if path.exists()]
    if existing_dirs:
        os.environ["PATH"] = ";".join(str(path) for path in existing_dirs) + ";" + os.environ.get("PATH", "")

    for path in existing_dirs:
        os.add_dll_directory(str(path))


configure_windows_cuda_dll_paths()
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)

import dlib  # noqa: E402
import face_recognition  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for real-time recognition.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed command-line options.
    """
    parser = argparse.ArgumentParser(description="Run real-time facial recognition with dlib CUDA.")
    parser.add_argument("--encodings", type=Path, default=DEFAULT_ENCODINGS_PATH, help="Pickle file with encodings.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--scale", type=float, default=0.5, help="Frame scale factor for detection.")
    parser.add_argument("--tolerance", type=float, default=0.52, help="Face comparison tolerance.")
    parser.add_argument("--upsample", type=int, default=1, help="Face detection upsample count.")
    parser.add_argument("--debug", action="store_true", help="Print per-frame detection and matching details.")
    return parser.parse_args()


def load_training_payload(encodings_path: Path) -> dict[str, Any]:
    """
    Load known face encodings from a pickle file.

    Args:
        encodings_path (Path): Pickle file created by the trainer.

    Returns:
        dict[str, Any]: Loaded training payload.
    """
    if not encodings_path.exists():
        raise FileNotFoundError(f"Encodings file does not exist: {encodings_path}")

    with encodings_path.open("rb") as file:
        payload = pickle.load(file)

    if not payload.get("encodings") or not payload.get("names"):
        raise RuntimeError("Encodings file does not contain usable training data.")

    return payload


def ensure_cuda_available() -> None:
    """
    Validate that dlib was compiled with CUDA support.

    Args:
        None.

    Returns:
        None.
    """
    if not dlib.DLIB_USE_CUDA:
        raise RuntimeError("dlib is not compiled with CUDA support.")

    device_count = dlib.cuda.get_num_devices()
    if device_count < 1:
        raise RuntimeError("dlib CUDA is enabled, but no CUDA devices were detected.")

    print(f"dlib CUDA enabled. CUDA devices detected: {device_count}")


def trigger_protected_action_placeholder(name: str, confidence: float) -> None:
    """
    Placeholder for application-specific protected actions.

    Args:
        name (str): Recognized identity.
        confidence (float): Confidence score derived from face distance.

    Returns:
        None.
    """
    print(f"Protected action placeholder triggered for {name} with confidence {confidence:.3f}.")


def choose_match(
    known_encodings: list[Any],
    known_names: list[str],
    face_encoding: Any,
    tolerance: float,
) -> tuple[str, float, float]:
    """
    Select the best matching known identity for one face encoding.

    Args:
        known_encodings (list[Any]): Stored known face encodings.
        known_names (list[str]): Stored names matching known encodings.
        face_encoding (Any): Encoding for the detected face.
        tolerance (float): Maximum accepted face distance.

    Returns:
        tuple[str, float, float]: Matched name, confidence score, and best face distance.
    """
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])

    if best_distance > tolerance:
        return "Unknown", 0.0, best_distance

    confidence = max(0.0, 1.0 - best_distance)
    return known_names[best_index], confidence, best_distance


def scale_box(box: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    """
    Scale a face bounding box back to the original frame size.

    Args:
        box (tuple[int, int, int, int]): Face box in top, right, bottom, left order.
        scale (float): Scale factor used before detection.

    Returns:
        tuple[int, int, int, int]: Scaled face box in top, right, bottom, left order.
    """
    top, right, bottom, left = box
    inverse_scale = 1.0 / scale
    return (
        int(top * inverse_scale),
        int(right * inverse_scale),
        int(bottom * inverse_scale),
        int(left * inverse_scale),
    )


def draw_face(frame, box: tuple[int, int, int, int], name: str, confidence: float, distance: float) -> None:
    """
    Draw a recognition result on a video frame.

    Args:
        frame: Current OpenCV frame.
        box (tuple[int, int, int, int]): Face box in top, right, bottom, left order.
        name (str): Recognition label.
        confidence (float): Recognition confidence.
        distance (float): Best face distance.

    Returns:
        None.
    """
    top, right, bottom, left = box
    color = (40, 210, 40) if name != "Unknown" else (40, 40, 220)
    label = f"Unknown d={distance:.2f}" if name == "Unknown" else f"{name} {confidence:.2f}"

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(frame, (left, bottom - 32), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, label, (left + 8, bottom - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_status_bar(
    frame,
    face_count: int,
    fps: float,
    scale: float,
    tolerance: float,
    upsample: int,
    status: str,
) -> None:
    """
    Draw runtime diagnostics on the webcam frame.

    Args:
        frame: Current OpenCV frame.
        face_count (int): Number of detected face boxes.
        fps (float): Estimated frames per second.
        scale (float): Frame scale factor used for detection.
        tolerance (float): Current matching tolerance.
        upsample (int): Current upsample value.
        status (str): Human-readable recognition state.

    Returns:
        None.
    """
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, STATUS_BAR_HEIGHT), (20, 20, 20), cv2.FILLED)

    line_one = f"{status} | faces: {face_count} | fps: {fps:.1f} | CUDA: {dlib.cuda.get_num_devices()} device"
    line_two = f"scale: {scale:.2f} | tolerance: {tolerance:.2f} | upsample: {upsample} | Q/ESC: quit"

    cv2.putText(frame, line_one, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2)
    cv2.putText(frame, line_two, (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 220, 255), 1)


def build_status(face_count: int, recognized_names: list[str]) -> str:
    """
    Build a compact status message for the current frame.

    Args:
        face_count (int): Number of detected face boxes.
        recognized_names (list[str]): Names recognized in the current frame.

    Returns:
        str: Status text for the overlay.
    """
    if face_count == 0:
        return "No face detected"

    known_names = [name for name in recognized_names if name != "Unknown"]
    if known_names:
        return "Recognized: " + ", ".join(sorted(set(known_names)))

    return "Face detected, no match"


def should_quit() -> bool:
    """
    Check whether the user requested the recognition loop to stop.

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


def recognize_faces(
    camera_index: int,
    known_encodings: list[Any],
    known_names: list[str],
    scale: float,
    tolerance: float,
    upsample: int,
    debug: bool,
) -> None:
    """
    Run real-time webcam recognition.

    Args:
        camera_index (int): OpenCV camera index.
        known_encodings (list[Any]): Stored known face encodings.
        known_names (list[str]): Stored names matching known encodings.
        scale (float): Frame scale factor used for detection.
        tolerance (float): Maximum accepted face distance.
        upsample (int): Face detection upsample count.
        debug (bool): Whether to print per-frame diagnostic details.

    Returns:
        None.
    """
    if not 0.0 < scale <= 1.0:
        raise ValueError("Scale must be greater than 0 and less than or equal to 1.")

    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    triggered_names = set()
    previous_time = time.perf_counter()
    fps = 0.0
    frame_index = 0

    try:
        while True:
            frame_index += 1
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read from the webcam.")

            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(
                rgb_small_frame,
                number_of_times_to_upsample=upsample,
                model="cnn",
            )
            encodings = face_recognition.face_encodings(rgb_small_frame, known_face_locations=boxes)
            recognized_names = []

            for box, face_encoding in zip(boxes, encodings):
                name, confidence, distance = choose_match(known_encodings, known_names, face_encoding, tolerance)
                recognized_names.append(name)
                draw_face(frame, scale_box(box, scale), name, confidence, distance)

                if debug:
                    print(
                        f"frame={frame_index} name={name} "
                        f"confidence={confidence:.3f} distance={distance:.3f} faces={len(boxes)}"
                    )

                if name != "Unknown" and name not in triggered_names:
                    trigger_protected_action_placeholder(name, confidence)
                    triggered_names.add(name)

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)

            status = build_status(len(boxes), recognized_names)
            draw_status_bar(frame, len(boxes), fps, scale, tolerance, upsample, status)
            cv2.imshow(WINDOW_NAME, frame)
            if should_quit():
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    """
    Run the real-time recognition workflow.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    ensure_cuda_available()
    payload = load_training_payload(args.encodings)
    recognize_faces(
        args.camera_index,
        payload["encodings"],
        payload["names"],
        args.scale,
        args.tolerance,
        args.upsample,
        args.debug,
    )


if __name__ == "__main__":
    main()
