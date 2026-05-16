"""
GPU-accelerated face encoding trainer.

Version: 1.0.0
Author: Project Maintainers
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import site
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path("dataset")
DEFAULT_OUTPUT_PATH = Path("encodings") / "face_encodings.pickle"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ImageTrainingResult:
    """
    Training result for a single input image.

    Args:
        image_path (Path): Processed image path.
        person_name (str): Identity label inferred from the parent folder.
        width (int): Source image width in pixels.
        height (int): Source image height in pixels.
        faces_detected (int): Number of detected face boxes.
        encodings_created (int): Number of encodings created from the image.
        elapsed_seconds (float): Processing time for the image.
        status (str): Processing status.

    Returns:
        None.
    """

    image_path: Path
    person_name: str
    width: int
    height: int
    faces_detected: int
    encodings_created: int
    elapsed_seconds: float
    status: str
    detail: str


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

    def path_exists(path: Path) -> bool:
        """
        Check whether a path exists without failing on restricted Windows folders.

        Args:
            path (Path): Path to inspect.

        Returns:
            bool: True when the path exists and is accessible.
        """
        try:
            return path.exists()
        except OSError:
            return False

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

    existing_dirs = [path for path in candidate_dirs if path_exists(path)]
    if existing_dirs:
        os.environ["PATH"] = ";".join(str(path) for path in existing_dirs) + ";" + os.environ.get("PATH", "")

    for path in existing_dirs:
        os.add_dll_directory(str(path))


configure_windows_cuda_dll_paths()
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)

import dlib  # noqa: E402
import face_recognition_models  # noqa: E402


def configure_face_recognition_model_paths() -> None:
    """
    Copy face_recognition model files to an ASCII-safe cache and point imports there.

    Args:
        None.

    Returns:
        None.
    """
    model_locations = {
        "pose_predictor_model_location": Path(face_recognition_models.pose_predictor_model_location()),
        "pose_predictor_five_point_model_location": Path(face_recognition_models.pose_predictor_five_point_model_location()),
        "cnn_face_detector_model_location": Path(face_recognition_models.cnn_face_detector_model_location()),
        "face_recognition_model_location": Path(face_recognition_models.face_recognition_model_location()),
    }

    cache_dir = Path(tempfile.gettempdir()) / "gpu_facial_recognition_models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    patched_locations = {}
    for function_name, source_path in model_locations.items():
        cached_path = cache_dir / source_path.name
        if not cached_path.exists():
            shutil.copy2(source_path, cached_path)
        patched_locations[function_name] = str(cached_path)

    face_recognition_models.pose_predictor_model_location = lambda: patched_locations["pose_predictor_model_location"]
    face_recognition_models.pose_predictor_five_point_model_location = (
        lambda: patched_locations["pose_predictor_five_point_model_location"]
    )
    face_recognition_models.cnn_face_detector_model_location = lambda: patched_locations["cnn_face_detector_model_location"]
    face_recognition_models.face_recognition_model_location = lambda: patched_locations["face_recognition_model_location"]


configure_face_recognition_model_paths()
import face_recognition  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed command-line options.
    """
    parser = argparse.ArgumentParser(description="Train face encodings with dlib's CUDA CNN model.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Input dataset directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output pickle file.")
    parser.add_argument("--upsample", type=int, default=1, help="Face detection upsample count.")
    parser.add_argument("--jitter", type=int, default=1, help="Encoding jitter count.")
    parser.add_argument(
        "--multi-face-policy",
        choices=("skip", "encode-all"),
        default="skip",
        help="How to handle images where more than one face is detected.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide per-image progress details.")
    return parser.parse_args()


def iter_image_paths(dataset_dir: Path) -> list[Path]:
    """
    Return all supported image files under the dataset directory.

    Args:
        dataset_dir (Path): Base dataset directory.

    Returns:
        list[Path]: Sorted image paths.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


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


def summarize_dataset(image_paths: list[Path]) -> None:
    """
    Print a compact overview of the input dataset.

    Args:
        image_paths (list[Path]): Image files selected for training.

    Returns:
        None.
    """
    counts = Counter(path.parent.name for path in image_paths)
    print("")
    print("Dataset summary")
    print(f"  Total images: {len(image_paths)}")
    for person_name, count in sorted(counts.items()):
        print(f"  {person_name}: {count} image(s)")
    print("")


def progress_line(index: int, total: int, result: ImageTrainingResult) -> str:
    """
    Build a formatted per-image progress line.

    Args:
        index (int): One-based image index.
        total (int): Total number of images.
        result (ImageTrainingResult): Result for the processed image.

    Returns:
        str: Human-readable progress line.
    """
    return (
        f"[{index:04d}/{total:04d}] {result.status:<12} "
        f"person={result.person_name:<16} "
        f"faces={result.faces_detected:<2} encodings={result.encodings_created:<2} "
        f"size={result.width}x{result.height} "
        f"time={result.elapsed_seconds:.2f}s "
        f"file={result.image_path.name} "
        f"{result.detail}"
    )


def encode_image(
    image_path: Path,
    upsample: int,
    jitter: int,
    multi_face_policy: str,
) -> tuple[list[Any], list[str], ImageTrainingResult]:
    """
    Detect and encode all faces in one image.

    Args:
        image_path (Path): Image file to process.
        upsample (int): Number of times to upsample during face detection.
        jitter (int): Number of jitter samples for face encoding.
        multi_face_policy (str): Policy for images with multiple detected faces.

    Returns:
        tuple[list[Any], list[str], ImageTrainingResult]: Encodings, names, and per-image result.
    """
    started_at = time.perf_counter()
    person_name = image_path.parent.name
    try:
        image = face_recognition.load_image_file(str(image_path))
    except Exception as exc:
        result = ImageTrainingResult(
            image_path=image_path,
            person_name=person_name,
            width=0,
            height=0,
            faces_detected=0,
            encodings_created=0,
            elapsed_seconds=time.perf_counter() - started_at,
            status="ERROR",
            detail=f"{type(exc).__name__}: {exc}",
        )
        return [], [], result

    height, width = image.shape[:2]
    try:
        boxes = face_recognition.face_locations(image, number_of_times_to_upsample=upsample, model="cnn")
    except Exception as exc:
        result = ImageTrainingResult(
            image_path=image_path,
            person_name=person_name,
            width=width,
            height=height,
            faces_detected=0,
            encodings_created=0,
            elapsed_seconds=time.perf_counter() - started_at,
            status="ERROR",
            detail=f"{type(exc).__name__}: {exc}",
        )
        return [], [], result

    if not boxes:
        result = ImageTrainingResult(
            image_path=image_path,
            person_name=person_name,
            width=width,
            height=height,
            faces_detected=0,
            encodings_created=0,
            elapsed_seconds=time.perf_counter() - started_at,
            status="SKIPPED",
            detail="no face detected",
        )
        return [], [], result

    status = "ENCODED"
    if len(boxes) > 1:
        if multi_face_policy == "skip":
            result = ImageTrainingResult(
                image_path=image_path,
                person_name=person_name,
                width=width,
                height=height,
                faces_detected=len(boxes),
                encodings_created=0,
                elapsed_seconds=time.perf_counter() - started_at,
                status="MULTI_FACE_SKIPPED",
                detail="multiple faces detected",
            )
            return [], [], result

        status = "MULTI_FACE_ENCODED"

    try:
        encodings = face_recognition.face_encodings(image, known_face_locations=boxes, num_jitters=jitter)
    except Exception as exc:
        result = ImageTrainingResult(
            image_path=image_path,
            person_name=person_name,
            width=width,
            height=height,
            faces_detected=len(boxes),
            encodings_created=0,
            elapsed_seconds=time.perf_counter() - started_at,
            status="ERROR",
            detail=f"{type(exc).__name__}: {exc}",
        )
        return [], [], result

    names = [person_name] * len(encodings)
    result = ImageTrainingResult(
        image_path=image_path,
        person_name=person_name,
        width=width,
        height=height,
        faces_detected=len(boxes),
        encodings_created=len(encodings),
        elapsed_seconds=time.perf_counter() - started_at,
        status=status,
        detail="",
    )
    return encodings, names, result


def print_training_summary(results: list[ImageTrainingResult], total_seconds: float) -> None:
    """
    Print aggregate training diagnostics.

    Args:
        results (list[ImageTrainingResult]): Per-image training results.
        total_seconds (float): Total training duration.

    Returns:
        None.
    """
    status_counts = Counter(result.status for result in results)
    per_person_images = Counter(result.person_name for result in results)
    per_person_encodings = defaultdict(int)

    for result in results:
        per_person_encodings[result.person_name] += result.encodings_created

    total_images = len(results)
    total_faces = sum(result.faces_detected for result in results)
    total_encodings = sum(result.encodings_created for result in results)
    no_encoding = sum(1 for result in results if result.encodings_created == 0)
    no_encoding_rate = no_encoding / total_images if total_images else 0.0
    images_per_second = total_images / total_seconds if total_seconds > 0 else 0.0

    print("")
    print("Training summary")
    print(f"  Images processed: {total_images}")
    print(f"  Faces detected: {total_faces}")
    print(f"  Encodings created: {total_encodings}")
    print(f"  Images without encodings: {no_encoding} ({no_encoding_rate:.1%})")
    print(f"  Total time: {total_seconds:.2f}s")
    print(f"  Throughput: {images_per_second:.2f} image(s)/second")
    print("")
    print("Status counts")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print("")
    print("Per-person output")
    for person_name in sorted(per_person_images):
        print(
            f"  {person_name}: "
            f"{per_person_images[person_name]} image(s), "
            f"{per_person_encodings[person_name]} encoding(s)"
        )

    skipped_images = [result for result in results if result.status in {"SKIPPED", "MULTI_FACE_SKIPPED", "ERROR"}]
    if skipped_images:
        print("")
        print("Skipped or failed image examples")
        for result in skipped_images[:10]:
            detail = f" ({result.detail})" if result.detail else ""
            print(f"  {result.status}: {result.image_path}{detail}")


def train_encodings(
    dataset_dir: Path,
    upsample: int,
    jitter: int,
    multi_face_policy: str,
    quiet: bool,
) -> dict[str, Any]:
    """
    Build a serialized training payload from collected face images.

    Args:
        dataset_dir (Path): Base dataset directory.
        upsample (int): Face detection upsample count.
        jitter (int): Encoding jitter count.
        multi_face_policy (str): Policy for images with multiple detected faces.
        quiet (bool): Whether to hide per-image progress details.

    Returns:
        dict[str, Any]: Training payload containing encodings, names, and metadata.
    """
    image_paths = iter_image_paths(dataset_dir)
    if not image_paths:
        raise RuntimeError(f"No training images found in {dataset_dir}.")

    summarize_dataset(image_paths)

    known_encodings = []
    known_names = []
    results = []
    started_at = time.perf_counter()
    total_images = len(image_paths)

    print("Training configuration")
    print(f"  Detector model: cnn")
    print(f"  Upsample: {upsample}")
    print(f"  Jitter: {jitter}")
    print(f"  Multi-face policy: {multi_face_policy}")
    print(f"  dlib version: {dlib.__version__}")
    print(f"  CUDA enabled: {dlib.DLIB_USE_CUDA}")
    print(f"  CUDA devices: {dlib.cuda.get_num_devices()}")
    print("")

    for index, image_path in enumerate(image_paths, start=1):
        encodings, names, result = encode_image(image_path, upsample, jitter, multi_face_policy)
        known_encodings.extend(encodings)
        known_names.extend(names)
        results.append(result)

        if not quiet:
            print(progress_line(index, total_images, result))

    total_seconds = time.perf_counter() - started_at
    print_training_summary(results, total_seconds)

    if not known_encodings:
        raise RuntimeError("No face encodings were produced. Collect clearer face images and retry.")

    return {
        "encodings": known_encodings,
        "names": known_names,
        "metadata": {
            "model": "cnn",
            "dlib_version": dlib.__version__,
            "cuda_enabled": dlib.DLIB_USE_CUDA,
            "cuda_devices": dlib.cuda.get_num_devices(),
            "images_processed": len(results),
            "faces_detected": sum(result.faces_detected for result in results),
            "encodings_created": len(known_encodings),
            "images_without_encodings": sum(1 for result in results if result.encodings_created == 0),
            "training_seconds": total_seconds,
            "upsample": upsample,
            "jitter": jitter,
            "multi_face_policy": multi_face_policy,
        },
    }


def save_training_payload(payload: dict[str, Any], output_path: Path) -> None:
    """
    Save face encodings to disk as a pickle file.

    Args:
        payload (dict[str, Any]): Training payload to serialize.
        output_path (Path): Destination pickle path.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(payload, file)


def main() -> None:
    """
    Run the GPU training workflow.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    ensure_cuda_available()
    payload = train_encodings(args.dataset_dir, args.upsample, args.jitter, args.multi_face_policy, args.quiet)
    save_training_payload(payload, args.output)
    print(f"Training complete. Saved {len(payload['encodings'])} encoding(s) to {args.output}.")


if __name__ == "__main__":
    main()
