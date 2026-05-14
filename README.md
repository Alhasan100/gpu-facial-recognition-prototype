# GPU Facial Recognition Prototype

A webcam-based facial recognition project built with Python, OpenCV, dlib, CUDA, cuDNN, and the `face_recognition` library.

The project is split into three clear steps:

1. Capture face images from a webcam.
2. Train 128-dimensional face encodings with dlib's CNN face detector.
3. Run real-time recognition from the webcam with GPU acceleration.

The code is designed as a practical prototype for learning, testing, and extending facial recognition workflows. It includes a generic protected-action placeholder that can be replaced by application-specific behavior.

## What It Uses

- **OpenCV** for webcam capture and drawing the live interface.
- **dlib CNN face detector** for GPU-accelerated face detection.
- **face_recognition** for face encoding and comparison.
- **CUDA + cuDNN** for NVIDIA GPU acceleration.
- **Pickle** for storing generated face encodings.

## Project Files

| File | Description |
| --- | --- |
| `01_collect_data.py` | Opens the webcam and saves labeled training images. |
| `02_train_gpu.py` | Reads the dataset, detects faces with the CNN model, creates encodings, and saves them. |
| `03_recognize.py` | Runs live webcam recognition with status text, FPS, CUDA info, and match distances. |
| `install_windows.cmd` | One-click Windows installer. |
| `scripts/install_windows.ps1` | Full Windows setup script. |
| `scripts/build_dlib_cuda_windows.ps1` | Rebuilds dlib with CUDA and cuDNN support on Windows. |

## Requirements

For the GPU setup, use:

- Windows 10 or Windows 11.
- Python 3.10 or newer.
- NVIDIA GPU.
- Current NVIDIA driver.
- CUDA Toolkit.
- CMake.
- Visual Studio Build Tools with the C++ workload.
- Webcam supported by OpenCV.

The Windows installer expects `cmake`, `nvcc`, and Visual Studio C++ Build Tools to be available.

## Quick Start on Windows

Clone the repository:

```powershell
git clone https://github.com/Alhasan100/gpu-facial-recognition-prototype.git
cd gpu-facial-recognition-prototype
```

Run the one-command installer:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1 -CudaArchitectures 89
```

For many RTX 40-series GPUs, `89` is the correct CUDA architecture. If unsure, try the default installer first:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1
```

You can also double-click:

```text
install_windows.cmd
```

The installer will:

- create `.venv`
- install Python dependencies
- install the cuDNN Python package
- build dlib from source with CUDA enabled
- verify that dlib can see the GPU
- print the next commands to run

## Verify GPU Support

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Check CUDA support:

```powershell
python -c "import dlib; print('DLIB_USE_CUDA:', dlib.DLIB_USE_CUDA); print('CUDA devices:', dlib.cuda.get_num_devices() if dlib.DLIB_USE_CUDA else 0)"
```

Expected output:

```text
DLIB_USE_CUDA: True
CUDA devices: 1
```

If `DLIB_USE_CUDA` is `False`, dlib is installed in CPU-only mode. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_dlib_cuda_windows.ps1 -CudaArchitectures 89
```

## Step 1: Collect Training Images

Run:

```powershell
python .\01_collect_data.py --name person_name
```

Replace `person_name` with the label you want to train.

Controls:

- Press `C` to capture an image.
- Press `Q` or `Esc` to quit.
- Closing the webcam window also exits.

The script saves images like this:

```text
dataset/
  person_name/
    person_name_YYYYMMDD_HHMMSS_microseconds_0001.jpg
```

Recommended capture guidelines:

- Use one visible face per image.
- Capture 30-100 images per identity.
- Include front-facing and slight side-angle images.
- Include the distance where recognition should work.
- Avoid blurry frames.
- Avoid strong backlight.
- Keep the face clearly visible.

## Step 2: Train Face Encodings

Run:

```powershell
python .\02_train_gpu.py
```

The trainer uses dlib's CNN detector and creates 128-dimensional face encodings.

The output file is:

```text
encodings/face_encodings.pickle
```

The trainer prints:

- dataset summary
- CUDA status
- dlib version
- detected face count
- skipped images
- multi-face image count
- encodings created per identity
- total training time
- images per second

Quiet mode:

```powershell
python .\02_train_gpu.py --quiet
```

Useful training options:

```powershell
python .\02_train_gpu.py --upsample 1 --jitter 1 --multi-face-policy skip
```

By default, images with multiple detected faces are skipped. This prevents assigning multiple different faces to the same identity.

## Step 3: Run Real-Time Recognition

Run:

```powershell
python .\03_recognize.py --debug
```

Controls:

- Press `Q` or `Esc` to quit.
- Closing the webcam window also exits.

The live overlay shows:

- recognition status
- detected face count
- FPS
- CUDA device count
- detection scale
- tolerance
- upsample setting
- match distance for unknown faces

## Recognition Tuning

Use these options when recognition needs adjustment:

| Option | What it does |
| --- | --- |
| `--scale` | Higher values preserve more image detail. Useful for distant faces, but slower. |
| `--upsample` | Helps detect smaller faces. Higher values are slower. |
| `--tolerance` | Controls how strict matching is. Lower is stricter, higher is looser. |
| `--debug` | Prints frame-by-frame detection and match information. |

Recommended default:

```powershell
python .\03_recognize.py --scale 0.5 --upsample 1 --tolerance 0.52 --debug
```

For longer distance:

```powershell
python .\03_recognize.py --scale 0.75 --upsample 1 --tolerance 0.52 --debug
```

For stricter matching:

```powershell
python .\03_recognize.py --scale 0.5 --upsample 1 --tolerance 0.45 --debug
```

For looser matching:

```powershell
python .\03_recognize.py --scale 0.5 --upsample 1 --tolerance 0.55 --debug
```

## Common Problems

### `DLIB_USE_CUDA` is `False`

dlib was installed without CUDA support.

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_dlib_cuda_windows.ps1 -CudaArchitectures 89
```

Then verify again:

```powershell
python -c "import dlib; print(dlib.DLIB_USE_CUDA); print(dlib.cuda.get_num_devices() if dlib.DLIB_USE_CUDA else 0)"
```

### No Face Detected

Try higher detection detail:

```powershell
python .\03_recognize.py --scale 0.75 --upsample 1 --debug
```

If detection is still weak, collect more images at the same distance and lighting where recognition should work.

### Face Detected but Not Recognized

Try a slightly higher tolerance:

```powershell
python .\03_recognize.py --tolerance 0.55 --debug
```

If matching becomes too loose, lower the tolerance and improve the dataset.

### Many Images Are Skipped During Training

The trainer skips multi-face images by default.

Best fix:

- collect images with only one visible face
- retrain

Override only when every detected face belongs to the same identity:

```powershell
python .\02_train_gpu.py --multi-face-policy encode-all
```

## Generated Files

These files and folders are generated during normal use and ignored by Git:

- `.venv/`
- `dataset/`
- `encodings/`
- `.dlib-cuda-build/`
- image files
- pickle files
- logs

The repository should contain source code and documentation only.

## Security Notice

This is a prototype, not a complete production authentication system.

Before using it for authentication or access control, add:

- liveness detection
- anti-spoofing
- encrypted encoding storage
- controlled enrollment
- audit logging
- rate limiting
- target-platform security review

## License

MIT License. See `LICENSE`.
