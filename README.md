# MemeSense

MemeSense is a real-time gesture-to-meme reaction system. It uses a webcam, a custom MediaPipe Gesture Recognizer model, and OpenCV to detect hand gestures and display matching reaction meme images during live video workflows.

The project is designed for experimentation with custom gesture-recognition models, model ablation testing, real-time webcam inference, and optional Telegram Desktop video-call integration through a virtual camera.

## Features

- Real-time hand-gesture recognition from a webcam
- Custom gesture-to-meme image mapping
- MediaPipe `.task` model support
- Single-model inference for local testing
- Multi-model comparison with confidence overlays and CSV logging
- Optional Telegram Desktop integration through virtual-camera output
- Jupyter/Colab-based model training and experimentation workflow

## Project Structure

| Path | Purpose |
| --- | --- |
| `Abolation_training-3.ipynb` | Colab notebook for training and ablation experiments |
| `gesture_recognizer.ipynb` | Gesture-recognition notebook workflow |
| `HagridSubset.ipynb` | Dataset preparation or subset workflow |
| `dataaugmenter.py` | Data augmentation utilities |
| `run.py` | Runs a single gesture-recognition model locally with webcam input |
| `run_multi_models_embedded.py` | Runs multiple models, displays confidence scores, and logs results to CSV |
| `run_for_telegram.py` | Runs gesture recognition with optional virtual-camera output for Telegram Desktop |
| `Models/` | Pretrained and experimental `.task` model files |
| `images/` | Meme images displayed when gestures are recognized |

## Requirements

MemeSense is intended for Linux or other Unix-like environments. It can also be adapted for Windows or macOS, but camera and virtual-camera setup may vary by platform.

You will need:

- Python 3.9+
- A working webcam
- Google Colab, if you plan to retrain models
- Telegram Desktop, if using the Telegram workflow
- OBS or another supported virtual-camera backend, if using virtual-camera output

Python packages:

```bash
mediapipe
opencv-python
pyvirtualcam   # Optional, only needed for virtual-camera output
```

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/JoshInOnePiece/MemeSense.git
cd MemeSense
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install mediapipe opencv-python
```

For Telegram or virtual-camera support, also install:

```bash
python -m pip install pyvirtualcam
```

## Training a Custom Model

To train or compare custom models:

1. Open `Abolation_training-3.ipynb` in Google Colab.
2. Follow the notebook cells to train gesture-recognition models.
3. Review the reported metrics, including F1 score and other evaluation results.
4. Download the generated `.task` model files.
5. Place the desired model file in the `Models/` directory.

The default single-model script expects a model at:

```text
Models/500_images_train.task
```

To use a different model, either rename your model to `500_images_train.task` or update the `model_path` value in `run.py`.

## Running MemeSense Locally

Run the single-model webcam demo:

```bash
python run.py
```

The script opens your webcam, runs gesture recognition, and displays the associated meme image when a known gesture is detected.

Press `q` to quit.

## Comparing Multiple Models

To compare several `.task` models in real time, run:

```bash
python run_multi_models_embedded.py
```

This script loads available models, shows per-model confidence scores, and writes results to:

```text
gesture_accuracy_log.csv
```

On case-sensitive systems, confirm that the model directory name used by the script matches your repository. The repository folder is named `Models/`, while some code may reference `models/`.

## Telegram Desktop Integration

MemeSense can display meme overlays in a Telegram Desktop video call through virtual-camera output.

Setup steps:

1. Install OBS or another supported virtual-camera backend.
2. Install the optional Python dependency:

   ```bash
   python -m pip install pyvirtualcam
   ```

3. Start or enable your system’s virtual camera.
4. Run the Telegram script:

   ```bash
   python run_for_telegram.py
   ```

5. In Telegram Desktop, select the virtual camera as your video input.
6. Start a video call and perform one of the supported gestures.

When a recognized gesture is detected, the matching meme image is overlaid on the outgoing video feed.

## Gesture Mapping

The available mappings depend on the script and model being used. Current mappings include:

| Gesture | Meme image |
| --- | --- |
| `timeout` | `shaqTimeout.jpg` |
| `stop` | `JERMAINE.PNG` |
| `fist` | `baby.jpeg` |
| `holy` | `prayin-samil.jpg` |
| `one` | `monkeyAha.jpeg` |
| `think` | `monkeyThink.jpg` |

Some gestures may only work with the included pretrained models or with specific scripts.

## Troubleshooting

### Model file not found

Check that your `.task` file exists in the expected directory and that the filename matches the script configuration.

### Webcam does not open

Make sure your webcam is connected, not being used by another application, and that the correct camera index is set in the script.

### No virtual camera appears in Telegram

Install and enable a supported virtual-camera backend, then restart Telegram Desktop. If `pyvirtualcam` is not installed, the Telegram script will still run locally but virtual-camera output will be disabled.

### Gesture is detected but no meme appears

Verify that the gesture label exists in the script’s gesture map and that the corresponding image file exists in the `images/` directory.

## Notes

- Lighting, camera angle, distance from the camera, and hand visibility can affect recognition accuracy.
- The included models and image mappings are intended for experimentation and demonstration.
- For best results, keep model names, image names, and gesture labels consistent across training and runtime scripts.

## License

No license file is currently included in the repository. Add a license before distributing or reusing this project publicly.

## Acknowledgments

MemeSense uses MediaPipe for gesture recognition and OpenCV for webcam capture and real-time image display.
