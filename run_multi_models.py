import csv
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
CSV_PATH = os.path.join(SCRIPT_DIR, "gesture_accuracy_log.csv")
PRIMARY_MODEL_NAME = "baseline_25imgs"
CAMERA_INDEX = 1
RECOGNITION_INTERVAL = 0.01

GESTURE_IMAGE_MAP = {
    "timeout": ("SHAQ", "shaqTimeout.jpg"),
    "stop": ("JERMAINE", "JERMAINE.PNG"),
    "fist": ("baby", "baby.jpeg"),
    "one": ("monkey", "monkeyAha.jpeg"),
    "think": ("MonkeyThink", "monkeyThink.jpg"),
}


def load_recognizers(models_dir):
    model_files = sorted(
        file_name for file_name in os.listdir(models_dir) if file_name.endswith(".task")
    )
    if not model_files:
        raise FileNotFoundError(f"No .task files found in {models_dir}")

    recognizers = {}
    for file_name in model_files:
        model_name = os.path.splitext(file_name)[0]
        model_path = os.path.join(models_dir, file_name)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(base_options=base_options)
        recognizers[model_name] = vision.GestureRecognizer.create_from_options(options)

    return recognizers


def get_primary_model_name(model_names):
    if PRIMARY_MODEL_NAME in model_names:
        return PRIMARY_MODEL_NAME
    return model_names[0]


def extract_scores(recognition_result):
    scores = {}
    if recognition_result.gestures and recognition_result.gestures[0]:
        for gesture in recognition_result.gestures[0]:
            scores[gesture.category_name] = gesture.score
    return scores


def ensure_csv_header(csv_path, model_names):
    if os.path.exists(csv_path):
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "trigger_type", "gesture", *model_names])


def append_csv_row(csv_path, trigger_type, gesture_name, model_names, model_scores):
    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        trigger_type,
        gesture_name,
        *[f"{model_scores.get(model_name, 0.0):.6f}" for model_name in model_names],
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(row)


def load_target_image(gesture_name):
    if gesture_name not in GESTURE_IMAGE_MAP:
        return None, None

    window_name, image_name = GESTURE_IMAGE_MAP[gesture_name]
    image_path = os.path.join(IMAGES_DIR, image_name)
    image = cv2.imread(image_path)
    return window_name, image


def main():
    if not os.path.isdir(MODELS_DIR):
        print(f"Error: Models directory not found at {MODELS_DIR}")
        return

    try:
        recognizers = load_recognizers(MODELS_DIR)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    model_names = list(recognizers.keys())
    primary_model_name = get_primary_model_name(model_names)
    ensure_csv_header(CSV_PATH, model_names)

    print("Loaded models:")
    for model_name in model_names:
        marker = " (primary display model)" if model_name == primary_model_name else ""
        print(f"  - {model_name}{marker}")
    print(f"CSV logging path: {CSV_PATH}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check if it's connected and not in use.")
        for recognizer in recognizers.values():
            recognizer.close()
        return

    print("Webcam opened successfully. Press 'q' to quit, 'r' to record current scores.")

    current_meme_window = None
    camera_window_name = "Gesture Recognition"
    last_recognition_time = 0.0
    last_logged_gesture = None
    last_scores_by_model = {model_name: {} for model_name in model_names}
    last_detected_gesture = None
    gesture_text = "No gesture detected"
    gesture_color = (0, 0, 255)
    target_window = None
    target_image = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            current_time = time.monotonic()
            if current_time - last_recognition_time >= RECOGNITION_INTERVAL:
                last_recognition_time = current_time

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                for model_name, recognizer in recognizers.items():
                    recognition_result = recognizer.recognize(mp_image)
                    last_scores_by_model[model_name] = extract_scores(recognition_result)

                primary_scores = last_scores_by_model[primary_model_name]
                gesture_text = "No gesture detected"
                gesture_color = (0, 0, 255)
                target_window = None
                target_image = None
                last_detected_gesture = None

                if primary_scores:
                    detected_gesture, detected_score = max(
                        primary_scores.items(), key=lambda item: item[1]
                    )
                    gesture_text = (
                        f"{primary_model_name}: {detected_gesture} ({detected_score:.2f})"
                    )
                    gesture_color = (0, 255, 0)
                    last_detected_gesture = detected_gesture
                    target_window, target_image = load_target_image(detected_gesture)

                    if detected_gesture in GESTURE_IMAGE_MAP:
                        model_scores = {
                            model_name: last_scores_by_model[model_name].get(detected_gesture, 0.0)
                            for model_name in model_names
                        }
                        if last_logged_gesture != detected_gesture:
                            append_csv_row(
                                CSV_PATH,
                                "auto",
                                detected_gesture,
                                model_names,
                                model_scores,
                            )
                            print(
                                f"Auto-logged '{detected_gesture}' scores to "
                                f"{os.path.basename(CSV_PATH)}"
                            )
                            last_logged_gesture = detected_gesture
                else:
                    last_logged_gesture = None

            if current_meme_window and current_meme_window != target_window:
                cv2.destroyWindow(current_meme_window)
                current_meme_window = None

            cv2.putText(
                frame,
                gesture_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                gesture_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press 'r' to log current model scores",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(camera_window_name, frame)

            if target_window and target_image is not None:
                cv2.imshow(target_window, target_image)
                current_meme_window = target_window

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("r"):
                if last_detected_gesture is None:
                    print("No gesture detected. Nothing was written to the CSV.")
                    continue

                model_scores = {
                    model_name: last_scores_by_model[model_name].get(last_detected_gesture, 0.0)
                    for model_name in model_names
                }
                append_csv_row(
                    CSV_PATH,
                    "manual",
                    last_detected_gesture,
                    model_names,
                    model_scores,
                )
                print(
                    f"Manually logged '{last_detected_gesture}' scores to "
                    f"{os.path.basename(CSV_PATH)}"
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for recognizer in recognizers.values():
            recognizer.close()


if __name__ == "__main__":
    main()
