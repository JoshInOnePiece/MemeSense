#import sys
#print(sys.executable)
# Imports necessary modules.
import mediapipe as mp
import cv2
import os
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None


def overlay_image(background, overlay, x, y, max_width=260, max_height=260):
    if overlay is None:
        return

    h, w = overlay.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

    y1, y2 = y, min(y + new_h, background.shape[0])
    x1, x2 = x, min(x + new_w, background.shape[1])
    overlay_cropped = resized[: y2 - y1, : x2 - x1]

    if overlay_cropped.shape[2] == 4:
        alpha = overlay_cropped[:, :, 3] / 255.0
        alpha = alpha[:, :, None]
        background[y1:y2, x1:x2] = (
            alpha * overlay_cropped[:, :, :3] +
            (1 - alpha) * background[y1:y2, x1:x2]
        ).astype("uint8")
    else:
        background[y1:y2, x1:x2] = overlay_cropped

# --- IMPORTANT NOTE FOR LOCAL EXECUTION ---
# This code is designed to run in a local Python environment on your machine,
# not directly within this Google Colab notebook, as Colab generally cannot
# access local webcams. Save this code as a .py file and run it locally.
# Ensure 'gesture_recognizer.task' is in the same directory or update its path.
# ------------------------------------------

# Create a GestureRecognizer object.
# Ensure this path is correct relative to where you run the script locally.
model_path = os.path.relpath("./Models/baseline_25imgs.task")

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}. ")
    print("Please ensure"
    " is in the correct directory.")
else:
    print(f"Loading model from: {model_path}")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # Initialize webcam capture
    cap = cv2.VideoCapture(0) # 0 indicates the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam. Please check if it's connected and not in use.")
    else:
        print("Webcam opened successfully. Press 'q' to quit.")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30

        vcam = None
        if pyvirtualcam is None:
            print("pyvirtualcam is not installed. Telegram virtual camera output is disabled.")
            print("Install with: pip install pyvirtualcam")
        else:
            try:
                vcam = pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=int(fps))
                print(f"Virtual camera started: {vcam.device}")
                print("In Telegram Desktop, select this virtual camera as your video source.")
            except Exception as err:
                print(f"Warning: Could not start virtual camera: {err}")
                print("Tip: Install/enable OBS Virtual Camera driver on Windows, then retry.")

        meme_hold_seconds = 1.5
        meme_map = {
            "timeout": cv2.imread("./images/shaqTimeout.jpg", cv2.IMREAD_UNCHANGED),
            "stop": cv2.imread("./images/JERMAINE.PNG", cv2.IMREAD_UNCHANGED),
            "fist": cv2.imread("./images/baby.jpeg", cv2.IMREAD_UNCHANGED),
            "one": cv2.imread("./images/monkeyAha.jpeg", cv2.IMREAD_UNCHANGED),
            "think": cv2.imread("./images/monkeyThink.jpg", cv2.IMREAD_UNCHANGED),
            "holy": cv2.imread("./images/prayin-samil.jpg", cv2.IMREAD_UNCHANGED),
        }
        persisted_meme = None
        last_detected_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            # Convert the BGR image from OpenCV to RGB, as MediaPipe typically expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create an mp.Image object from the RGB numpy array
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Run gesture recognition.
            recognition_result = recognizer.recognize(mp_image)

            # Important: draw overlays on a copy so the model always sees raw camera pixels.
            display_frame = frame.copy()
            chosen_meme = None
            now = time.monotonic()

            # Display the most likely gesture on the frame
            if recognition_result.gestures and recognition_result.gestures[0]:
                top_gesture = recognition_result.gestures[0][0]
                gesture_text = f"Gesture: {top_gesture.category_name} ({top_gesture.score:.2f})"
                display = top_gesture.category_name
                cv2.putText(display_frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                chosen_meme = meme_map.get(display)
                if chosen_meme is not None:
                    persisted_meme = chosen_meme
                    last_detected_time = now
            else:
                cv2.putText(display_frame, "No gesture detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if chosen_meme is not None:
                x_pos = max(display_frame.shape[1] - 280, 10)
                overlay_image(display_frame, chosen_meme, x_pos, 10)
            elif persisted_meme is not None and (now - last_detected_time) <= meme_hold_seconds:
                x_pos = max(display_frame.shape[1] - 280, 10)
                overlay_image(display_frame, persisted_meme, x_pos, 10)
            else:
                persisted_meme = None

            if vcam is not None:
                vcam.send(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                vcam.sleep_until_next_frame()

            cv2.imshow("Gesture Recognition", display_frame)

            # Break the loop when 'q' is pressed    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and destroy all OpenCV windows
        if vcam is not None:
            vcam.close()
        cap.release()
        cv2.destroyAllWindows()
