#import sys
#print(sys.executable)
# Imports necessary modules.
import mediapipe as mp
import cv2
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- IMPORTANT NOTE FOR LOCAL EXECUTION ---
# This code is designed to run in a local Python environment on your machine,
# not directly within this Google Colab notebook, as Colab generally cannot
# access local webcams. Save this code as a .py file and run it locally.
# Ensure 'gesture_recognizer.task' is in the same directory or update its path.
# ------------------------------------------

# Create a GestureRecognizer object.
# Ensure this path is correct relative to where you run the script locally.
model_path = os.path.relpath("./Models/500_images_train.task")

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

            # Display the most likely gesture on the frame
            if recognition_result.gestures and recognition_result.gestures[0]:
                top_gesture = recognition_result.gestures[0][0]
                gesture_text = f"Gesture: {top_gesture.category_name} ({top_gesture.score:.2f})"
                display = top_gesture.category_name
                cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if display == "timeout":
                    img = cv2.imread('./images/shaqTimeout.jpg')
                    cv2.imshow('SHAQ', img)
                elif display == "stop":
                    img = cv2.imread('./images/JERMAINE.PNG')
                    cv2.imshow('JERMAINE', img)
                elif display == "fist":
                    img = cv2.imread('./images/baby.jpeg')
                    cv2.imshow('baby', img)
                    
                else:
                    cv2.imshow('Gesture Recognition', frame)
            else:
                cv2.putText(frame, "No gesture detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Gesture Recognition', frame)
            # Display the frame
                

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and destroy all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
