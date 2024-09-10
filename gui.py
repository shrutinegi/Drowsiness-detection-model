import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from pygame import mixer
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Initialize the alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Correctly load face and eye detection models with error checking
face_cascade_path = r"C:\Users\Shruti Negi\OneDrive\Desktop\drowsiness detection\haarcascade_frontalface_default.xml"
eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"

if not os.path.exists(face_cascade_path):
    raise FileNotFoundError(f"Face cascade file not found at {face_cascade_path}")

if not os.path.exists(eye_cascade_path):
    raise FileNotFoundError(f"Eye cascade file not found at {eye_cascade_path}")

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Load the drowsiness detection model with error checking
drowsiness_model_path = r"C:\Users\Shruti Negi\OneDrive\Desktop\drowsiness detection\model.keras"
age_model_path = r'C:\Users\Shruti Negi\OneDrive\Desktop\drowsiness detection\best_model.keras'

if not os.path.exists(drowsiness_model_path):
    raise FileNotFoundError(f"Drowsiness detection model not found at {drowsiness_model_path}")

if not os.path.exists(age_model_path):
    raise FileNotFoundError(f"Age detection model not found at {age_model_path}")

model = load_model(drowsiness_model_path)
age_model = load_model(age_model_path)

# Age groups typically predicted by the model
age_groups = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

lbl = ['Close', 'Open']
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

def process_frame(frame):
    result = "Awake"

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))

    drowsy_count = 0
    awake_count = 0
    detected_ages = []

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face_gray = gray[y:y + h, x:x + w]
        
        # Resize the face to match the model's expected input size
        face_resized = cv2.resize(face_gray, (128, 128))  # Resize to 128x128 for the age model
        
        # Normalize the face for the model
        face_normalized = face_resized / 255.0  # Normalize pixel values to the range [0, 1]
        face_normalized = np.expand_dims(face_normalized, axis=-1)  # Add a channel dimension
        face_normalized = np.expand_dims(face_normalized, axis=0)   # Add a batch dimension

        # Age detection on the face
        age_prediction = age_model.predict(face_normalized)

        # Assuming the model outputs a probability distribution over age categories
        if isinstance(age_prediction, np.ndarray):
            if isinstance(age_prediction[0], np.ndarray) and age_prediction[0].shape == (2, 1):
                # After model prediction
                continuous_age = age_prediction[1][0][0]

                # Logging the continuous age for debugging
                print(f"Raw continuous age prediction: {continuous_age}")

                # Determine the closest age group
                closest_age_group = min(age_groups, key=lambda age: abs(int(age.split('-')[0].strip('()')) - continuous_age))

                detected_age = closest_age_group
                print(f"Detected age group: {detected_age}")

            else:
                age_index = np.argmax(age_prediction)
                detected_age = age_groups[age_index]
        else:
            combined_age_predictions = np.array([pred[0] for pred in age_prediction])
            age_index = np.argmax(combined_age_predictions)
            detected_age = age_groups[age_index]

        detected_ages.append(detected_age)

        print(f"Detected age: {detected_age}")

        eyes = eye_cascade.detectMultiScale(face_gray, minNeighbors=1, scaleFactor=1.1)

        eye_results = []

        for (ex, ey, ew, eh) in eyes:
            eye = face[ey:ey + eh, ex:ex + ew]
            eye = cv2.resize(eye, (80, 80))
            eye = eye / 255.0
            eye = eye.reshape(80, 80, 3)
            eye = np.expand_dims(eye, axis=0)
            prediction = model.predict(eye)

            eye_results.append(prediction[0])

        if len(eye_results) > 0:
            avg_prediction = np.mean(eye_results, axis=0)

            if avg_prediction[0] > 0.45:
                result = "Drowsy"
                drowsy_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"Drowsy {detected_age}", (x, y - 10), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                result = "Awake"
                awake_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Awake {detected_age}", (x, y - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame, result, drowsy_count, detected_ages


def show_image(image):
    global panel, current_zoom

    if image is None:
        return  # Safeguard against None image

    # Convert BGR to RGB for correct color display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image based on the current zoom level
    height, width = image_rgb.shape[:2]
    new_width = max(1, int(width * (current_zoom / 100)))  # Ensure width is at least 1
    new_height = max(1, int(height * (current_zoom / 100)))  # Ensure height is at least 1
    resized_image = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Convert the image to PIL format
    im = Image.fromarray(resized_image)
    imgtk = ImageTk.PhotoImage(image=im)

    # Update the canvas with the new image
    panel.create_image(0, 0, anchor="nw", image=imgtk)
    panel.imgtk = imgtk  # Keep a reference to avoid garbage collection
    panel.config(scrollregion=panel.bbox("all"))


def open_image():
    global processed_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        processed_image = cv2.imread(file_path)
        processed_image, result, drowsy_count, detected_ages = process_frame(processed_image)
        show_image(processed_image)
        messagebox.showinfo("Result", f"Detection Result: {result}\nDrowsy Count: {drowsy_count}\nAges: {', '.join(detected_ages)}")

def open_video():
    global cap
    cap = cv2.VideoCapture(0)  # Capture from webcam
    video_loop()

def upload_video():
    global cap
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        video_loop()

def video_loop():
    global cap
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            processed_image, result, drowsy_count, detected_ages = process_frame(frame)
            show_image(processed_image)
            if drowsy_count > 0:
                sound.play()  # Play alarm sound
            root.after(10, video_loop)
        else:
            cap.release()

def stop_video():
    global cap
    if cap:
        cap.release()

def zoom(event):
    global current_zoom, processed_image
    if processed_image is None:
        return  # Do nothing if no image is loaded

    if event.delta > 0:
        current_zoom += 10  # Zoom in
    else:
        current_zoom -= 10  # Zoom out
    
    # Restrict zoom level between 10% and 200%
    current_zoom = max(10, min(200, current_zoom))
    
    # Update the image display with the new zoom level
    show_image(processed_image)


def create_gui():
    global root, panel, current_zoom, processed_image
    current_zoom = 100  # Default zoom level
    processed_image = None  # To store the processed image for zooming

    root = tk.Tk()
    root.title("Drowsiness and Age Detection")

    # Create a canvas to hold the image
    panel = tk.Canvas(root)
    panel.pack(side="left", fill="both", expand=True)

    # Create vertical and horizontal scrollbars
    vbar = tk.Scrollbar(root, orient="vertical", command=panel.yview)
    vbar.pack(side="right", fill="y")
    hbar = tk.Scrollbar(root, orient="horizontal", command=panel.xview)
    hbar.pack(side="bottom", fill="x")

    # Configure the canvas to work with the scrollbars
    panel.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

    # Bind mouse scroll to zoom function
    panel.bind("<MouseWheel>", zoom)

    # Buttons
    btn_open_image = tk.Button(root, text="Open Image", command=open_image)
    btn_open_image.pack(side="top", padx=10,  pady=5)

    # Buttons
    btn_open_video = tk.Button(root, text="Open Webcam", command=open_video)
    btn_open_video.pack(side="top", padx=10, pady=5)

    btn_upload_video = tk.Button(root, text="Upload Video", command=upload_video)
    btn_upload_video.pack(side="top", padx=10, pady=5)

    btn_stop_video = tk.Button(root, text="Stop Video", command=stop_video)
    btn_stop_video.pack(side="top", padx=10, pady=5)

    root.mainloop()

create_gui()
