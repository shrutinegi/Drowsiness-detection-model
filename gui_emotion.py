import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import tkinter as tk
from tkinter import filedialog, Scale, HORIZONTAL, messagebox
from PIL import Image, ImageTk

# Initialize the alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load face and eye detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load the drowsiness detection model
model = load_model(os.path.join("models", "model.keras"))

lbl = ['Close', 'Open']

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Global variables
panel = None
cap = None
zoom_scale = None

def process_frame(frame):
    result = "Awake"

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))

    drowsy_count = 0
    awake_count = 0

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face_gray = gray[y:y + h, x:x + w]
        
        eyes = eye_cascade.detectMultiScale(face_gray, minNeighbors=1, scaleFactor=1.1)

        eye_results = []

        for (ex, ey, ew, eh) in eyes:
            eye = face[ey:ey + eh, ex:ex + ew]
            eye = cv2.resize(eye, (80, 80))
            eye = eye / 255.0
            eye = eye.reshape(80, 80, 3)
            eye = np.expand_dims(eye, axis=0)
            prediction = model.predict(eye)

            # Add the prediction result for each eye
            eye_results.append(prediction[0])

        if len(eye_results) > 0:
            # Average the results of both eyes (if both eyes are detected)
            avg_prediction = np.mean(eye_results, axis=0)

            # Adjusted thresholds
            if avg_prediction[0] > 0.40:  # More confidence required to mark as drowsy
                # you can adjust between 0.40-0.50 best result
                result = "Drowsy"
                drowsy_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Drowsy", (x, y - 10), font, 10, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                result = "Awake"
                awake_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Awake", (x, y - 10), font, 10, (0, 255, 0), 2, cv2.LINE_AA)

    return frame, result, drowsy_count

def show_image(image):
    global panel

    # Convert BGR to RGB for correct color display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image according to the zoom level
    scale_percent = zoom_scale.get()  # Get zoom level from scale widget
    width = int(image_rgb.shape[1] * scale_percent / 1000)
    height = int(image_rgb.shape[0] * scale_percent / 1000)
    resized_image = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)

    # Convert the image to PIL format
    im = Image.fromarray(resized_image)
    imgtk = ImageTk.PhotoImage(image=im)

    # Update the image on the panel
    panel.config(image=imgtk)
    panel.image = imgtk

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        image = cv2.imread(file_path)
        processed_image, result, drowsy_count = process_frame(image)
        show_image(processed_image)
        messagebox.showinfo("Result", f"Detection Result: {result}\nDrowsy Count: {drowsy_count}")

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
            processed_image, result, drowsy_count = process_frame(frame)
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

def create_gui():
    global root, panel, zoom_scale

    root = tk.Tk()
    root.title("Drowsiness Detection")

    # Create a panel to display the image
    panel = tk.Label(root)
    panel.pack()

    # Create a scale widget for zooming in and out
    zoom_scale = Scale(root, from_=50, to=200, orient=HORIZONTAL, label="Zoom (%)")
    zoom_scale.set(100)  # Default zoom level at 100%
    zoom_scale.pack(pady=10)

    # Buttons
    btn_open_image = tk.Button(root, text="Open Image", command=open_image)
    btn_open_image.pack(side="left", padx=10, pady=10)

    btn_open_video = tk.Button(root, text="Start Webcam", command=open_video)
    btn_open_video.pack(side="left", padx=10, pady=10)

    btn_upload_video = tk.Button(root, text="Upload Video", command=upload_video)
    btn_upload_video.pack(side="left", padx=10, pady=10)

    btn_stop_video = tk.Button(root, text="Stop Video", command=stop_video)
    btn_stop_video.pack(side="left", padx=10, pady=10)

    btn_exit = tk.Button(root, text="Exit", command=root.quit)
    btn_exit.pack(side="right", padx=10, pady=10)

    root.mainloop()

create_gui()
