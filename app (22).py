from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from PIL import Image
import time

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads/'
CASCADE_PATH = 'haarcascade/haarcascade_frontalface_default.xml'

# --- App setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        # Save uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Run face detection
        result, num_faces, time_taken = detect_faces(filepath)

        return render_template('index.html', 
                               filename=file.filename,
                               num_faces=num_faces,
                               time_taken=round(time_taken, 2))

    return render_template('index.html')

# --- Face detection function ---
def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    img_bgr = cv2.imread(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    start = time.time()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    end = time.time()

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # Save result
    output_path = image_path  # overwrite the same file
    cv2.imwrite(output_path, img_bgr)

    return output_path, len(faces), end - start

# --- Run app ---
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
