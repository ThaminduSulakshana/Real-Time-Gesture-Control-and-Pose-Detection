# Import necessary libraries
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Initialize an array to hold our "bag of words"
bag_of_words = []

def extract_features(frame):
    """Extract features (histogram) from a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray, bins=256, range=[0, 256])
    return hist

def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)  # Use the webcam (ID 0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise Exception("Could not open video device")

    try:
        while True:
            # Capture frame-by-frame
            ret, img = cap.read()
            if not ret:
                break

            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

            # Extract features (BoW "words") from the frame
            features = extract_features(img)

            # Append the features to our "bag of words"
            bag_of_words.append(features)

            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)

    except GeneratorExit:
        # Release the video capture object when the generator is closed
        cap.release()

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
