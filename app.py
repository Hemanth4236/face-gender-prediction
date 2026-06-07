from flask import Flask
from flask import render_template
from flask import Response

import cv2
import pickle
import numpy as np

from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("models/face_model.h5")

encoder = pickle.load(
    open(
        "models/label_encoder.pkl",
        "rb"
    )
)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            1.3,
            5
        )

        for (x,y,w,h) in faces:

            face = frame[y:y+h, x:x+w]

            face = cv2.resize(
                face,
                (100,100)
            )

            face = face / 255.0

            face = np.expand_dims(
                face,
                axis=0
            )

            prediction = model.predict(
                face,
                verbose=0
            )

            person_id = np.argmax(
                prediction
            )

            confidence = (
                np.max(prediction) * 100
            )

            name = encoder.inverse_transform(
                [person_id]
            )[0]

            gender = "Male"

            label = (
                f"{name} | "
                f"{gender} | "
                f"{confidence:.2f}%"
            )

            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )