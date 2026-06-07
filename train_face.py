import os
import cv2
import pickle
import numpy as np

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

data = []
labels = []

dataset_path = "dataset"

for person in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person)

    for image in os.listdir(person_folder):

        img_path = os.path.join(person_folder, image)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (100,100))

        data.append(img)
        labels.append(person)

X = np.array(data, dtype="float32") / 255.0

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

pickle.dump(
    encoder,
    open("models/label_encoder.pkl", "wb")
)

y = to_categorical(y)

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(100,100,3)
    )
)

model.add(MaxPooling2D())

model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(
    Dense(
        y.shape[1],
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X,
    y,
    epochs=20,
    batch_size=16
)

model.save("models/face_model.h5")

print("Training Completed")