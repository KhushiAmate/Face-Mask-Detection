import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('models/face_mask_model.keras')

def detect_mask_on_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    predictions = []

    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))
        face = np.array(face, dtype="float32") / 255.0
        face = np.expand_dims(face, axis=0)
        prediction = model.predict(face)
        mask_label = 'Mask' if prediction[0][0] < 0.5 else 'No Mask'
        predictions.append((x, y, w, h, mask_label))

    return predictions
