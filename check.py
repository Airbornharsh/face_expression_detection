import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2


def get_emotion(index):
    if index == 0:
        return "angry"
    elif index == 1:
        return "disgust"
    elif index == 2:
        return "fear"
    elif index == 3:
        return "happy"
    elif index == 4:
        return "neutral"
    elif index == 5:
        return "sad"
    else:
        return "surprise"


input_img = "test/2.jpg"

# img = Image.open(input_img)
img = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)
# img = img.resize((48, 48), Image.ANTIALIAS)
img = cv2.resize(img, (48, 48))
img = np.asarray(img)
img = img.reshape(48, 48, 1) / 255.0
img = np.expand_dims(img, axis=0)

model = tf.keras.models.load_model("face_expression-v1.h5")

prediction = model.predict(img)

print(prediction)

for i in range(7):
    print(get_emotion(i), ":", int(prediction[0][i] * 100), "%")

prediction = np.argmax(prediction)

print(get_emotion(prediction))
