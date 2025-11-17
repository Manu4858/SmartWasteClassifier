
import tensorflow as tf
import cv2
import numpy as np
import sys

model = tf.keras.models.load_model('recyclable_vs_nonrecyclable_cnn.h5')

img_path = sys.argv[1]
img = cv2.imread(img_path)
img = cv2.resize(img, (128,128))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]

if pred >= 0.5:
    print("Recyclable")
else:
    print("Non-Recyclable")
