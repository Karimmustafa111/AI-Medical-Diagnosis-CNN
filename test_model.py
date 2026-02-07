import cv2
import numpy as np
import tensorflow as tf
import os
import random

print("Artifitial Brain is Loading... 🧠")
model = tf.keras.models.load_model('brain_tumor_detector.h5')

def prepare_image(filepath):
  IMG_SIZE = 64
  img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # Read Image and Turn it to Gray
  new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0 # -1 (any number of images)

random_filename = random.choice(os.listdir("dataset/yes"))
image_path = f"dataset/yes/{random_filename}"

print(f"Check Image: {random_filename} ... ⌛")
prediction = model.predict([prepare_image(image_path)]) # [[1.0]]

print("------------------------------------------------")
print(f"Result of Checking (Confidence): {prediction[0][0]}")

if prediction[0][0] > 0.5:
  print("Tumor Detected 🚨")
else:
  print("Healthy ✅")
print("------------------------------------------------")

img = cv2.imread(image_path)
cv2.imshow("MRI Scan", img)
cv2.waitKey(0)
cv2.destroyAllWindows()