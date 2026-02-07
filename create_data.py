import cv2
import numpy as np
import os

if not os.path.exists("dataset"):
  os.makedirs("dataset/yes") # if not exists, create new dataset (folder)
  os.makedirs("dataset/no") # and put yes & no (files) in it.

print("Making Data of phantom rays is Loading... 🧠")

for i in range(100):
  img = np.zeros((128, 128), dtype=np.uint8)

  center = (np.random.randint(30, 90), np.random.randint(30, 90))
  cv2.circle(img, center, 15, (255), -1)

  cv2.imwrite(f"dataset/yes/tumor_{i}.jpg", img)

for i in range(100):
  img = np.zeros((128, 128), dtype=np.uint8)

  noise = np.random.randint(0, 50, (128, 128))
  img = img + noise
  cv2.imwrite(f"dataset/no/healthy_{i}.jpg", img)

print("200 ray images is created successfully! ✅")