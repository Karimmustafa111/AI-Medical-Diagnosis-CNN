import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Data Preprocessing
# ==========================================
DATA_DIR = "dataset"
CATEGORIES = ["no", "yes"]
IMG_SIZE = 64

data = []
labels = []

print("1. Download Image and Transfering it to Digits is Loading... ⌛")

for category in CATEGORIES:
  path = os.path.join(DATA_DIR, category) # "dataset/no"
  class_num = CATEGORIES.index(category) # 0 for no, 1 for yes

  for img_name in os.listdir(path): # show all images in the file (path)
    try:

      img_path = os.path.join(path, img_name) # create "dataset/no/img1"
      img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

      new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

      data.append(new_array)
      labels.append(class_num)

    except Exception as e:
      pass

X = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X / 255.0
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data is Loaded! Number of Train Images: {len(X_train)}, Test Images: {len(X_test)}")

# =========================================
# 2. CNN Architecture
# =========================================
print("2. Artifital Brain is Loading (CNN)... 🧠")

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:])) # (64,64,1)
model.add(MaxPooling2D(pool_size=(2, 2))) # (32,32)

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# =========================================
# 3. Training
# =========================================
print("3. Training is Start (Model is Study Now)... 📚")

model.fit(X_train, y_train, batch_size=10, epochs=5, validation_split=0.1)

# =========================================
# 4. Evaluation
# =========================================
print("4. Exam (moment of truth)... 📑")
loss, accuracy = model.evaluate(X_test, y_test)

print(f"\n====================")
print(f"Model IQ: {accuracy * 100:.2f}%")
print(f"====================\n")

model.save('brain_tumor_detector.h5')
print("Model is Saved Successfully! 📰")