import os
import ssl
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
import tensorflow as tf 
from tensorflow import keras 
from keras_facenet import FaceNet

embedder = FaceNet()

dataset_path = "C:\Face_Attendance_Project\Dataset"
embeddings = []
labels = []

for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)
    if not os.path.isdir(folder):
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (160, 160))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        emb = embedder.embeddings([img])[0]
        embeddings.append(emb)
        labels.append(label)

np.save("embeddings.npy", embeddings)
np.save("labels.npy", labels)

print("✅ FaceNet embeddings created successfully")
print("💾 Embeddings saved as 'embeddings.npy' and 'labels.npy'")