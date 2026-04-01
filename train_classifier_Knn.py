import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load embeddings
X = np.load("C:\\Face_Attendance_Project\\embeddings.npy")
y = np.load("C:\\Face_Attendance_Project\\labels.npy")

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(X, y)

# Save model
joblib.dump(knn, "face_classifier.pkl")

print("✅ KNN model trained and saved")
print("💾 Model saved as 'face_classifier.pkl'")