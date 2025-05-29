from imageio import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Parameters for LBP
radius = 3
n_points = 8 * radius

# Load images and extract LBP features
def extract_lbp_features(folder_path, label):
    features = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = imread(img_path)
        img_resized = resize(img, (150, 150, 3))
        gray = rgb2gray(img_resized)
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        features.append(hist)
        labels.append(label)
    return features, labels

# Paths to cat and dog image directories
cat_path = 'DS/cats'
dog_path = 'DS/dogs'

cat_features, cat_labels = extract_lbp_features(cat_path, 0)
dog_features, dog_labels = extract_lbp_features(dog_path, 1)

# Combine data and train model
X = cat_features + dog_features
y = cat_labels + dog_labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate model
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model accuracy:", accuracy)

# Predict new image
def predict_image(path):
    img = imread(path)
    img_resized = resize(img, (150, 150, 3))
    gray = rgb2gray(img_resized)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    prediction = model.predict([hist])
    print("Predicted class:", "Cat" if prediction[0] == 0 else "Dog")

# Example usage
# predict_image("DS/cats/cat.30.jpg")
