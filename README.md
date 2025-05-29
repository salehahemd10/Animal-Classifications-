# Cat vs Dog Image Classification using LBP and Gaussian Naive Bayes

This project demonstrates a basic image classification task to distinguish between cats and dogs using Local Binary Pattern (LBP) as a feature extractor and Gaussian Naive Bayes (GNB) as the classifier.

## Table of Contents

- Overview
- Tech Stack
- How LBP Works
- Dataset Structure
- Installation and Setup
- How to Run
- Model Performance
- Results
- Future Work
- Contact

## Overview

This project shows how traditional machine learning methods can be used for image classification. Instead of deep learning, we use handcrafted features (LBP) extracted from images and feed them into a Gaussian Naive Bayes model to classify images as either a cat or a dog.

## Tech Stack

- Python
- NumPy
- scikit-learn
- matplotlib
- scikit-image
- imageio

## How LBP Works

Local Binary Pattern (LBP) is a simple texture descriptor. It works by comparing each pixel in a grayscale image with its surrounding neighbors:

1. For each pixel, compare it with 8 surrounding pixels.
2. If the surrounding pixel is greater than or equal to the center pixel, assign 1; else assign 0.
3. This results in an 8-bit binary number.
4. Convert that binary number to decimal and assign it to the center pixel.
5. Compute a histogram of all values in the LBP image.

This histogram is then used as the feature vector for the image.

## Dataset Structure

The dataset directory should follow this structure:

```
DS/
├── cats/
│   ├── cat.1.jpg
│   ├── cat.2.jpg
│   └── ...
├── dogs/
│   ├── dog.1.jpg
│   ├── dog.2.jpg
│   └── ...
```

Images are resized to 150x150 pixels before feature extraction.

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cat-vs-dog-lbp.git
cd cat-vs-dog-lbp
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
numpy
scikit-learn
matplotlib
scikit-image
imageio
```

## How to Run

### Training

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Assume X and y are your LBP features and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = GaussianNB()
model.fit(X_train, y_train)
```

### Predicting on a New Image

```python
from imageio import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
import numpy as np

radius = 3
n_points = 8 * radius

img = imread("path/to/image.jpg")
img_resized = resize(img, (150, 150, 3))
gray = rgb2gray(img_resized)
lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
hist = hist.astype("float")
hist /= (hist.sum() + 1e-7)

prediction = model.predict([hist])
print("Predicted class:", "Cat" if prediction[0] == 0 else "Dog")
```

### Evaluating Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, model.predict(X_test))
print("Accuracy:", accuracy)
```

## Model Performance

| Model              | Accuracy      |
|--------------------|---------------|
| Gaussian Naive Bayes | Highest accuracy |
| K-Nearest Neighbors | Moderate       |
| SVM                 | Lower          |

## Results

The model successfully classifies unseen images into cat or dog classes based on LBP texture features. Visualization of predictions is done using matplotlib.

## Future Work

- Try other handcrafted features (HOG, SIFT)
- Compare with CNN-based deep learning models
- Add a simple GUI for prediction
- Deploy as a web application

## Contact

Created by [Your Name] - feel free to reach out or contribute to the project.

