# Import the necessary packages

from __future__ import print_function
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from imutils import paths
import numpy as np
import mahotas
import cv2
import pickle


# Describe image using colour statistics and Haralick texture
def describe(image):
    (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    colorstats = np.concatenate([means, stds]).flatten()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    haralick = mahotas.features.haralick(gray).mean(axis=0)

    return np.hstack([colorstats, haralick])

# Extract features and train a Random Forest Classifier
def train_rfc():

    imagepaths = sorted(paths.list_images('training_data/'))
    labels = []
    data = []

    for imagepath in imagepaths:
        label = imagepath[imagepath.rfind('/') + 1:].split('-')[0]
        image = cv2.imread(imagepath)

        features = describe(image)
        labels.append(label)
        data.append(features)

    (traindata, testdata, trainlabels, testlabels) = train_test_split(np.array(data), np.array(labels), test_size=0.25, random_state=42)

    model = RandomForestClassifier(n_estimators=20, random_state=42)

    model.fit(traindata, trainlabels)

    predictions = model.predict(testdata)
    print(classification_report(testlabels, predictions))

    datafile = "models/features.pkl"
    td_file = open(datafile, 'wb')
    pickle.dump(data, td_file)

    td_file.close()

    labelfile = "models/labels.pkl"
    ld_file = open(labelfile, 'wb')
    pickle.dump(labels, ld_file)

    ld_file.close()

    return

# Train a Random Forest Classifier using pre-extracted data
def load_model():
    # Load features
    datafile = "models/features.pkl"
    td_file = open(datafile, 'r')
    data = pickle.load(td_file)

    # Load labels
    labelfile = "models/labels.pkl"
    ld_file = open(labelfile, 'r')
    labels = pickle.load(ld_file)

    # Split data for training and testing
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), np.array(labels), test_size=0.25, random_state=42)

    # Set up Random Forest Classifier
    model = RandomForestClassifier(n_estimators=20, random_state=42)

    # Train the classifier
    model.fit(trainData, trainLabels)

    # Return the trained classifier
    return model

# Classify image using model
def classify(features, model):
    # Classify input image
    prediction = model.predict(features)[0]

    # Return classification
    return prediction
