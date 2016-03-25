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

# Extract features
def extract():

    imagepaths = sorted(paths.list_images('training_data/'))
    labels = []
    data = []

    for imagepath in imagepaths:
        label = imagepath[imagepath.rfind('/') + 1:].split('-')[0]
        image = cv2.imread(imagepath)

        features = describe(image)
        labels.append(label)
        data.append(features)

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
def train():

    # Load features
    datafile = "models/features.pkl"
    td_file = open(datafile, 'r')
    data = pickle.load(td_file)

    # Load labels
    labelfile = "models/labels.pkl"
    ld_file = open(labelfile, 'r')
    labels = pickle.load(ld_file)

    # Split data for training and testing
    (traindata, testdata, trainlabels, testlabels) = train_test_split(np.array(data), np.array(labels), test_size=0.25, random_state=42)

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(traindata, trainlabels)

    # Evaluate the classifier

    predictions = model.predict(testdata)
    print(classification_report(predictions, testlabels, target_names=testlabels))

    # Return trained classifier
    return model

# Classify image using model
def classify(features, model):
    # Reshape the features
    features = features.reshape(1, -1)

    # Classify input image
    prediction = model.predict(features)[0]

    # Return classification
    return prediction
