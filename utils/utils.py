# Import the necessary packages

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imutils import paths
import numpy as np
import mahotas
import cv2
import pickle

# Describe image using colour statistics and Haralick texture

def describe_haralick_stats(image):
    """
	Describe multiple regions of the image using colour statistics and Haralick texture.

	Args:
		image: Input image.

	Returns:
		A concatenated numpy vector consisting of colour statistics and Haralick texture descriptor.
	"""
    # Get input image height and width
    h, w = image.shape[:2]

    # Calculate segments for vertical direction
    top_end_y = int(h * 0.2)
    bottom_start_y = h - int(h * 0.2)

    # Calculate segments for horizontal direction
    left_end_x = int(w * 0.2)
    right_start_x = w - int(h * 0.2)

    # Define coordinates for the different segments
    zones = [(0, top_end_y, 0, w), (top_end_y, bottom_start_y, 0, w), (bottom_start_y, h, 0, w), (0, h, 0, left_end_x),
             (0, h, left_end_x, right_start_x), (0, h, right_start_x, w)]

    # A list to hold the feature vector
    vector = []

    # Loop over the zones
    for (starty, endy, startx, endx) in zones:
        # Calculate colour statistics
        (means, stds) = cv2.meanStdDev(cv2.cvtColor(image[startx:endx, starty:endy], cv2.COLOR_BGR2HSV))
        colorstats = np.concatenate([means, stds]).flatten()

        # Extract Haralick texture
        gray = cv2.cvtColor(image[starty:endy, startx:endx], cv2.COLOR_BGR2GRAY)
        haralick = mahotas.features.haralick(gray).mean(axis=0)

        # Combine the features
        feats = np.concatenate([colorstats, haralick])

        # Extend the feature vector
        vector.extend(feats)

    # Return the feature vector as a numpy array
    return np.asarray(vector)

# Extract features
def extract():

    imagepaths = sorted(paths.list_images('training_data/'))
    labels = []
    data = []

    for imagepath in imagepaths:
        label = imagepath[imagepath.rfind('/') + 1:].split('-')[0]
        image = cv2.imread(imagepath)

        features = describe_haralick_stats(image)
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
    (traindata, testdata, trainlabels, testlabels) = train_test_split(np.array(data),
                                                                      np.array(labels),
                                                                      test_size=0.25,
                                                                      random_state=42)

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(traindata, trainlabels)

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
