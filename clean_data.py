# Import the necessary packages

import pandas as pd
import argparse
import cv2
from utils import classify, describe, train

# Load Random Forest Classifier
model = train()

# Set up the argument parser

ap = argparse.ArgumentParser()

# Define arguments

ap.add_argument("-f", "--file", required=True, help="Path to the pandas dataframe with Instagram metadata.")

# Parse arguments

args = vars(ap.parse_args())

# Assign arguments to variables

df_file = args["file"]

# Load dataframe

df = pd.read_pickle(df_file)

print "*** Loading Instagram metadata from {} ...".format(df_file)

# Loop over images in the dataframe
for index, row in df.iterrows():
    # Define path
    ipath = "test_output/" + row['Filename']

    # Load image
    image = cv2.imread(ipath)

    # Extract features
    features = describe(image)

    # Classify image
    prediction = classify(features, model)

    print "*** Classifying {} ... prediction: {}".format(ipath, prediction)

    # Take action based on prediction
    if prediction == 'photo':
        cv2.imwrite("test_output/photos/%s" % row['Filename'], image)
    if prediction == 'other':
        df = df[df.index != index]
        cv2.imwrite("test_output/others/%s" % row['Filename'], image)

# Reset dataframe index
df = df.reset_index(drop=True)
df.index += 1

print "*** Updating dataframe index ..."

# Pickle cleaned data
new_df_file = "test_output/cleaned.pkl"
print "*** Saving the cleaned dataframe into {}".format(new_df_file)

df.to_pickle(new_df_file)
print "*** ... Done."
