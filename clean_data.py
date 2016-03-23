# Import the necessary packages

import pandas as pd
import argparse
import cv2
from utils import classify, describe, load_model

# Load model for Random Forest Classifier
model = load_model()

# Set up the argument parser

ap = argparse.ArgumentParser()

# Define arguments

ap.add_argument("-f", "--file", required=True, help="Path to dataframe.")

# Parse arguments

args = vars(ap.parse_args())

# Assign arguments to variables

df_file = args["file"]

# Load dataframe

df = pd.read_pickle(df_file)

# Loop over images in the dataframe
for index, row in df.iterrows():
    # Define path
    ipath = "test_output/" + row['Filename']

    # Load image
    image = cv2.imread(ipath)

    # Extract features
    features = describe(image)

    # Classify image
    prediction = classify(features.reshape(1, -1), model)

    # Take action based on prediction
    if prediction == 'photo':
        continue
    elif prediction == 'meme':
        df = df[df.index != index]
        cv2.imwrite("test_output/memes/%s" % row['Filename'], image)

# Reset dataframe index
df = df.reset_index(drop=True)
df.index += 1

# Pickle cleaned data
new_df_file = "test_output/cleaned.pkl"
df.to_pickle(new_df_file)
