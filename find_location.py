# Import the necessary packages and Instagram API credentials

from instagram.client import InstagramAPI
import argparse
import pandas as pd
from api_credentials import access_token, client_secret, client_id

# Authenticate with Instagram API

api = InstagramAPI(access_token=access_token, client_id=client_id, client_secret=client_secret)

# Set up the argument parser

ap = argparse.ArgumentParser()

# Define arguments

ap.add_argument("-lat", "--latitude", required=True, help="Define the latitude of the location.")
ap.add_argument("-lon", "--longitude", required=True, help="Define the longitude of the location.")
ap.add_argument("-d", "--distance", required=True, help="Define a radius for the defined point in metres.")
ap.add_argument("-c", "--count", required=True, help="Define how many locations are retrieved.")

# Test string for command line query
# python find_location.py -lat 60.169444 -lon 24.9525 -d 20 -c 33

# Parse the arguments

args = vars(ap.parse_args())

# Assign the arguments to variables

latitude = float(args["latitude"])
longitude = float(args["longitude"])
distance = int(args["distance"])
count = int(args["count"])

# Retrieve the locations

location_search = api.location_search(lat=latitude, lng=longitude, distance=distance, count=count)

# Set up a list for the locations

locations = []

# Loop over the locations

for location in location_search:
    locations.append({'id': location.id,
                      'name': location.name,
                      'coordinates': location.point})

# Pass the locations into a pandas dataframe

df = pd.DataFrame(locations)

df.index += 1  # Set index to start from 1

# Define a filename for saving the data

filename = str(latitude) + "_" + str(longitude) + "_" + str(distance) + ".pkl"

# Pickle the dataframe

df.to_pickle(filename)

# TODO Find a way to implement some sort of grid search to overcome the 33 location limit
