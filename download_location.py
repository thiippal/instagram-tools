# Import the necessary packages and Instagram API credentials

from instagram.client import InstagramAPI
import requests
import time
import cv2
import numpy as np
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
ap.add_argument("-r", "--resolution", required=True, help="Define the resolution of the image to be retrieved.")

# Test string for command line query
# python download_location.py -lat 60.169444 -lon 24.9525 -d 20 -c 40 -r thumbnail

# Parse arguments

args = vars(ap.parse_args())

# Assign arguments to variables

latitude = float(args["latitude"])
longitude = float(args["longitude"])
distance = int(args["distance"])
count = int(args["count"])
size = args["resolution"]

def download_location(lat, lng, dist, number, resolution):

    # Set up a list for metadata
    metadata = []

    # Retrieve data
    photos = api.media_search(lat=lat, lng=lng, distance=dist, count=number)

    # Download images
    for m in range(0, number):
            photo = photos[m]
            if photo.type == 'image':  # Limit the query to images
                identifier = photo.id  # Unique image identifier
                user = photo.user.username  # Instagram username
                imurl = photo.images["%s" % resolution].url  # Values: thumbnail, low_resolution, standard_resolution

                print "*** Downloading", "#%s" % str(m + 1), identifier, "taken by", user, "at", lat, lng

                tags = []
                for tag in photo.tags:
                    tags.append(tag.name)

                response = requests.get(imurl)

                # Decode response
                image = np.asarray(bytearray(response.content), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                # Save image
                filename = str(lat) + '_' + str(lng) + '-' + str(identifier) + '.png'
                cv2.imwrite("test_output/%s" % filename, image)

download_location(latitude, longitude, distance, count, size)
