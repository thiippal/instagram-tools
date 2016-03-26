# Import the necessary packages and Instagram API credentials

from instagram.client import InstagramAPI
import requests
import cv2
import numpy as np
import argparse
import pandas as pd
from api_credentials import access_token, client_secret, client_id
from utils import classify, describe, train

# Authenticate with Instagram API

api = InstagramAPI(access_token=access_token, client_id=client_id, client_secret=client_secret)

# Set up the argument parser

ap = argparse.ArgumentParser()

# Define arguments

ap.add_argument("-lat", "--latitude", required=True, help="Define the latitude of the location.")
ap.add_argument("-lon", "--longitude", required=True, help="Define the longitude of the location.")
ap.add_argument("-d", "--distance", required=True, help="Define a radius for the defined point in metres.")
ap.add_argument("-n", "--number", required=True, help="Define how many locations are retrieved.")
ap.add_argument("-r", "--resolution", required=True, help="Define the resolution of the image to be retrieved.")
ap.add_argument('-c', "--clean", action='store_true', help="Remove memes, screenshots and other clutter from the data.")

# Test string for command line query
# python download_location.py -lat 60.169444 -lon 24.9525 -d 20 -c 40 -r thumbnail

# Parse arguments

args = vars(ap.parse_args())

# Assign arguments to variables

latitude = float(args["latitude"])
longitude = float(args["longitude"])
distance = int(args["distance"])
count = int(args["number"])
size = args["resolution"]
clean = args["clean"]

print "*** Downloading {} images at {}, {} in {} size ...".format(count, latitude, longitude, size)

if clean:
    print "*** Will attempt to clean the data from memes, screenshots and other clutter ..."
    model = train()


def download_location(lat, lng, dist, number, resolution):
    # Set up a list for metadata
    metadata = []

    # Retrieve data
    photos = api.media_search(lat=lat, lng=lng, distance=dist, count=number)
    # TODO How to retrieve over 100 images

    # Download images
    for m in range(0, number):
            photo = photos[m]
            if photo.type == 'image':  # Limit the query to images
                identifier = photo.id  # Unique image identifier
                user = photo.user.username  # Instagram username
                imurl = photo.images["%s" % resolution].url  # Resolution: thumbnail, low_resolution, standard_resolution
                created = photo.created_time
                caption = photo.caption

                # Check if location data is available.
                try:
                    location = photo.location
                except AttributeError:
                    location = "Location: N/A"
                    pass

                print "*** Downloading", "#%s" % str(m + 1), identifier, "taken by", user, "at", location.name, (lat, lng)

                tags = []
                for tag in photo.tags:
                    tags.append(tag.name)

                response = requests.get(imurl)

                # Decode response
                image = np.asarray(bytearray(response.content), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                save = True

                # Describe and classify the image if requested
                if clean:
                    # Extract features
                    features = describe(image)

                    # Classify image
                    prediction = classify(features, model)

                    if prediction == 'photo':
                        pass
                    if prediction == 'other':
                        save = False

                if save:
                    # Save image
                    filename = str(lat) + '_' + str(lng) + '-' + str(identifier) + '.png'
                    cv2.imwrite("test_output/%s" % filename, image)

                    # Store metadata
                    metadata.append({'Identifier': identifier,
                                     'User': user,
                                     'URL': imurl,
                                     'Coordinates': (latitude, longitude),
                                     'Location': location,
                                     'Tags': ' '.join(tags),
                                     'Created': created,
                                     'Filename': filename,
                                     'Caption': caption})

                else:
                    pass

            print created

    print "*** Retrieved a total of {} images ... ".format(len(metadata))

    dataframe = pd.DataFrame(metadata)
    dataframe.index += 1  # Set dataframe index to start from 1

    return dataframe


df = download_location(latitude, longitude, distance, count, size)

# Define a filename for dataframe

df_file = "test_output/{}.pkl".format(str(latitude) + '_' + str(longitude))

print "*** Saving metadata into {}".format(df_file)

# Pickle dataframe
df.to_pickle(df_file)

print "*** ... Done."
