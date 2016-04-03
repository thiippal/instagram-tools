# Import the necessary packages and Instagram API credentials

from instagram.client import InstagramAPI
import requests
import cv2
import numpy as np
import argparse
import pandas as pd
from api_credentials import access_token, client_secret, client_id
from utils import classify, describe, train
from datetime import datetime
from progress.bar import Bar

# Authenticate with Instagram API

api = InstagramAPI(access_token=access_token, client_id=client_id, client_secret=client_secret)

# Define functions

def convert_utc(datets):
    """Converts a Python datetime into a Unix timestamp.
    :param datets: Python datetime
    :return: Unix timestamp
    """
    unix_ts = int((datets - datetime(1970, 1, 1)).total_seconds())
    return unix_ts

def get_stamp(timestamp):
    """Retrieves timestamp for the 100th image following the input timestamp, used to retrieve the next 100 images.
    :param timestamp: Unix timestamp
    :return: Unix timestamp
    """
    media = api.media_search(lat=latitude, lng=longitude, distance=distance, count=100, max_timestamp=timestamp)
    new_max_timestamp = convert_utc(media[-1].created_time)
    return new_max_timestamp

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
# python download_location.py -lat 60.169444 -lon 24.9525 -d 100 -n 180 -r thumbnail

# Parse arguments

args = vars(ap.parse_args())

# Assign arguments to variables

latitude = float(args["latitude"])
longitude = float(args["longitude"])
distance = int(args["distance"])
count = int(args["number"])
size = args["resolution"]
clean = args["clean"]

print "*** Downloading {} photos taken at {}, {} in {} size ...".format(count, latitude, longitude, size)

if clean:
    print "*** Attempting to remove memes, screenshots and other clutter ..."
    model = train()


def download_location(lat, lng, dist, number, resolution):
    # Set up a list for metadata
    metadata = []

    if number <= 100:
        # Retrieve the data directly without pagination
        photos = api.media_search(lat=lat, lng=lng, distance=dist, count=number)

    if number >= 100:
        # Set up a list for photos
        photos = []

        # Get current time
        init_timestamp = convert_utc(datetime.now())

        # Calculate the number of loops required to fetch the required number of photos
        loops = int(count / float(100))

        # Initialize progress bar
        lbar = Bar('*** Retrieving timestamps', max=loops)

        # Get the maximum timestamp for each batch of 100 photos
        timestamps = [init_timestamp]
        for stamp in range(0, loops):
            timestamps.append(get_stamp(timestamps[-1]))
            # Update progress bar
            lbar.next()

        # Fetch the images using timestamps
        for timestamp in timestamps:
            more_photos = api.media_search(lat=lat, lng=lng, distance=dist, count=100, max_timestamp=timestamp)
            photos.extend(more_photos)

        # Finish progress bar
        lbar.finish()

    # Check the number of retrieved photos
    if len(photos) <= number:
        retnum = len(photos)
        print "*** Not enough photos available at this location! Downloading only {} photos ...".format(retnum)
    else:
        retnum = number

    # Initialize progress bar
    dlbar = Bar('*** Downloading photos   ', max=retnum)

    # Download images
    for m in range(0, retnum):
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

            tags = []
            for tag in photo.tags:
                tags.append(tag.name)

            # Check response
            response = requests.get(imurl)
            if response.status_code != 200:
                print 'Aborting ... error {} (}'.format(response.status_code, response.reason)

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

            # Update progress bar
            dlbar.next()

    # Finish progress bar
    dlbar.finish()

    print "*** Retrieved a total of {} photos ... ".format(len(metadata))

    dataframe = pd.DataFrame(metadata)
    dataframe.index += 1  # Set dataframe index to start from 1

    return dataframe


df = download_location(latitude, longitude, distance, count, size)

# Define a filename for dataframe
df_file = "test_output/{}.pkl".format(str(latitude) + '_' + str(longitude))
print "*** Saving metadata into {}".format(df_file)

# Check API rate
print "*** API rate: {}/{} remaining ...".format(api.x_ratelimit_remaining, api.x_ratelimit)

# Pickle dataframe
df.to_pickle(df_file)
print "*** ... Done."
