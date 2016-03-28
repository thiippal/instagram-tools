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

ap.add_argument("-ht", "--hashtag", required=True, help="Define the hashtag for retrieving images.")
ap.add_argument("-n", "--number", required=True, help="Define the number of images to be retrieved.")
ap.add_argument("-r", "--resolution", required=True, help="Define the resolution of the image to be retrieved.")
ap.add_argument('-c', "--clean", action='store_true', help="Remove memes, screenshots and other clutter from the data.")

# Parse arguments

args = vars(ap.parse_args())

# Assign arguments to variables

count = int(args["number"])
ht = args["hashtag"]
size = args["resolution"]
clean = args["clean"]

print "*** Downloading {} images for #{} in {} size ...".format(count, ht, size)

if clean:
    print "*** Will attempt to clean the data from memes, screenshots and other clutter ..."
    model = train()

# Define a function for downloading images

def download_hashtag(number, tag, resolution):
    # Set up a list for metadata
    metadata = []

    # Retrieve data
    hashtag, following = api.tag_recent_media(count=number, tag_name=tag)
    _, max_tag = following.split("max_tag_id=")
    max_tag = str(max_tag)

    print "*** Looping over the images to be retrieved (this might take a while) ..."

    # Loop over paginated data
    while following and len(hashtag) <= number:
        more_photos, following = api.tag_recent_media(tag_name=tag, max_tag_id=max_tag)
        _, max_tag = following.split("max_tag_id=")
        max_tag = str(max_tag)
        hashtag.extend(more_photos)

    # TODO Download only images, ie. if media.type == 'image':
    # Download images
    for m in range(0, number):
        photo = hashtag[m]
        if photo.type == 'image':
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

            print "*** Downloading", "#%s" % str(m + 1), identifier, "taken by", user, "at", location

            tags = []
            for tag in photo.tags:
                tags.append(tag.name)

            # Get response and print status
            response = requests.get(imurl)
            print "*** {} {} ...".format(response.status_code, response.reason)

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
                filename = str(ht) + '-' + str(identifier) + '.png'
                cv2.imwrite("test_output/%s" % filename, image)

                # Store metadata
                metadata.append({'Identifier': identifier,
                                 'User': user,
                                 'URL': imurl,
                                 'Location': location,
                                 'Tags': ' '.join(tags),
                                 'Created': created,
                                 'Filename': filename,
                                 'Caption': caption})

            else:
                pass

    print "*** Retrieved a total of {} images ... ".format(len(metadata))

    dataframe = pd.DataFrame(metadata)
    dataframe.index += 1  # Set dataframe index to start from 1

    return dataframe

df = download_hashtag(count, ht, size)

# Define a filename for dataframe

df_file = "test_output/%s" % str(ht) + ".pkl"

print "*** Saving metadata into {}".format(df_file)

# Pickle dataframe
df.to_pickle(df_file)

print "*** ... Done."
