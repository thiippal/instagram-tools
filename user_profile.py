# Import the necessary packages and Instagram API credentials

from instagram.client import InstagramAPI
from api_credentials import access_token, client_secret, client_id
import pandas as pd
import argparse
from collections import Counter

# Authenticate with Instagram API

api = InstagramAPI(access_token=access_token, client_id=client_id, client_secret=client_secret)

# Define functions

def get_user_id(usernames):
    # Set up a list for identifiers
    ids = []
    # Loop over the list of usernames
    for uname in usernames:
        # Search for the username
        names = api.user_search(q=uname)
        # Loop over the returned names
        for n in names:
            # Append the exact match to the list of ids
            if uname == n.username:
                ids.append(n.id)
    # Return the list of identifiers
    return ids

def check_location(lat, lng):
    # Geographical latitude / longitude bounding box for Finland
    fi_nlat = 70.092293
    fi_slat = 59.705440
    fi_wlon = 20.547411
    fi_elon = 31.587100

    # Check if latitude and longitude fall within the bounding box
    if fi_slat <= lat <= fi_nlat and fi_wlon <= lng <= fi_elon:
        fin = True
    else:
        fin = False
    return fin

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

# Set up a list for usernames
users = []

# Retrieve user information from the dataframe
for index, row in df.iterrows():
    # Retrieve username and append it to the list
    user = row['User']
    users.append(user)

# Retrieve user identifiers
user_ids = get_user_id(users)

# Add user identifiers to the dataframe
df['User ID'] = user_ids

# Set up a dictionary for the user profiles
uprof = []

# Loop over unique users
for u in user_ids:
    # Retrieve 20 media
    media, more = api.user_recent_media(user_id=u, count=20)

    # Set up a list for the location vector
    lvec = []

    # Check if user has uploaded more than 20 photos ...
    if len(media) == 20:
        # Check if the photos have location information
        for m in media:
            if m.type == 'image':
                try:
                    location = m.location.name
                    latitude = m.location.point.latitude
                    longitude = m.location.point.longitude
                except AttributeError:
                    pass

                # Check the coordinates
                lvec.append(check_location(latitude, longitude))

        # Count instances in location vector
        count = Counter(lvec)

    # Assign the user to a class
    if count.most_common()[0][0]:
        uprof.append("local")
    if not count.most_common()[0][0]:
        uprof.append("visitor")

# Append the user profiles to the dataframe
df['Type'] = uprof

# Pickle the updated data
new_df_file = "test_output/profiles.pkl"
print "*** Saving the updated dataframe into {}".format(new_df_file)

df.to_pickle(new_df_file)
print "*** ... Done."

print df