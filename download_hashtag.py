# Import the necessary packages and Instagram API credentials
from utils import *

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
    print "*** Attempting to remove memes, screenshots and other clutter ..."
    model = train()

# Define a function for downloading images

def download_hashtag(number, tag, resolution):
    # Set up a list for metadata
    metadata = []

    # Retrieve data
    photos, following = api.tag_recent_media(count=number, tag_name=tag)
    _, max_tag = following.split("max_tag_id=")
    max_tag = str(max_tag)

    # Calculate the number of pages: the first page has 33 images, the following 20
    loops = int((number - 13) / float(20))

    # Initialize progress bar if multiple loops are necessary
    if loops >= 1:
        # Initialize progress bar
        pbar_widgets = ["*** Looping over pages:", " ", progressbar.Percentage(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=loops, widgets=pbar_widgets)
        pbar.start()

        # Counter for while loop
        counter = 1

        # Loop over paginated data
        while following and len(photos) <= number:
            more_photos, following = api.tag_recent_media(tag_name=tag, max_tag_id=max_tag)
            _, max_tag = following.split("max_tag_id=")
            max_tag = str(max_tag)
            photos.extend(more_photos)

            # Update counter
            counter += 1

            # Update progess bar
            if counter <= loops:
                pbar.update(counter)

        # Finish progress bar
        pbar.finish()

    # Check the available number of photos
    if len(photos) <= number:
        retnum = len(photos)
    else:
        retnum = number

    # Initialize progress bar
    dlbar_widgets = ["*** Downloading photos:", " ", progressbar.Percentage(), " ", progressbar.ETA()]
    dlbar = progressbar.ProgressBar(maxval=retnum, widgets=dlbar_widgets)
    dlbar.start()

    # Download images
    for m in range(0, retnum):
        photo = photos[m]
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

            tags = []
            for tag in photo.tags:
                tags.append(tag.name)

            # Check response
            response = requests.get(imurl)
            if response.status_code != 200:
                print 'Aborting ... response {} (}'.format(response.status_code, response.reason)

            # Decode response
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            save = True

            # Describe and classify the image if requested
            if clean and size is 'thumbnail':
                # Extract features
                features = describe_haralick_stats(image)

                # Classify image
                prediction = classify(features, model)

                if prediction == 'photo':
                    pass
                if prediction == 'other':
                    save = False

            if clean and size is not 'thumbnail':
                # Make a copy of the image
                image_rz = image.copy()

                # Resize image
                resized = imutils.resize(image_rz, width=150)

                # Extract features
                features = describe_haralick_stats(resized)

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

        # Update progress bar
        dlbar.update(m)

    # Finish progress bar
    dlbar.finish()

    print "*** Retrieved a total of {} images ... ".format(len(metadata))

    dataframe = pd.DataFrame(metadata)
    dataframe.index += 1  # Set dataframe index to start from 1

    return dataframe

df = download_hashtag(count, ht, size)

# Define a filename for dataframe
df_file = "test_output/%s" % str(ht) + ".pkl"
print "*** Saving metadata into {}".format(df_file)

# Check API rate
print "*** API rate: {}/{} remaining ...".format(api.x_ratelimit_remaining, api.x_ratelimit)

# Pickle dataframe
df.to_pickle(df_file)
print "*** ... Done."
