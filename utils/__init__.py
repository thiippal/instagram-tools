# TODO add try statements

# Import the necessary packages and Instagram API credentials

# API requests
import requests
from datetime import datetime
from collections import Counter
import progressbar

# Argument parser
import argparse

# Instagram API
from instagram.client import InstagramAPI
from api_credentials import access_token, client_secret, client_id

# Computer vision
import cv2

# Data processing
import numpy as np
import pandas as pd

# Support functions
from utils import classify, describe, train
