'''
This script downloads TACO's images from Flickr given an annotation json file
'''

import os.path
import argparse
import json
from PIL import Image
import requests
from io import BytesIO
import sys

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_path', required=False, default= './data/annotations.json', help='Path to annotations')
args = parser.parse_args()

dataset_dir = os.path.dirname(args.dataset_path)


# Load annotations
with open(args.dataset_path, 'r') as f:
    annotations = json.loads(f.read())

    nr_images = len(annotations['images'])
    for i in range(nr_images):

        image = annotations['images'][i]

        file_name = image['file_name']
        url_original = image['flickr_url']
        url_resized = image['flickr_640_url']

        file_path = os.path.join(dataset_dir, file_name)

        # Create subdir if necessary
        subdir = os.path.dirname(file_path)
        if not os.path.isdir(subdir):
            os.mkdir(subdir)

        if not os.path.isfile(file_path):
            try:
                # Load and Save Image
                response = requests.get(url_original, timeout=10)
                img = Image.open(BytesIO(response.content))
                if hasattr(img, '_getexif') and img._getexif():
                    img.save(file_path, exif=img.info["exif"])
                else:
                    img.save(file_path)
            except Exception as e:
                pass

        # Show loading bar
        bar_size = 30
        x = int(bar_size * i / nr_images)
        sys.stdout.write("%s[%s%s] - %i/%i\r" % ('Loading: ', "=" * x, "." * (bar_size - x), i, nr_images))
        sys.stdout.flush()
        i+=1

    sys.stdout.write('Finished\n')
