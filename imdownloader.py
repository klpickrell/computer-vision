#!/usr/bin/env python
import numpy as np
import requests
import os

from PIL import Image
from csv import DictReader
from cStringIO import StringIO
from tqdm import tqdm

from pymongo import MongoClient

def _main():

    images = [ line for line in DictReader( open('photos.csv','rU') ) ]
    for image in tqdm(images):
        filename = 'raw/{}.jpg'.format(image['id'] )
        if os.path.exists( filename ):
            print( '{} skipped'.format(filename) )
            continue

        try:
            response = requests.get(image['image_url'])
            if response.status_code != 200:
                print( 'failed to retrieve {}, skipping'.format(image['image_url']) )
                continue
            image_data = StringIO( response.content )
            im = Image.open(image_data)
            im.thumbnail( (im.size[0]*0.5,im.size[1]*0.5) )
            im.save(filename)

        except Exception, ex:
            print( 'exception {} on {}'.format(ex,image['image_url']) )
            continue
        
        image['filename'] = filename
        image['image'] = im

    database_name = 'recommendation'
    collection_name = 'tblphotos'
    print( 'done downloading, inserting {} images into {}.{}'.format(len(images),database_name,collection_name) )
    client = MongoClient()
    tblphotos = client[database_name][collection_name]
    for image in images:
        image['_id'] = image['id']
        if 'image' in image:
            del image['image']
    tblphotos.insert_many( images )
    print( 'done' )
    return 0


if __name__ == '__main__':
    import sys
    sys.exit( _main() )
