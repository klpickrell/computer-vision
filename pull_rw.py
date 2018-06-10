#!/usr/bin/env python

import requests
import os

from PIL import Image
from pymongo import MongoClient
from cStringIO import StringIO

STANDARD_SIZE = (495, 535)
output_dir = 'output'

def _main():
    client = MongoClient( 'localhost' )
    collection = client['real_weddings']['photos']
    destination_collection = client['real_weddings_qa']['photo_features']
    completed_ids = set([ item['_id'] for item in destination_collection.find({},{'_id':1}) ])
    all_content = { item['_id'] : { 'url' : item['image_url'], 'terms' : item['terms'] } 
                    for item in collection.find( {}, { 'image_url' : 1, 'terms' : 1 } ) }
    needed_ids = set( all_content.keys() ).difference( completed_ids )
    print( 'pulling content for {} images'.format(len(needed_ids)) )
    for content_id in needed_ids:
        url = all_content[content_id]['url']
        response = requests.get( url )
        if response.status_code != 200:
            print( 'failed to get image for {}'.format( url ) )
            continue

        image = Image.open( StringIO( response.content ) ).convert('RGB')
        image = image.resize(STANDARD_SIZE)

        output_file = os.path.join( os.getcwd(), output_dir, '{}.jpeg'.format(content_id) )
        image.save( output_file )

        memo = { '_id' : content_id, 
                 'terms' : all_content[content_id]['terms'], 
                 'url' : url,
                 'filename' : output_file }

        destination_collection.insert( memo )
        print( 'downloaded {}, converted {}, and inserted {}'.format(url, output_file, content_id) )
#        pixels = list(image.getdata())
#        image = np.array( map(list,pixels), dtype=np.int16)
        
        
        
        
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit( _main() )
