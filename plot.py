#!/usr/bin/env python
import cv2
import os
import pandas as pd
import pylab as pl
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
import re

from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from random import sample, shuffle


STANDARD_SIZE = (300, 167)
THUMBNAIL_SIZE = (256, 256)
#root_dir = 'items/cakesvsbouquets'
root_dir = 'test_images/items/cbsmall'
#root_dir='.'
def img_to_matrix( filename, verbose=False ):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    if verbose==True:
        print "(%s) changing size from %s to %s" % (filename, str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    pixels = list(img.getdata())
    if len(pixels[0]) != 3:
        print( 'skipping {} due to alpha channel presence'.format(filename) )
        return None,None
    data = map(list, pixels)
    data = np.array(data, dtype=np.int16)
    img.thumbnail( THUMBNAIL_SIZE, Image.ANTIALIAS )
    return img,data

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

def _main():
    images = [ os.path.join(root_dir,f) for f in os.listdir(root_dir) if 'jpg' in f ]
    shuffle( images )
    labels = []
    data = []
    image_data = []
    reg = re.compile( r'^([a-zA-Z]+)[0-9]+.*' )
    for image in images:
        print( 'process {}'.format(image) )
        if 'jpg' not in image:
            print( 'skipping {}'.format(image) )
            continue
        idata, img = img_to_matrix(image, True)
        if img is not None:
            image_data.append( np.array(idata) )
            img = flatten_image(img)
            data.append(img)
            image_name = os.path.basename(image)
            mo = reg.match(image_name)
            if mo:
                labels.append(mo.groups()[0])
            else:
                labels.append('unknown')
     
    data = np.array(data, dtype=np.ndarray)
    pca = RandomizedPCA(n_components=3)
    X = pca.fit_transform(data)
    chunk = 100
    train_x = X[chunk:]
    train_y = labels[chunk:]
    test_x = X[:chunk]
    test_y = labels[:chunk]
    knn = KNeighborsClassifier()
    knn.fit( train_x, train_y )
    results = cross_val_score( knn, test_x, test_y, scoring='accuracy', cv=10 )
    print( 'K-nearest neighbors (k=5) accuracy: {}'.format( sum(results)/float(len(results)) ) )
#    pd.crosstab( test_y, knn.predict(test_x), rownames=['Actual'], colnames=['Predicted'] )

    df = pd.DataFrame( {"x": X[:, 0], "y": X[:, 1], "z": X[:,2]}, index=labels )
    colors = { 'bouquet' : (0, 1, 0), 'cake' : (0, 0, 1), 'brights' : (1,0,0), 'neutrals' : (1,1,0) }
    descriptives = { 'bright' : 'Bright', 'neutral' : 'Neutral' }

    mlab.figure()
    for ctype in set(labels):
        mlab.points3d(df.x[ctype], df.y[ctype], df.z[ctype], color=colors[ctype])#, label=descriptives[ctype])
    mlab.axes()
    mlab.show()

#    fig = pl.figure()
#    ax = fig.add_subplot( 111, projection='3d' )
#    for ctype in set(labels):
#        ax.scatter(df.x[ctype], df.y[ctype], df.z[ctype], c=colors[ctype], label=descriptives[ctype])
#    pl.legend()
#    pl.show()

    return 0

if __name__ == '__main__':
    import sys
    sys.exit( _main() )
