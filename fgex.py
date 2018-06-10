#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import filters
from scipy import linalg
import os
import pandas as pd
import pylab as pl
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import re

from scipy.cluster.vq import *
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from random import shuffle

STANDARD_SIZE = (75,100)#(300, 400)
THUMBNAIL_SIZE = (256, 512)
#root_dir = 'items/cakesvsbouquets'
#root_dir = 'test_images/items/cbsmall'
#root_dir = 'test_images/colors/kbvscp'
#root_dir = 'test_images/items/brightsvsneutrals'
root_dir='raw'
#root_dir='.'
def img_to_matrix( filename, verbose=False ):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    if verbose==True:
        print "(%s) changing size from %s to %s" % (filename, str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    data = np.array(img.getdata(), dtype=np.int16)
    try:
        if data.shape != (STANDARD_SIZE[0]*STANDARD_SIZE[1],3):
            print( 'skipping {} due to alpha channel presence'.format(filename) )
            return None,None

    except Exception, ex:
        print( 'skipping {} due to unknown exception'.format(ex) )
        return None,None

    img.thumbnail( THUMBNAIL_SIZE, Image.ANTIALIAS )
    return img,data

def img_to_edges( filename, dense_detector, sift, verbose=False ):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename).convert('RGB')
    if verbose==True:
        print "(%s) changing size from %s to %s" % (filename, str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    pixels = list(img.getdata())
    if len(pixels[0]) != 3:
        print( 'skipping {} due to alpha channel presence'.format(filename) )
        return None,None

#    edge_data = np.array(img, dtype=np.int16)
    edge_data = cv2.cvtColor( np.array(img), cv2.COLOR_RGBA2GRAY )
    edges = cv2.Canny( edge_data, 100, 200 )

    kp = dense_detector.detect( edges )
    kp, des = sift.compute( edge_data, kp)
    data = map(list, list(des))
    data = np.array(data, dtype=np.int16)

    img.thumbnail( THUMBNAIL_SIZE, Image.ANTIALIAS )
    return img,edges

def img_to_dense_sift( filename, dense_detector, sift, verbose=False ):
    """
    takes a filename and turns it into a numpy array of dense sift descriptors
    """
    img = Image.open(filename)
    if verbose==True:
        print "(%s) changing size from %s to %s" % (filename, str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    pixels = list(img.getdata())
    if len(pixels[0]) != 3:
        print( 'skipping {} due to alpha channel presence'.format(filename) )
        return None,None

    img.convert('RGB')
    dimg = np.array(img)
#    img_cv = cv2.cvtColor(data, cv2.cv.CV_BGR2RGB)
#    gray = cv2.cvtColor( dimg, cv2.COLOR_BGR2GRAY )
#    kp = dense_detector.detect( gray )
    kp = dense_detector.detect( dimg )
#    kp, des = sift.compute( gray, kp)
    kp, des = sift.compute( dimg, kp)
    data = map(list, list(des))
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


def test_remove_bg():
    img = cv2.imread('items/cakesvsbouquets/cake99.jpg')
    mask = np.zeros(img.shape[:2],np.uint8)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    yheight = (img.shape[0]*0.6)
    ystart = int((img.shape[0]-yheight)/2.0)
    yend   = int(ystart+yheight)
    rect = (50,ystart,img.shape[1]-50,yend)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    
    plt.imshow(img),plt.colorbar(),plt.show()


def test_remove_blur():
    img = cv2.imread('items/cakesvsbouquets/cake99.jpg')
    blurred = np.zeros(img.shape)
    for i in range(3):
        blurred[:,:,i] = filters.gaussian_filter(img[:,:,i],3)
    blurred = np.array( blurred, 'uint8' )
    masked = blurred
    cv2.addWeighted(img, 1.5, blurred, -0.5, 0, masked)
    for i in range(3):
        masked[:,:,i] = filters.gaussian_filter(img[:,:,i],3)
    blurred = masked
    cv2.addWeighted(img, 1.5, masked, -0.5, 0, blurred)

# here
    img = blurred
    mask = np.zeros(img.shape[:2],np.uint8)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    yheight = (img.shape[0]*0.6)
    ystart = int((img.shape[0]-yheight)/2.0)
    yend   = int(ystart+yheight)
    rect = (50,ystart,img.shape[1]-50,yend)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    
    plt.imshow(img),plt.colorbar(),plt.show()

#    plt.imshow(blurred)
#    plt.show()

def test_cluster():
    images = [ os.path.join(root_dir,f) for f in os.listdir(root_dir) if 'jpg' in f ]
    shuffle( images )
    labels = []
    data = []
    image_data = []
    reg = re.compile( r'^([a-zA-Z]+)[0-9]+.*' )
#    dense=cv2.FeatureDetector_create("Dense")
#    sift = cv2.SIFT()
    for image in images:
        print( 'process {}'.format(image) )
        if 'jpg' not in image:
            print( 'skipping {}'.format(image) )
            continue

        idata, img = img_to_matrix(image, True)  # by color
#        idata, img = img_to_dense_sift(image, dense, sift, True)  # by texture
#        idata, img = img_to_edges(image, dense, sift, True)  # by edges
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
     
    data = np.array( data, dtype=np.ndarray )
    pca = RandomizedPCA(n_components=8)
    X = pca.fit_transform(data)
    n = len(X)
    projected = X

    # spectral clustering
    # Compute k-means on the eigenvectors of the similarity matrix
    # compute distance matrix
    S = np.array([[ np.sqrt(np.sum((projected[i]-projected[j])**2))
                    for i in range(n) ] for j in range(n)], 'f')
    # create Laplacian matrix
    rowsum = np.sum(S,axis=0)
    D = np.diag(1 / np.sqrt(rowsum))
    I = np.identity(n)
    L = I - np.dot(D,np.dot(S,D))
    # compute eigenvectors of L
    U,sigma,V = linalg.svd(L)
#    kmax = int(np.sqrt( len(image_data)/2.0 ))  # 16?
    kmax = 30
    # create feature vector from k first eigenvectors
    # by stacking eigenvectors as columns
    features = whiten(np.array(V[:kmax]).T)
    centroids, distortion = kmeans(features,kmax)
    code, distance = vq( features, centroids )

# Compute k-means on the projected components of data
#    kmax=int(np.sqrt( len(image_data)/2.0 ))
#    projected = whiten(projected)
#    centroids, distortion = kmeans(projected,kmax)
#    code, distance = vq( projected, centroids )

    plt.figure()

#    ndx = np.where(code==0)[0]
#    plt.plot( projected[ndx,0], projected[ndx,1], 'r*' )
#
#    ndx = np.where(code==1)[0]
#    plt.plot( projected[ndx,0], projected[ndx,1], 'b*' )
#
#    ndx = np.where(code==2)[0]
#    plt.plot( projected[ndx,0], projected[ndx,1], 'g*' )
#
#    ndx = np.where(code==3)[0]
#    plt.plot( projected[ndx,0], projected[ndx,1], 'y*' )
#
#    plt.plot( centroids[:,0], centroids[:,1], 'go' )
#    plt.plot( centroids[:,2], centroids[:,3], 'go' )
#    plt.axis('off')
#    plt.show()

    for k in range(kmax):
        ind = np.where(code==k)[0]
        plt.figure()
        plt.gray()
        for i in range(min(len(ind),40)):
            plt.subplot(4,10,i+1)
            plt.imshow( np.array(image_data[ind[i]]) )
            plt.axis('off')
    plt.show()

def _main():
#    test_remove_bg()
#    test_remove_blur()
    test_cluster()

if __name__ == '__main__':
    import sys
    sys.exit( _main() )
