#!/usr/bin/env python
import cv2
import numpy as np
from pylab import *
from PIL import Image
from collections import OrderedDict
import sys
import os
from pymongo import MongoClient
import requests

from scipy.cluster.vq import *
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.neighbors     import NearestNeighbors

from gensim import corpora, models, similarities
from cStringIO import StringIO

target_filename = 'brights11.jpg'
#target_filename = 'kbimage239.jpg'
#directory = 'test_images/colors/kbvscp_small/'
directory = 'test_images/items/brightsvsneutrals/'
tfidf = None
THUMBNAIL_RATIO=1.0
jaccard_threshold=0.7

def img_to_matrix( filename, verbose=False ):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
#    if verbose==True:
#        print "(%s) changing size from %s to %s" % (filename, str(img.size), str(STANDARD_SIZE))
#    img = img.resize(STANDARD_SIZE)
    pixels = list(img.getdata())
#    if len(pixels[0]) != 3:
#        print( 'skipping {} due to alpha channel presence'.format(filename) )
#        return None,None
    data = map(list, pixels)
    data = np.array(data, dtype=np.int16)
    img.thumbnail( tuple((np.array(img.size)*THUMBNAIL_RATIO).astype(int)), Image.ANTIALIAS )
    return img,data

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

class TFIDFCompare:
    def __init__( self, documents ):
        corpus_file = '/var/tmp/taxonomy_corpus.mm'
        dict_file = '/var/tmp/taxonomy_dict.mm'
        if os.path.exists( corpus_file ) and os.path.exists( dict_file ):
            self.corpus = corpora.MmCorpus( corpus_file )
            self.dictionary = corpora.Dictionary.load( dict_file )
        else:
            self.tokenized_terms = [ item for item in documents ]
            self.dictionary = corpora.Dictionary( self.tokenized_terms )
            corpus = [ self.dictionary.doc2bow( item ) for item in self.tokenized_terms ]
            corpora.MmCorpus.serialize( corpus_file, corpus )
            self.corpus = corpora.MmCorpus( corpus_file )
            self.dictionary.save( dict_file )

        self.tfidf = models.TfidfModel( self.corpus )

    def compare( self, s1, s2 ):
        e1 = self.dictionary.doc2bow([ item.lower() for item in s1 ])
        e2 = self.dictionary.doc2bow([ item.lower() for item in s2 ])
        t1 = self.tfidf[e1]
        t2 = self.tfidf[e2]
        index = similarities.MatrixSimilarity( [t1], num_features=self.corpus.num_terms )
        return index[t2][0]

def tfidf_compare_terms( s1, s2 ):
    global tfidf
    return 1.0-tfidf.compare(s1,s2)

def init_tfidf( mongo_collection ):
    global tfidf
    terms = [ rec.get('terms',[]) for rec in mongo_collection.find( {}, { 'terms' : 1 } ) ]
    tfidf = TFIDFCompare( terms )

def jaccard_distance( A, B ):
    # jaccard index is ( AnB ) / (AuB)
    return 1.0-( len(set(A).intersection( set(B) )) / max( 0.00000001, float(len(set(A).union(set(B)))) ) )


def _main():

    client = MongoClient('localhost')
    collection = client['real_weddings']['photos']
    features_collection = client['real_weddings_qa']['photo_features']

# tfidf term similarity by image (find 25 most similar)
    print('initializing tfidf')
    init_tfidf( collection )
    print('done')

#    test_ids = [ '1c3d9c00-dcdc-a933-1e7c-3ae6165eaf84' ]
#    test_ids = [u'1e811f23-e833-e98c-ad6b-54ff25791682',
#                u'6fc9a8ad-1dba-f299-14af-75a558972630',
#                u'8211ec14-9752-7618-03d2-875729900bf2',
#                u'a2015dae-e00c-c526-f232-aa517949eddc',
#                u'5cf2fabe-350a-ed36-6602-f57eb4b25275']

#    test_ids = [u'8e10f72f-e748-9188-9539-02e4b2a92515',
#                u'891e60b4-9dee-1f5e-3c6c-e1a946cbc9ea',
#                u'fd6d2731-635a-0cc9-8d2d-4d504d3bb242',
#                u'8ecb8d31-332c-9f94-7345-a71778c2c191',
#                u'84b13d08-7a4b-926e-9ee5-4c9ae106423b']

#    test_ids = [u'a4a1cc29-7469-d360-30a5-eb3a39f6b60e',
#                u'bbe6ad52-96fb-de3d-908d-8d0e9b778b83',
#                u'75fa439b-6118-3e4f-18de-3ae4669796a1',
#                u'0f71d4fd-b73a-0135-21eb-49559ad8b359',
#                u'889a7627-9e40-c075-6367-1a402a569e93']

#    test_ids = [u'1c3d9c00-dcdc-a933-1e7c-3ae6165eaf84',
#                u'a4a1cc29-7469-d360-30a5-eb3a39f6b60e',
#                u'8e10f72f-e748-9188-9539-02e4b2a92515',
#                u'5c7fe76d-561e-3f78-8770-72b19c96651a']

    test_ids = [ #u'13c266d4-c53c-66b5-ff46-0e3c67695638', #purple reception
                 #u'792fce48-21ea-e28c-d55b-9edd8e430408', #red reception decor
                 #u'f7cfe0e7-e0d3-f664-89c1-db444a10f430', #lavishtented
                 #u'ab4f77ba-8e9a-0803-2839-de98104ab857', #imamermaid
                 #u'82ccacd9-abd7-492e-1a52-b1c0b966c9c8', #beach2
                 #u'fe4fd7bc-d1b5-11e4-be0a-22000aa61a3e',  #beach1
                 #u'00355f48-9943-3978-c163-5bc702fe1cc1', #image8 - boutonierre
                 #u'2dec2b05-3d6e-80be-2b63-44d274759103', #image9
                 #u'372b9e12-f7f3-8404-07bf-586f30be557a', #image7
#                 u'0bd01650-c62a-a395-f72e-5e02bfba2419', #image6 - car
                 u'50509852-ec15-cd27-9c1b-4018e198f8e0'] #image5 - mint cake
#                 u'b319629a-7c5b-afa7-5efe-ffce984417ff',  #image4
#                 u'bbb5de15-6577-70ea-1f04-c0673c96c1fa',  #image3
#                 u'c310397c-5bf4-0a4f-e865-8143b80d2a26', #image2
#                 u'35a64492-6542-de82-456f-83303847e0de' ] #image1

    test_items = { record['_id'] : record for record in collection.find( { '_id' : { '$in' : test_ids } } ) }

    for test_id in test_ids:
        test_item = test_items[test_id]
        print( 'target is {}'.format(test_item.get('legacy_url','')) )

        test_terms = test_item.get('terms',[])
    
        print( 'pulling terms and images from real weddings photos' )
        all_terms = {}
        all_images = {}
        for record in collection.find( {}, { 'terms' : 1, 'image_url' : 1 } ):
            rid = record['_id']
            t = record.get('terms',[])
            if t is None:
                t = []
            all_terms[rid] = t
            all_images[rid] = record.get('image_url','')
        print( 'done' )
    
        print( 'calculating all tfidf distances' )
        sorted_distances = sorted( { key : tfidf_compare_terms( test_terms, terms ) for key,terms in all_terms.iteritems() }.items(), key=lambda x:x[1] )
        print( 'done' )
    
        print( 'calculating all jaccard distances' )
        sorted_jaccard_distances = sorted( { key : jaccard_distance( test_terms, terms ) for key,terms in all_terms.iteritems() }.items(), key=lambda x:x[1] )
        print( 'done' )
    
        images = []
        top_count = 10
        print( 'retrieving top {} images'.format(top_count) )
        top_keys = [ test_id ]
        top_keys.extend([ item[0] for item in sorted_distances[:top_count] if item[0] != test_id ])
        top_items = [ all_images[item] for item in top_keys ]
        for image_url in top_items:
            if not image_url:
                print( 'empty image_url' )
                continue
            print( 'image is {}'.format(image_url) )
            response = requests.get(image_url)
            if response.status_code != 200:
                print( 'failed to retrieve {}, skipping'.format(image_url) )
            image_data = StringIO( response.content )
            image = Image.open(image_data)
            image.thumbnail( (image.size[0]*0.2,image.size[1]*0.2) )
            images.append(image)
    
        print( 'done' )
    
        plt.figure()
        plt.gray()
        plt.suptitle('tfidf')
        for idx, image in enumerate(images):
            plt.subplot(5,15,idx+1)
            plt.imshow( image )
            plt.axis('off')
    
        jimages = []
        jtop_count = 10
        print( 'retrieving top {} jaccard images'.format(jtop_count) )
        jtop_keys = [ test_id ]
        jtop_keys.extend([ item[0] for item in sorted_jaccard_distances[:jtop_count] if item[0] != test_id ])
        jtop_items = [ all_images[item] for item in jtop_keys ]
        for image_url in jtop_items:
            if not image_url:
                print( 'empty image_url' )
                continue
            print( 'image is {}'.format(image_url) )
            response = requests.get(image_url)
            if response.status_code != 200:
                print( 'failed to retrieve {}, skipping'.format(image_url) )
            image_data = StringIO( response.content )
            image = Image.open(image_data)
            image.thumbnail( (image.size[0]*0.2,image.size[1]*0.2) )
            jimages.append(image)
    
        print( 'done' )
    
        plt.figure()
        plt.gray()
        plt.suptitle('jaccard')
        for idx, image in enumerate(jimages):
            plt.subplot(5,15,idx+1)
            plt.imshow( image )
            plt.axis('off')
    
    
        relevant_image_keys = [ item[0] for item in sorted_jaccard_distances if item[1] < jaccard_threshold ]
        print( '{} relevant images to analyze'.format(len(relevant_image_keys)) )
        relevant_images = []
        for key in relevant_image_keys:
            record = features_collection.find_one( { '_id' : key } )
            if record:
               relevant_images.append( record.get( 'filename','') )
    
        
        record = features_collection.find_one( { '_id' : test_id } )
        if relevant_images and record:
            shuffle( relevant_images )
        
            test_image_file = record['filename']
            test_image_index = relevant_images.index(test_image_file)
        
            data = []
            thumbnails = []
            for image in relevant_images:
    #            print( 'process {}'.format(image) )
                if 'jpeg' not in image:
                    print( 'skipping {}'.format(image) )
                    continue
        
                idata, img = img_to_matrix(image, True)
                if img is not None:
                    thumbnails.append( np.array(idata) )
                    img = flatten_image(img)
                    data.append(img)
        
            data = np.array( data, dtype=np.ndarray )
            pca = RandomizedPCA(n_components=8)
            X = pca.fit_transform(data)
    
    #spectral clustering
            n = len(X)
            projected = X
            S = np.array([[ np.sqrt(np.sum((projected[i]-projected[j])**2))
                            for i in range(n) ] for j in range(n)], 'f')
            rowsum = np.sum(S,axis=0)
            D = np.diag(1 / np.sqrt(rowsum))
            I = np.identity(n)
            L = I - np.dot(D,np.dot(S,D))
            U,sigma,V = linalg.svd(L)
            features = whiten(np.array(V[:16]).T)
    
            nn = NearestNeighbors(min(10,len(features)))
            nn.fit(features)
            neighbors = nn.kneighbors(features[test_image_index], return_distance=False)[0]
            plt.figure()
            plt.gray()
            plt.suptitle('Spectral')
            for idx, neighbor in enumerate(neighbors):
                plt.subplot(5,15,idx+1)
                plt.imshow( thumbnails[neighbor] )
                plt.axis('off')
                print( 'image is {}'.format(relevant_images[neighbor]) )
        else:
            print( '{} not found in collection or no relevant images'.format(test_id) )

        plt.show()

#neighborhood search on PCA
#        nn = NearestNeighbors(min(10,len(X)))
#        nn.fit(X)
#        neighbors = nn.kneighbors(X[test_image_index], return_distance=False)[0]
#        plt.figure()
#        plt.gray()
#        plt.suptitle('PCA')
#        for idx, neighbor in enumerate(neighbors):
#            plt.subplot(5,15,idx+1)
#            plt.imshow( thumbnails[neighbor] )
#            plt.axis('off')
#    
#        plt.show()


#    n = len(X)
#    projected = X
#    S = np.array([[ np.sqrt(np.sum((projected[i]-projected[j])**2))
#                    for i in range(n) ] for j in range(n)], 'f')
#    rowsum = np.sum(S,axis=0)
#    D = np.diag(1 / np.sqrt(rowsum))
#    I = np.identity(n)
#    L = I - np.dot(D,np.dot(S,D))
#    U,sigma,V = linalg.svd(L)
#    kmax = 17
#    features = whiten(np.array(V[:kmax]).T)
#    centroids, distortion = kmeans(features,kmax)
#    code, distance = vq( features, centroids )


# jaccard similarity by image (find 25 most similar)


# this does some image similarity
#    global filename
#    filename = sys.argv[0]

#    target = os.path.join( directory, target_filename )
#    target_image_pil = Image.open(target).convert('RGB')
#    STANDARD_SIZE = (int(target_image_pil.size[0]/2.0), int(target_image_pil.size[1]/2.0))
#    
#    target_image_pil = target_image_pil.resize( STANDARD_SIZE )
#    target_image = cv2.cvtColor( np.array(target_image_pil), cv2.cv.CV_BGR2GRAY )
#
#    images = {}
#    feature_detector = cv2.FeatureDetector_create('ORB')
#    descriptor = cv2.DescriptorExtractor_create('BRIEF')
#
#    target_edges = cv2.Canny( target_image, 100, 200 )
#    target_features = feature_detector.detect( target_edges )
#    target_keypoints, target_descriptors = descriptor.compute( target_edges, target_features )
#
#    matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming") #BruteForce,BruteForce-Hamming,BruteForce-L1,FlannBased
#
#    for (top, _, files) in os.walk( directory ):  #directory, subdirectories in directory, files in directory
#        for filename in files:
#            image_file = os.path.join( top, filename )
#            print( 'processing {}'.format(image_file) )
#            images[image_file] = {}
#            img = Image.open(image_file).convert('RGB')
#            img = img.resize(STANDARD_SIZE)
#            cv_img = cv2.cvtColor( np.array(img), cv2.cv.CV_RGBA2RGB )
#            
#            # grabcut the foreground
##            crop_pct = 0.9
##            width,height = img.size
##            halfwidth,halfheight = int((1-crop_pct)*width/2.0), int((1-crop_pct)*height/2.0)
##            rect = (0+halfwidth,0+halfheight,width-halfwidth,height-halfheight)
##            mask = np.zeros(cv_img.shape[:2],dtype = np.uint8)
##            output = None
##            for i in xrange(5):
##                print( '...iterating {} grabcut {}'.format(i+1,image_file) )
##                if i == 0:
##                    init_options = cv2.GC_INIT_WITH_RECT
##                else:
##                    init_options = cv2.GC_INIT_WITH_MASK
##
##                bgdmodel = np.zeros((1,65),np.float64)
##                fgdmodel = np.zeros((1,65),np.float64)
##                cv2.grabCut(cv_img,mask,rect,bgdmodel,fgdmodel,1,init_options)
##                mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
##                output = cv2.bitwise_and(cv_img,cv_img,mask=mask2)
#            output = cv_img.copy()
#            output = cv2.cvtColor( output, cv2.cv.CV_RGB2GRAY )
#            output = cv2.Canny(cv_img,100,200)
#
#            images[image_file]['data'] = output
#            images[image_file]['original_image'] = cv_img
#            features = feature_detector.detect(output)
#            keypoints, descriptors = descriptor.compute( output,features )
#            images[image_file]['keypoints'] = keypoints
#            images[image_file]['descriptors'] = descriptors
#            matches = matcher.match( target_descriptors, descriptors )
#            if len(matches) != len(target_descriptors):
#                print 'not the same number of matches!'
#                continue
#            matches = sorted( matches, key=lambda x:x.distance )
#            score = sum( item.distance for item in list(matches)[:10] )
#            images[image_file]['score'] = score
#            print( 'score is {}'.format( images[image_file]['score'] ) )
#
#    imsorted = OrderedDict({})
#    for image_file, content in sorted( (item for item in images.items()), key=lambda x:x[1]['score'] ):
#        imsorted[image_file] = content
#
#    plt.figure()
#    plt.gray()
#    for idx, (image_file, content) in enumerate(imsorted.iteritems()):
#        plt.subplot(5,15,idx+1)
#        plt.title('{}'.format(os.path.basename(image_file)) )
#        plt.imshow( content['original_image'] )
#        plt.axis('off')
#        if idx > 50:
#            break
#    plt.show()

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(_main())

