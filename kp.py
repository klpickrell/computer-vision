#!/usr/bin/env python
import cv2
import numpy as np
from pylab import *

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def _main():
    surf = cv2.SURF(400)
    orb = cv2.ORB()
    
    #img1 = cv2.imread( 'test_images/items/cbsmall/cake136.jpg', cv2.COLOR_BGR2GRAY )
    #img2 = cv2.imread( 'test_images/items/cbsmall/cake140.jpg', cv2.COLOR_BGR2GRAY )
    
    img1 = cv2.imread( 'test_images/aerial/ap/DCIM/100MEDIA/DJI_0001.JPG', 0 )
    img2 = cv2.imread( 'test_images/aerial/ap/DCIM/100MEDIA/DJI_0002.JPG', 0 )

    img1 = cv2.resize(img1,None,fx=0.2, fy=0.2)
    img2 = cv2.resize(img2,None,fx=0.2, fy=0.2)

    
    #kp1, desc1 = surf.detectAndCompute(img1,None)
    #kp2, desc2 = surf.detectAndCompute(img2,None)
    
    kp1, desc1 = orb.detectAndCompute(img1,None)
    kp2, desc2 = orb.detectAndCompute(img2,None)
    
    bf = cv2.BFMatcher( cv2.NORM_HAMMING, crossCheck=True )
    matches = bf.match( desc1, desc2 )
    matches = sorted( matches, key=lambda x:x.distance )

    drawMatches(img1,kp1,img2,kp2,matches[:100])
#    img3 = cv2.drawMatches( img1, kp1, img2, kp2, matches[:10], flags=2 )

#    plt.figure()
#    plt.imshow(img3)
#    plt.show()
    
    #gr1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #img1 = cv2.drawKeypoints(gr1,kp1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    #gr2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #img2 = cv2.drawKeypoints(gr2,kp2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    #plt.figure()
    #plt.imshow(img1)
    #plt.figure()
    #plt.imshow(img2)
    #plt.show()

# Dense sift features for Knn
#dense=cv2.FeatureDetector_create("Dense")
#kp=dense.detect(imgGray)
#kp,des=sift.compute(imgGray,kp)

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(_main())

