import numpy as np
import cv2 as cv

img = cv.imread('home.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# SIFT, SURF, ORB, BRISK, BRIEF, FREAK ..
'''
surf = cv.SURF_create()
orb = cv.ORB_create()
brisk = cv.BRISK_create()
~~
'''
sift = cv.SIFT_create()
kp = sift.detect(gray,None)

img = cv.drawKeypoints(gray,kp,img)

cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()