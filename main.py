import cv2 as cv
import numpy as np
import time

# Process image without opencv
def threshold(thresh, img):
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if img[x,y] > thresh:
                img[x,y] = 255
            else:
                img[x,y] = 0
    return img

#read in an image into memory
for i in range(1,16):
    img = cv.imread('Orings/Oring' + str(i) + '.jpg')
    thresh = 100
    bw = threshold(thresh, img)
    rgb = cv.cvtColor(bw, cv.COLOR_GRAY2RGB)
    cv.imshow("Thresholded Image", rgb)
    cv.waitKey(0)
    cv.destroyAllWindows()