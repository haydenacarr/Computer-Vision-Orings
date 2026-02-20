import cv2 as cv
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

# Otsu method for thresholding (code adapted from lecture 2)
def otsu(img):
    hist = [0]*img.size
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i, j]] += 1

    total_pixels = img.shape[0] * img.shape[1]

    total_sum = 0
    for t in range(256):
        total_sum += t * hist[t]

    wBackground = 0
    wForeground = 0

    sumB = 0
    varMax = 0
    thresh = 0

    for t in range(256):
        wBackground += hist[t]
        if wBackground == 0: continue

        wForeground = total_pixels - wBackground
        if wForeground == 0: break

        sumB += t * hist[t]
        mBackground = sumB / wBackground
        mForeground = (total_sum - sumB) / wForeground

        varBetween = wBackground * wForeground * (mBackground - mForeground) * (mBackground - mForeground)
        if varBetween > varMax:
            varMax = varBetween
            thresh = t

    return thresh

#read in an image into memory
for i in range(1,16):
    img = cv.imread('Orings/Oring' + str(i) + '.jpg', 0)
    before = time.time()
    thresh = otsu(img)
    bw = threshold(thresh, img)
    rgb = cv.cvtColor(bw, cv.COLOR_GRAY2RGB)
    after = time.time()
    fin = after-before
    print("Time taken to process: " + str(after-before))
    cv.putText(rgb, f"Time: {fin:.3f}", (40,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv.imshow("Thresholded Image", rgb)
    cv.waitKey(0)
    cv.destroyAllWindows()