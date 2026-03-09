import cv2 as cv
import time
import numpy as np

# Process image without opencv
def threshold(thresh, img):
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if img[x,y] < thresh:
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

# Dialate code (adapted from erode code)
def dialate(img, num_levels) :
    for level in range (num_levels):
        out = img.copy()
        neighbours = [(-1, -1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                if img[i,j] == 0:
                    toErode = False
                    for y,x in neighbours:
                        if img[i+y,j+x] == 255:
                            toErode = True
                    if toErode:
                        out [i,j] = 255
        img = out
    return out

# Erode code (adapted from lab 3)
def erode(img, num_levels) :
    for level in range (num_levels):
        out = img.copy()
        neighbours = [(-1, -1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                if img[i,j] == 255: 
                    toErode = False
                    for y,x in neighbours:
                        if img[i+y,j+x] == 0:
                            toErode = True
                    if toErode:
                        out [i,j] = 0
        img = out
    return out

# Idea adapted from Lecture 2
def cclabelling(binary_img):
    rows, cols = binary_img.shape
    labels = np.zeros((rows, cols), dtype=int)
    label_counter = 1
    parent = {}

    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    for r in range(1, rows):
        for c in range(1, cols):
            if binary_img[r, c] == 255:
                n = labels[r-1, c]
                w = labels[r, c-1]

                if n == 0 and w == 0:
                    labels[r, c] = label_counter
                    parent[label_counter] = label_counter
                    label_counter += 1
                elif n != 0 and w != 0:
                    labels[r, c] = min(n, w)
                    parent[find(max(n, w))] = find(min(n, w))
                else:
                    labels[r, c] = max(n, w)

    for r in range(rows):
        for c in range(cols):
            if labels[r, c] > 0:
                labels[r, c] = find(labels[r, c])
    return labels

# Idea adapted from Lecture 2
def analysis(labels):
    unique = np.unique(labels)
    unique = unique[unique != 0] 
    broken = [l for l in unique if np.sum(labels == l) > 50]
    if len(broken) > 1:
        return "FAILED"

    target_label = broken[0]
    coords = np.column_stack(np.where(labels == target_label))
    centroid = coords.mean(axis=0)
    dists = np.sqrt(np.sum((coords - centroid)**2, axis=1))
    
    #main circularity formula from lecture
    mu_r = np.mean(dists)
    sigma_r = np.std(dists)
    circularity = sigma_r / mu_r

    if circularity > 0.08:
        return "PASS"
    else:
        return "FAILED"

#read in an image into memory
for i in range(1,16):
    img = cv.imread('Orings/Oring' + str(i) + '.jpg', 0)
    before = time.time()
    thresh = otsu(img)
    bw = threshold(thresh, img.copy())
    bw = dialate(bw, 3)
    bw = erode(bw, 3)
    labeled_img = cclabelling(bw)
    res = analysis(labeled_img)
    if res == "PASS": 
        color = (0, 255, 0) 
    else:
        color = (0, 0, 255)
    rgb = cv.cvtColor(bw, cv.COLOR_GRAY2RGB)
    after = time.time()
    fin = after-before
    print("Time taken to process: " + str(after-before))
    cv.putText(rgb, f"Status: {res}", (40, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv.putText(rgb, f"Time: {fin:.3f}", (40,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv.imshow("Thresholded Image", rgb)
    cv.waitKey(0)
    cv.destroyAllWindows()