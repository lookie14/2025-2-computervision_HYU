import numpy as np
from PIL import Image
import scipy
import cv2 as cv
import random

img = cv.imread("frame10.png")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

idx = dst>0.01*dst.max()

(img_h, img_w) = dst.shape 
corner_thr = 0.01*dst.max()

features = []
for i in range(0, img_h):
    for j in range(0, img_w):
        if dst[i][j] > corner_thr:#==corner!
            # img[i][j]=[0,0,255]
            features.append((j,i))


feature = random.choice(features)
img[feature[0]][feature[1]]=[0,255,255]
cv.circle(img, feature, 5, (0, 255, 255), 1)


cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()

