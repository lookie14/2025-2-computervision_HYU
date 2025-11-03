import numpy as np
from PIL import Image
import scipy
import cv2 as cv
import random

img = cv.imread("scale1.png")
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

#step 1.
(feat_x, feat_y) = random.choice(features) # a single corner point (x, y), boundary condition chk!
print((feat_x, feat_y) )

#step 2.
patch_size = 23
patch_size_half = patch_size//2 
W = gray[feat_y-patch_size_half:feat_y+patch_size_half+1, feat_x-patch_size_half:feat_x+patch_size_half+1]/255.0 #23x23 patch, brightness range=(0,1)


#step 3
# df_dx = conv(W, s_x)
# dw_dy = conv(W, s_y)

#step 4.
alngle_histogram = np.ones(36)#36bins
# for i in range(patch_size):
#     for j in range(patch_size):
#         angle = XX
#         alngle_histogram[angle] +=1

#step 5.
dominant_angle = max(alngle_histogram)#degree (not radian)


#step 6. do rotation normalization
rot_matrix = cv.getRotationMatrix2D((patch_size/2, patch_size/2), -dominant_angle, 1)
rot_W = cv.warpAffine(W, rot_matrix, (patch_size, patch_size))


#step 7. extract 16x16 window
w = rot_W[patch_size_half-8 : patch_size_half+8, patch_size_half-8 : patch_size_half+8]#16x16 window w

#step 8. divide w into 4x4 cells
cell = []
cell[0] = w[0:4, 0;4]
cell[1] = w[4:0, 0:4]
...

#step 9. build hog for each cell
hog = []
#hog[0] = hog from cell[0]
#hog[1] = hog from cell[1]

#step 10. construct 128-dim dscr
sift_dscr = []
for n in range(16):
    sift_dscr = sift_dscr + cell[n]

