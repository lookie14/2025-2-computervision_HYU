import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

import flow_vis

def warp(u, v, img):
    
    flow = np.concatenate((np.expand_dims(u, -1), np.expand_dims(v, -1)), axis = 2)
    flow = -flow
    h, w = flow.shape[:2]
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    
    img = img.astype(np.float32)
    flow = flow.astype(np.float32)
    
    warpedImg = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return warpedImg
    



def optical_flow(I1g, I2g, window_size):

    

    I1g = I1g / 255. # normalize pixels
    I2g = I2g / 255. # normalize pixels 

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode='same')
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode='same')
    ft = I1g -I2g

    # ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)

    w = window_size//2 # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            b = np.reshape(It, (It.shape[0],1))
            A = np.vstack((Ix, Iy)).T

            U = np.matmul(np.linalg.pinv(A), b)     # Solving for (u,v) i.e., U

            u[i,j] = U[0][0]
            v[i,j] = U[1][0]
 
    return (u,v)



#   Read Input
img1 = cv2.imread("./tsukuba1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./tsukuba2.png", cv2.IMREAD_GRAYSCALE)
h, w = img1.shape[:2]
w_8 = w//8
h_8 = h//8
w_4 = w//4
h_4 = h//4
w_2 = w//2
h_2 = h//2

scale = 8
img1_resized = cv2.resize(img1, dsize=(w_8, h_8), interpolation=cv2.INTER_CUBIC)
img2_resized = cv2.resize(img2, dsize=(w_8, h_8), interpolation=cv2.INTER_CUBIC)

# Obtain (u,v) from Lucas Kanade's optical flow approach
u, v = optical_flow(img1_resized, img2_resized, 3)

img1_resized = cv2.resize(img1, dsize=(w_4, h_4), interpolation=cv2.INTER_CUBIC)
img2_resized = cv2.resize(img2, dsize=(w_4, h_4), interpolation=cv2.INTER_CUBIC)

u = 2*cv2.resize(u, dsize=(w_4, h_4), interpolation=cv2.INTER_LINEAR)
v = 2*cv2.resize(v, dsize=(w_4, h_4), interpolation=cv2.INTER_LINEAR)
warpedImg = warp(u, v, img1_resized)
u_, v_ = optical_flow(warpedImg, img2_resized, 3)
u += u_
v += v_


img1_resized = cv2.resize(img1, dsize=(w_2, h_2), interpolation=cv2.INTER_CUBIC)
img2_resized = cv2.resize(img2, dsize=(w_2, h_2), interpolation=cv2.INTER_CUBIC)
u = 2*cv2.resize(u, dsize=(w_2, h_2), interpolation=cv2.INTER_LINEAR)
v = 2*cv2.resize(v, dsize=(w_2, h_2), interpolation=cv2.INTER_LINEAR)
warpedImg = warp(u, v, img1_resized)
u_, v_ = optical_flow(warpedImg, img2_resized, 3)
u += u_
v += v_


u = 2*cv2.resize(u, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
v = 2*cv2.resize(v, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
warpedImg = warp(u, v, img1)
u_, v_ = optical_flow(warpedImg, img2, 3)
u += u_
v += v_



# u = scale*cv2.resize(u, dsize=(w,h), interpolation=cv2.INTER_LINEAR)
# v = scale*cv2.resize(v, dsize=(w,h), interpolation=cv2.INTER_LINEAR)

# warpedImg = warp(u, v, img1)

# # u, v = optical_flow( img1, img2, 3)

# u_, v_ = optical_flow( warpedImg, img2, 3)
# u += u_
# v += v_


# print(U)

flow_color = flow_vis.flow_to_color(u, v, thr = 10)
plt.imshow(flow_color)
plt.show()