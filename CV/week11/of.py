import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

import flow_vis


def optical_flow(I1g, I2g, window_size):

    

    I1g = I1g / 255. # normalize pixels
    I2g = I2g / 255. # normalize pixels 

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode='same')
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode='same')
    ft = I1g - I2g

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
# img1 = cv2.imread("./sphere0.png")
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# img2 = cv2.imread("./sphere1.png")
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img1 = cv2.imread("./bt0.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./bt1.png", cv2.IMREAD_GRAYSCALE)


# Obtain (u,v) from Lucas Kanade's optical flow approach
u, v = optical_flow( img1, img2, 3)

# print(U)

flow_color = flow_vis.flow_to_color(u, v, thr = 10)
plt.imshow(flow_color)
plt.show()