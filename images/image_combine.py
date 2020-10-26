import cv2
import numpy as np
im1 = cv2.imread('img105.jpg')
im2 = cv2.imread('img105_rd.png')
#im2 = cv2.cvtColor(im2,cv2.COLOR_GRAY2RGB)
im_new = np.zeros((320,960,3))
im_new[0:320, 0:480] = im1
im_new[0:320, 480:960] = im2

cv2.imwrite('dataset1.jpg', im_new)
