from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

image_1 = plt.imread('11.jpg')/255
image = cv2.resize(image_1, (0, 0), fx = 0.1, fy = 0.1) 
print(image.shape)
plt.imshow(image)
pic_n = image.reshape(image.shape[0]*image.shape[1],image.shape[2])

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=25, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

cluster_pic = pic2show.reshape(image.shape[0], image.shape[1], image.shape[2])
plt.imshow(cluster_pic)

gray = rgb2gray(image)
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
print(gray_r)
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 3
    elif gray_r[i] > 0.5:
        gray_r[i] = 2
    elif gray_r[i] > 0.25:
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0],gray.shape[1])

# defining the sobel & laplace filters
sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])]) 
sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])

out_h = ndimage.convolve(gray, sobel_horizontal, mode='reflect')
out_v = ndimage.convolve(gray, sobel_vertical, mode='reflect')
out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')

# plt.imshow(gray, cmap='gray')
# plt.imshow(out_h, cmap='gray')
# plt.imshow(out_l, cmap='gray')
plt.show()



