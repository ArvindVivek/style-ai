import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

img_path = f"data/subset/photo_2023-05-29-12-28-37-1-_jpeg_jpg.rf.8e5d19cd76994a1a6e09862b32f878b6.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, dim = img.shape

img = img[(height / 4) : (3 * height / 4), (width / 4) : (3 * width / 4), :]
height, width, dim = img.shape

img_vec = np.reshape(img, [height * width, dim])

kmeans = KMeans(n_clusters=3)
kmeans.fit(img_vec)

print(kmeans)
