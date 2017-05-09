from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from numpy import genfromtxt
import Image
import matplotlib.pyplot as plt
import cv2
from itertools import cycle




X = genfromtxt('/home/falmasri/Desktop/22.1-tosend.csv', delimiter=',')
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

# image = Image.open('/home/falmasri/Desktop/Tomorrowland B.jpg')
# image = np.array(image)
print X.shape

blur = cv2.GaussianBlur(X,(101,101),0)
plt.figure()
plt.subplot(121)
plt.imshow(X)
plt.subplot(122)
plt.imshow(blur)
plt.show()


# X[X > 0] = 1
# X[X <= 0] = 0

# X=np.reshape(X, [-1, 1])
# print X.shape
# bandwidth = 0.429546860904 #estimate_bandwidth(X, quantile=0.2, n_samples=500)
#
# print bandwidth
# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# ms.fit(X)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_
#
# labels_unique = np.unique(labels)
# n_clusters_ = len(labels_unique)
#
# print("number of estimated clusters : %d" % n_clusters_)
#
# print cluster_centers[1]
# print np.where(X == cluster_centers[0])
#
# # print cluster_centers
# plt.figure()
# plt.subplot(121)
# X = np.reshape(X, (424,437))
# plt.imshow(X)
# plt.colorbar()
# plt.subplot(122)
# labels = np.reshape(labels, (424,437))
# plt.imshow(labels)
# plt.colorbar()
# plt.show()

# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     # my_members = labels == k
#     cluster_center = cluster_centers[k]

#     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     # plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#     #          markeredgecolor='k', markersize=14)
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()

# plt.figure(2)
# plt.subplot(2, 1, 1)
# plt.imshow(X)
# plt.axis('off')
# plt.subplot(2, 1, 2)
# plt.imshow(np.reshape(labels, [424,437]))
# plt.axis('off')