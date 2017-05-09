import numpy as np
from numpy import genfromtxt
import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import sphviewer as sph
import weave


def myplot(x, y, nb=32, xsize=500, ysize=500):
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    x0 = (xmin+xmax)/2.
    y0 = (ymin+ymax)/2.

    pos = np.zeros([3, len(x)])
    pos[0,:] = x
    pos[1,:] = y
    w = np.ones(len(x))

    P = sph.Particles(pos, w, nb=nb)
    S = sph.Scene(P)
    S.update_camera(r='infinity', x=x0, y=y0, z=0,
                    xsize=xsize, ysize=ysize)
    R = sph.Render(S)
    R.set_logscale()
    img = R.get_image()
    extent = R.get_extent()
    for i, j in zip(xrange(4), [x0,x0,y0,y0]):
        extent[i] += j
    print extent
    return img, extent

fig = plt.figure(1, figsize=(10,10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)


# # Generate some test data
# x = np.random.randn(1000)
# y = np.random.randn(1000)

X = genfromtxt('/home/falmasri/Desktop/22.1-tosend.csv', delimiter=',')

X[X>0] = 1

loc = np.where(X>0)
loc0 = np.reshape(loc[0], (len(loc[0]),1))
loc1 = np.reshape(loc[1], (len(loc[1]),1))
locx= np.concatenate((loc0,loc1), axis=1)

#Plotting a regular scatter plot
ax1.plot(loc[0],loc[1],'.k')
# ax1.set_xlim(-3,3)
# ax1.set_ylim(-3,3)

# heatmap_16, extent_16 = myplot(loc[0],loc[1], nb=16)
# heatmap_32, extent_32 = myplot(loc[0],loc[1], nb=32)
# heatmap_64, extent_64 = myplot(loc[0],loc[1], nb=64)
#
# ax2.imshow(heatmap_16, extent=extent_16, origin='lower', aspect='auto')
# ax2.set_title("Smoothing over 16 neighbors")
#
# ax3.imshow(heatmap_32, extent=extent_32, origin='lower', aspect='auto')
# ax3.set_title("Smoothing over 32 neighbors")
#
# #Make the heatmap using a smoothing over 64 neighbors
# ax4.imshow(heatmap_64, extent=extent_64, origin='lower', aspect='auto')
# ax4.set_title("Smoothing over 64 neighbors")
#
# plt.show()







# m_plt = plt.plot(loc0, loc1, '.')
# # plt.gca().invert_yaxis()
# plt.show()
print locx [0:5]
print locx.shape
bandwidth = 0.9 #estimate_bandwidth(locx, quantile=0.5, n_samples=300)
print bandwidth
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

