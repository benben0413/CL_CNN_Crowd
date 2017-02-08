import numpy as np
import matplotlib.pyplot as plt
from numpy import unravel_index

def ExtractFace(orgImg, AttentionImg, LastLayer_maps, AttentionSize):

    class1 = LastLayer_maps[0][1]
    x0,y0 = unravel_index(class1.argmax(), class1.shape)

    h0 = x0 - 20
    h1 = x0 + 20
    w0 = y0 - 20
    w1 = y0 + 20

    if h0 < 0:
        diff = np.abs(h0)
        h0 += diff
        h1 += diff
    if w0 < 0:
        diff = np.abs(w0)
        w0 += diff
        w1 += diff
    if h1 > class1.shape[0]:
        diff = h1 - class1.shape[0]
        h0 -= diff
        h1 -= diff
    if w1 > class1.shape[1]:
        diff = w1 - class1.shape[1]
        w0 -= diff
        w1 -= diff

    croped_maps = class1[h0:h1, w0:w1]

    added_img = np.add(np.add(AttentionImg[0], AttentionImg[1]), AttentionImg[2])
    np.putmask(added_img, added_img > 0, 1)
    x0 = -1
    x1 = -1
    for i in range(added_img.shape[0]):
        if x0 < 0:
            if np.sum(added_img[i,:]) > 0:
                x0 = i
        if x0 > -1 and x1 < 0:
            if np.sum(added_img[i,:]) == 0:
                x1 = i
    y0 = -1
    y1 = -1
    for i in range(added_img.shape[1]):
        if y0 < 0:
            if np.sum(added_img[:,i]) > 0:
                y0 = i
        if y0 > -1 and y1 < 0:
            if np.sum(added_img[:,i]) == 0:
                y1 = i
    x0 = np.int(x0 + np.floor((x1-x0)/2))
    y0 = np.int(y0 + np.floor((y1-y0)/2))

    h0= x0-50
    h1= x0+50
    w0= y0-50
    w1= y0+50

    if h0 <0:
        diff = np.abs(h0)
        h0 += diff
        h1 += diff
    if w0 <0:
        diff = np.abs(w0)
        w0 += diff
        w1 += diff
    if h1 > orgImg.shape[0]:
        diff = h1 - orgImg.shape[0]
        h0 -= diff
        h1 -= diff
    if w1 > orgImg.shape[1]:
        diff = w1 - orgImg.shape[1]
        w0 -= diff
        w1 -= diff

    croped_img = orgImg[h0:h1,w0:w1,:]
    locs = [h0,h1,w0,w1]
    return croped_img, croped_maps, locs

    # new_img = np.array(np.multiply(batch[0].transpose(2, 0, 1), added_img), dtype=np.uint8).transpose(1, 2, 0)
