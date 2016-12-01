from __future__ import division
import os
import sys, getopt
import time
import numpy as np
import six.moves.cPickle as pickle
from DataLoading import *
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from OCSVM import *
from os.path import isfile, join
import cv2

from cnn_training import fit, predict

def run():


    #for training purpose
    # # read the data, labels
    # mydict = load("data/resized_ds_50-50.pkl")
    # train_set_x, train_set_y = mydict
    # tr_data = train_set_x.get_value(borrow=True)[:2950,:]
    # # tr_labels = train_set_y.get_value(borrow= True)
    #
    # #load data to test including other dataset.
    # train_set_y = train_set_y.get_value(borrow=True)[:450]
    #
    # mydict = load("data/resized_INRIA_HOLIDAY_50-50.pkl")
    # test_set_x, test_set_y = mydict
    # test_data = test_set_x.get_value(borrow=True)
    # test_labels = np.zeros(len(tr_data))
    #
    # random_set = np.zeros((50,2500))
    # random_labels = np.zeros(50)
    # for i in range(50):
    #     Z = np.random.random((50, 50)).flatten()
    #     random_set[i] = Z
    #
    # test_data = np.concatenate((tr_data,random_set, test_data), axis=0)
    # test_labels = np.concatenate((train_set_y,random_labels,test_labels), axis=0)

    # for testing purpose
    mydict = load("data/resized_Tomorrowland_50-50.pkl")
    test_data, test_set_y = mydict

    print ". . finished reading"
   
    # fit(tr_data, tr_labels)
    # print "finished training"

    # rv = predict(valid_data)

    # create and predict test set
    # for i in range(600,len(test_data),1):
    # plt.imshow(test_data[618].reshape((50, 50)), cmap=cm.Greys_r)
    # plt.grid(False)
    # plt.show()

    # test_data = test_data[:500,:]

    print ("extracting features by CNN...")

    rt, p = predict(test_data)

    # V = plot_representations(p)
    # fit_model(p)
    OCSVM_results = test_OCSVM(p,10000)
    print OCSVM_results[OCSVM_results == -1].size


    for i in xrange(len(OCSVM_results)):
        if OCSVM_results[i] == -1:

            img = np.array(test_data[i]).reshape((50,50))
            # plt.imshow(img, cmap=cm.Greys_r)
            # plt.show()
            cv2.imwrite(join("/media/falmasri/14BAB2D2BAB2AF9A/frames_concert/t",
                             "img_%d.png" % i), img)


     # diff =  test_labels[0:len(rt)] - rt
    # print ("error is %f" %float(len(diff[np.nonzero(diff)]) * 100 / len(rt)))


    # UNDO argmax and save results x 2
    # r = rv
    # N = len(r)
    # res = np.zeros((N, 10))
    # for i in range(N):
    #     res[i][r[i]] = 1
    
    # np.savetxt("mnist_valid.predict", rv, fmt='%i')
    
    # r = rt
    # N = len(r)
    #
    #
    # res = np.zeros((N, 10))
    # for i in range(N):
    #     res[i][r[i]] = 1
    #
    #
    # np.savetxt("mnist_test.predict", res, fmt='%i')
    # print "finished predicting."
   

def plot_representations(p):
    # print p[0:500]
    # print p[500:1000]

    if len(p[0]) < 3:
        px = []
        py = []
        for x in xrange(len(p)):
            px.append(p[x][0])
            py.append(p[x][1])

        plt.scatter(px[0:2950],py[0:2950],color='r')
        plt.scatter(px[2950:3000],py[2950:3000],color='g')
        plt.scatter(px[3000:3500], py[3000:3500], color='b')
    else:
        pca = PCA(n_components=2)
        pca.fit(p.T)
        V = pca.components_
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(V[0][0:2500], V[1][0:2500], color='r')
        ax.scatter(V[0][2500:2950], V[1][2500:2950], color='y')
        ax.scatter(V[0][2950:3000], V[1][2950:3000], color='g')
        ax.scatter(V[0][3000:3500], V[1][3000:3500], color='b')


        # pca = PCA(n_components=3)
        # pca.fit(p.T)
        # V = pca.components_
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(V[0][0:2500], V[1][0:2500], V[2][0:2500], color='r')
        # ax.scatter(V[0][2500:2950], V[1][2500:2950], V[2][2500:2950], color='y')
        # ax.scatter(V[0][2950:3000], V[1][2950:3000], V[2][2950:3000], color='g')
        # ax.scatter(V[0][3000:3500], V[1][3000:3500], V[2][3000:3500], color='b')

    # plt.show()
    return V

def load(filename):
    pkl_file = open(filename, 'rb')
    mydict = pickle.load(pkl_file)
    pkl_file.close()
    return mydict

if __name__ == '__main__':
    run()

    
    
    
    
    
    

    
    
    
    
    
    













