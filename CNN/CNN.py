import numpy
import theano.tensor as T
from cnn_structure import *
from cnn_training import fit_predict
from cnn_Prediction import cnn_predict
from helper import CNN_tester

import pickle as cPickle
import os.path

class CNN:
    def __init__(self,filename, batch_size = 50, learning_rate = 1e-2):

        seed = 8000
        rng = numpy.random.RandomState(seed)
        self.nkerns = [32, 50, 64, 96, 128, 128, 256]

        self.batch_size = batch_size
        self.x = T.tensor4('x')  # the data is presented as rasterized images
        self.lr = learning_rate

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'

        # construct the CNN class
        self.classifier = CNN_struct(
            rng=rng,
            input=self.x,
            nkerns=self.nkerns,
            batch_size=batch_size,
            image_size=[100, 100],
            image_dimension=3
        )


        self.file_name = filename
        if(os.path.isfile(filename)):
            f = open(filename, 'rb')
            self.classifier.__setstate__(cPickle.load(f))
            f.close()
            print ("Parameters loaded")
        else:
            print "No trained parameteres exist. network is initiated"

    def fit(self, data):
        fit_predict(self.classifier, self.x, data, self.batch_size, self.lr)

        # save parameters
        f = file(self.file_name, 'wb')
        cPickle.dump(self.classifier.__getstate__(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print ("parameters saved")


    def predict(self, test_dataset):
        return cnn_predict(self.classifier, self.x, test_dataset, self.batch_size)

    def tester_only(self, test_dataset,layer):
        return CNN_tester(self.classifier, self.x, test_dataset, self.batch_size, layer)

