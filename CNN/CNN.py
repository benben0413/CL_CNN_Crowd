import os.path
import pickle as cPickle

from Structures.cnn_structure3 import *
from Sequemtial_cnn_Prediction import *
from cnn_Prediction import cnn_predict
from cnn_training import fit_predict
from helper import CNN_tester


class CNN:
    def __init__(self,filename, learning_rate= 1e-5, batch_size = 50, imageSize = [100, 100]):

        seed = 8000
        rng = numpy.random.RandomState(seed)
        self.nkerns = [64, 64, 96, 96, 128, 128, 128]
        # self.nkerns = [64, 128, 128, 256, 512, 512, 512]
        # self.nkerns = [32, 64, 64, 96, 128, 128]
        self.batch_size = batch_size
        self.x = T.tensor4('x')  # the data is presented as rasterized images
        self.lr = learning_rate

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'

        # construct the CNN class
        self.classifier = CNN_struct3(
            rng=rng,
            input=self.x,
            nkerns=self.nkerns,
            batch_size=batch_size,
            image_size=imageSize,
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

    def fit(self, data, save_params,plot_fig ):
        fit_predict(self.classifier, self.x, data, self.batch_size, self.lr, plot_fig)

        # save parameters
        if save_params:
            self.file_name= 'weights/1.3-3.pkl'
            f = file(self.file_name, 'wb')
            cPickle.dump(self.classifier.__getstate__(), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            print ("parameters saved")

    def predict(self, test_dataset, seq= False):
        if seq:
            return Sequemtial_cnn_Prediction(self.classifier, self.x, test_dataset, 1)
        else:
            return cnn_predict(self.classifier, self.x, test_dataset, self.batch_size)


    def tester_only(self, test_dataset,layer):
        return CNN_tester(self.classifier, self.x, test_dataset, self.batch_size, layer)

