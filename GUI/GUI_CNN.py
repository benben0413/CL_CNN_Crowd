import os.path
import pickle as cPickle

from  CNN.Structures.cnn_structure6 import *

from CNN.Sequemtial_cnn_Prediction import *
from CNN.cnn_Prediction import cnn_predict
from CNN.cnn_training import fit_predict
from CNN.helper import CNN_tester
from helpers.utilities import SaveData
from CNN.cnn_PostTraining import Postfit

class GUI_CNN:
    def __init__(self,filename, learning_rate= 1e-5, batch_size = 50, epochs = 100, imageSize = [100, 100]):

        seed = 8000
        rng = numpy.random.RandomState(seed)
        self.nkerns = [32, 64, 64, 128, 256, 2, 2, 2]


        self.L_sizes = self.create_L_Sizes(imageSize)
        self.batch_size = batch_size
        self.x = T.tensor4('x')  # the data is presented as rasterized images
        self.lr = learning_rate
        self.epochs = epochs

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'

        # construct the CNN class
        self.classifier = CNN_struct6(
            rng=rng,
            input=self.x,
            nkerns=self.nkerns,
            batch_size=batch_size,
            image_size=imageSize,
            image_dimension=3,
            L_sizes= self.L_sizes
        )

        self.file_name = filename
        if(os.path.isfile(filename)):
            f = open(filename, 'rb')
            params = cPickle.load(f)

            #ADD new generated layer between already trained layers
            # params.insert(2,self.classifier.layer7.params[1].get_value())
            # params.insert(2, self.classifier.layer7.params[0].get_value())

            self.classifier.__setstate__(params)
            f.close()
            print ("Parameters loaded")
        else:
            print "No trained parameteres exist. network is initiated"

    def fit(self, data, save_params,plot_fig ):
        if len(data) > 1:
            fit_predict(self.classifier, self.x, data, self.batch_size, self.lr, plot_fig, self.epochs)
        else:
            Postfit(self.classifier, self.x, data, self.batch_size, self.lr, plot_fig)
        # save parameters
        if save_params:
            self.file_name= 'weights/FCNN-noP3-2/1.3.pkl'
            SaveData(self.classifier.__getstate__(), self.file_name)

    def predict(self, test_dataset, seq= False):
        if seq:
            return Sequemtial_cnn_Prediction(self.classifier, self.x, test_dataset, 1)
        else:
            return cnn_predict(self.classifier, self.x, test_dataset, self.batch_size)


    def tester_only(self, test_dataset,layer):
        return CNN_tester(self.classifier, self.x, test_dataset, self.batch_size, layer)

    def create_L_Sizes(self,image_size):
        L0_k = [7, 7]
        L1_k = [3, 3]
        L2_k = [3, 3]
        L3_k = [3, 3]
        L4_k = [3, 3]
        L5_k = [1, 1]
        L6_k = [3, 3]
        L7_k = [3, 3]


        L0_out_size = [(image_size[0] - L0_k[0] + 1), (image_size[1] - L0_k[1] + 1)]
        L1_out_size = [(L0_out_size[0] - L1_k[0] + 1), (L0_out_size[1] - L1_k[1] + 1)]
        L2_out_size = [(L1_out_size[0] - L2_k[0] + 1), (L1_out_size[1] - L2_k[1] + 1)]
        L3_out_size = [(L2_out_size[0] - L3_k[0] + 1), (L2_out_size[1] - L3_k[1] + 1)]
        L4_out_size = [(L3_out_size[0] - L4_k[0] + 1), (L3_out_size[1] - L4_k[1] + 1)]
        L5_out_size = L4_out_size
        L6_out_size = [int(numpy.floor(float(L5_out_size[0] - L6_k[0] + 1) / 2)),
         int(numpy.floor(float(L5_out_size[1] - L6_k[1] + 1) / 2))]
        L7_out_size = [int(numpy.floor(float(L6_out_size[0] - L7_k[0] + 1) / 2)),
         int(numpy.floor(float(L6_out_size[1] - L7_k[1] + 1) / 2))]


        # print L0_out_size
        # print L1_out_size
        # print L2_out_size
        # print L3_out_size
        # print L4_out_size
        # print L5_out_size
        # print L6_out_size
        # print L7_out_size

        return [L0_out_size, L1_out_size, L2_out_size, L3_out_size, L4_out_size, L5_out_size, L6_out_size, L7_out_size] #

