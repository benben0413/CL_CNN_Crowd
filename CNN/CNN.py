import os.path
import pickle as cPickle
from os.path import isfile, join

# from Structures.complex1.cnn_structure import *
# from Structures.complex1.cnn_structure_2 import *
# from Structures.complex1.cnn_structure_3 import *
# from Structures.complex1.cnn_structure_4 import *
from Structures.complex1.cnn_structure_5 import *


from Sequemtial_cnn_Prediction import *
from cnn_Prediction import cnn_predict
from cnn_training import fit_predict
from helper import CNN_tester
from helpers.utilities import SaveData
from cnn_PostTraining import Postfit

class CNN:
    def __init__(self,file_location, training_detailes, learning_rate= 1e-5, batch_size = 50, epochs = 100, imageSize = [100, 100]):

        seed = 8000
        rng = numpy.random.RandomState(seed)
        # self.nkerns = [32, 64, 64, 128, 256, 256, 2, 2, 2, 2]
        # self.nkerns = [64, 64, 96, 96, 128, 128, 128]
        # self.nkerns = [64, 128, 128, 256, 512, 512, 512]
        # self.nkerns = [32, 64, 64, 96, 128, 128]

        self.nkerns = [1, 3, 16, 32, 24, 48, 96, 1]

        self.L_sizes, K_sizes = self.create_L_Sizes(imageSize)
        self.batch_size = batch_size
        self.x = T.tensor4('x')  # the data is presented as rasterized images
        self.lr = learning_rate
        self.epochs = epochs
        self.file_location = file_location
        self.training_detailes = training_detailes


        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'

        # construct the CNN class
        self.classifier = CNN_struct_comp5(
            rng=rng,
            input=self.x,
            nkerns=self.nkerns,
            batch_size=batch_size,
            image_size=imageSize,
            image_dimension=3,
            L_sizes= self.L_sizes,
            K_sizes = K_sizes
        )

        param_toRead = join(self.file_location, ('%i.%i.pkl' %(training_detailes[0],training_detailes[1])))
        if(os.path.isfile(param_toRead)):
            f = open(param_toRead, 'rb')
            params = cPickle.load(f)

            #ADD new generated layer between already trained layers
            # params.insert(2,self.classifier.layer7.params[1].get_value())
            # params.insert(2, self.classifier.layer7.params[0].get_value())

            self.classifier.__setstate__(params)
            f.close()
            print ("Parameters loaded")
        else:
            print "No trained parameteres found. network is initiated"

    def fit(self, data, save_params,plot_fig):
        if len(data) > 1:
            fit_predict(self.classifier, self.x, data, self.batch_size, self.lr, plot_fig, self.epochs, self.training_detailes)
        else:
            Postfit(self.classifier, self.x, data, self.batch_size, self.lr, plot_fig)
        # save parameters
        if save_params:
            param_toSave = join(self.file_location, (('%i.%i.pkl' %(self.training_detailes[0],self.training_detailes[1]+1))))
            SaveData(self.classifier.__getstate__(), param_toSave)

    def predict(self, test_dataset, seq= False):
        if seq:
            return Sequemtial_cnn_Prediction(self.classifier, self.x, test_dataset)
        else:
            return cnn_predict(self.classifier, self.x, test_dataset, self.batch_size)


    def tester_only(self, test_dataset,layer):
        return CNN_tester(self.classifier, self.x, test_dataset, self.batch_size, layer)

    def create_L_Sizes(self,image_size):
        L0_k = [1, 1]
        L1_k = [7, 7]
        L2_k = [3, 3]
        L3_k = [3, 3]
        L4_k = [1, 1]
        L5_k = [3, 3]
        L6_k = [5, 5]
        L7_k = [1, 1]

        L0_out_size = image_size
        L1_out_size = image_size
            # [int(numpy.floor(float(L0_out_size[0] - L1_k[0] + 1) / 2)),
            #             int(numpy.floor(float(L0_out_size[1] - L1_k[1] + 1) / 2))]

        L2_out_size = L1_out_size
        L3_out_size = L2_out_size
        L4_out_size = L3_out_size
        L5_out_size = L4_out_size
        L6_out_size =  L5_out_size
        # [int(numpy.ceil(float(L5_out_size[0] - L6_k[0] + 1) / 2)),
        #                 int(numpy.ceil(float(L5_out_size[1] - L6_k[1] + 1) / 2))]
        L7_out_size = L6_out_size

        return [L0_out_size, L1_out_size, L2_out_size, L3_out_size, L4_out_size, L5_out_size, L6_out_size, L7_out_size], [L0_k, L1_k, L2_k, L3_k, L4_k, L5_k, L6_k, L7_k]

