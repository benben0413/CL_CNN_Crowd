import sys
from os import listdir
from os.path import isfile, join
import shutil
import Image
import cv2

from CNN.CNN import *
import numpy as np
import matplotlib.pyplot as plt

def Train():
    dataset_path = 'Datasets/TLand_2016_faces_13516pos_4279neg.pkl'

    if(os.path.isfile(dataset_path)):
        print ("Loading dataset...")
        f = open(dataset_path, 'rb')
        Train, Valid, Test = cPickle.load(f)
        f.close()
    else:
        print "Dataset not found"
        sys.exit()

    imageSize = [100, 100]
    testing = 1
    saving = True
    for i in range(testing):
        learning_rate = 10 ** np.random.uniform(-5, -8)
        print ("learning %d test" %i)
        cnn = CNN('weights/1.3-3.pkl', learning_rate= 1e-5, batch_size=50, imageSize = imageSize)
        if testing > 1:
            saving = False
        # cnn.fit([Train, Valid], save_params=saving, plot_fig= saving)

        test_error = cnn.predict(Test, False)
        print "Test loss: %2f   " %(test_error *100)

def Test():

    # testing_folder = r'/home/falmasri/Desktop/Tomorrowland Belgium 2016 cropped faces/2'
    testing_folder = r'/home/falmasri/Downloads/BioID-FaceDatabase-V1.2/test/neg'
    if not os.path.exists(testing_folder):
        print ("Testing folder not exist")
        sys.exit()

    # if not os.path.exists(join(testing_folder,'pos')):
    #     os.makedirs(join(testing_folder,'pos'))
    # if not os.path.exists(join(testing_folder,'neg')):
    #     os.makedirs(join(testing_folder,'neg'))

    def read_files(folderPath):
        onlyfiles = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
        return onlyfiles

    file_names = read_files(testing_folder)
    for idx in range(len(file_names)): #
        currentfile = join(testing_folder, file_names[idx])
        jpgfile = np.array(Image.open(currentfile))

        #convert grayscale to RGB
        if len(jpgfile.shape) <3:
            jpgfile = cv2.cvtColor(jpgfile, cv2.COLOR_GRAY2RGB)
        # plt.imshow(jpgfile)
        # plt.show()
        # print jpgfile.shape
        #the minimum size for this structure is 43 before the size became 0 after convolution and pooling
        if jpgfile.shape[0] or jpgfile.shape[1] < 43:
            requiredSize = 43, 43
            jpgfile = cv2.resize(jpgfile, requiredSize, interpolation=cv2.INTER_AREA)

        #prepare batch of size one to feed the network
        imageSize = [jpgfile.shape[0], jpgfile.shape[1]]
        Batched_jpgfile = jpgfile.reshape(1,jpgfile.shape[0], jpgfile.shape[1],3)

        if jpgfile.shape[0] < 600:
            cnn = CNN('weights/1.3-2.pkl', batch_size=1, imageSize=imageSize)

            prediction =  cnn.predict(Batched_jpgfile, True)
            print prediction

            prediction = np.argmax(prediction, axis=1)[0]
            print prediction

            #if predicted face move to pos otherwise to neg
            # if prediction == 1:
            #     dist = join(testing_folder,'pos')
            #     shutil.move(currentfile, join(dist, file_names[idx]))
            # else:
            #     dist = join(testing_folder, 'neg')
            #     shutil.move(currentfile, join(dist, file_names[idx]))



Test()