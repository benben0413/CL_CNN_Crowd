import Image
import numpy as np
# import cv2
import sys
from os import listdir
from os.path import isfile, join
import shutil

from CNN.CNN import *
from GUI.GUI_CNN import *
from CNN.Visualizer.complex1.visualize_optimized import *


def loadImages(ImagesList):
    ImagesList = ["/home/falmasri/Desktop/Datasets/Mixed2/" + s for s in ImagesList]
    Imgs = []
    for idx in range(len(ImagesList)):
        jpgfile = np.array(Image.open(ImagesList[idx]))
        Imgs.append(jpgfile)
    return np.array(Imgs)


def load_dataset(dataset_path):
    if (os.path.isfile(dataset_path)):
        print ("Loading dataset...")
        f = open(dataset_path, 'rb')
        ds = cPickle.load(f)
        f.close()
        return ds
    else:
        print "Dataset not found"
        sys.exit()

def callDataset(DS_selection,testing_folder= None, img_name = None):
    if DS_selection == 0:
        jpgfile = np.array(Image.open(join(testing_folder,img_name)))
        train_x = jpgfile.reshape(1, jpgfile.shape[0], jpgfile.shape[1], jpgfile.shape[2])
        train_y = 0
    if DS_selection == 1:
        dataset_path = 'Datasets/MixedFacesDataset.pkl'
        Train, Valid, Test = load_dataset(dataset_path)
        train_x, train_y = Train
    elif DS_selection == 2:
        dataset_path = 'Datasets/1028621-Mixed2.pkl'
        Train, Valid, Test = load_dataset(dataset_path)
        train_x, train_y = Train
        train_x = loadImages(train_x[0:500])
    return train_x,train_y

def TrainProcess():
    dataset_path = 'Datasets/MixedFacesDataset.pkl'
    # dataset_path = 'Datasets/data_0_1.pkl'
    Train, Valid, Test = load_dataset(dataset_path)
    tr_x, tr_y = Train
    Train = tr_x[:10000], tr_y[:10000]

    print Train[0].shape

    imageSize = [100, 100]
    testing_itr = 1
    epochs = 3000
    saving = True

    param_location = 'weights/FCNN-comp1'
    train_no = 50
    train_iter = 2

    # lr_seq = np.array(testing_itr)
    # learning_rate = np.arange(1, 10, 1) * 1e-4
    for i in range(testing_itr):
        learning_rate = (10 ** np.random.uniform(-4, -6))


        # print learning_rate[i]
        # for lr_idx in range(testing_itr):
        #     if lr_seq[lr_idx] == learning_rate:
        #         learning_rate = 10 ** np.random.uniform(-7, -8)
        #
        # while learning_rate > 3e-7:
        #     learning_rate = 10 ** np.random.uniform(-7, -8)
        #
        # lr_seq[i] = learning_rate

        print ("learning %d test" %i)
        if testing_itr > 1:
            saving = False
            epochs = 3

        cnn = CNN(param_location, [train_no, train_iter], learning_rate=1e-4, batch_size=50, epochs = epochs, imageSize = imageSize)
        cnn.fit([Train, Valid], save_params=saving, plot_fig= saving)
        test_error = cnn.predict(Test, False)
        print "Test loss: %2f   " %(test_error *100)

        # iter = Train[0].shape[0]/ 50
        # for i in range(iter):
        #     Data_x, data_y = Train[0][i * 50 : i * 50 + 50],Train[1][i * 50 : i * 50 + 50]
        #     test_error = cnn.predict(Data_x, True)
        #     for j in range(len(test_error)):
        #         if test_error[j] != data_y[j]:
        #             print "imageNo: %i;  org label: %i" %((i *50) +j, data_y[j])
        #             plt.imsave('tmp/%i(%i).png'%((i *50) +j, data_y[j]), Data_x[j])

        fig, ax1 = plt.subplots(1,2)


def Test():

    testing_folder = r'/home/falmasri/Desktop/Datasets/Tomorrowland Belgium 2016 cropped faces/4'
    # testing_folder = r'/home/falmasri/Downloads/BioID-FaceDatabase-V1.2/test/neg'
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
    for idx in range(1): #len(file_names)
        currentfile = join(testing_folder, file_names[idx])
        jpgfile = np.array(Image.open(currentfile))

        #convert grayscale to RGB
        if len(jpgfile.shape) <3:
            jpgfile = cv2.cvtColor(jpgfile, cv2.COLOR_GRAY2RGB)

        #the minimum size for this structure is 43 before the size became 0 after convolution and pooling
        if jpgfile.shape[0] or jpgfile.shape[1] < 30:
            requiredSize = 30, 30
            jpgfile = cv2.resize(jpgfile, requiredSize, interpolation=cv2.INTER_AREA)

        #prepare batch of size one to feed the network
        imageSize = [jpgfile.shape[0], jpgfile.shape[1]]
        Batched_jpgfile = jpgfile.reshape(1,jpgfile.shape[0], jpgfile.shape[1],3)

        params = r'weights/FCNN-noP2/zxxx.pkl'
        if jpgfile.shape[0] < 6000: #600
            cnn = CNN(params, batch_size=1, imageSize=imageSize)
            prediction =  cnn.predict(Batched_jpgfile, True)

            # print prediction.shape
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(Batched_jpgfile[0])
            # plt.subplot(122)
            # plt.imshow(prediction[0][0], cmap='gray')
            # plt.show()

            # print prediction

            # prediction = np.argmax(prediction, axis=1)[0]
            # print prediction

            # if predicted face move to pos otherwise to neg
            # if prediction == 1:
            #     dist = join(testing_folder,'pos')
            #     shutil.move(currentfile, join(dist, file_names[idx]))
            # else:
            #     dist = join(testing_folder, 'neg')
            #     shutil.move(currentfile, join(dist, file_names[idx]))

def visualizer(params_path,params, DS):
    jpgfile, labels = DS

    cnn = CNN(params_path, params, batch_size=jpgfile.shape[0], imageSize=[jpgfile.shape[1], jpgfile.shape[2]])

    print jpgfile.shape
    map, avg, max = Attention(cnn.classifier, cnn.x, cnn.nkerns, jpgfile, labels)

    return map, avg, max, jpgfile

def visualizer_allMaps(params_path,params, DS):
    jpgfile, labels = DS
    cnn = CNN(params_path, params, batch_size=jpgfile.shape[0], imageSize=[jpgfile.shape[1], jpgfile.shape[2]])
    allMaps = CeckAllMaps(cnn.classifier, cnn.x, jpgfile)
    return allMaps


if __name__ == "__main__":
    # testing_folder = r'/media/falmasri/14BAB2D2BAB2AF9A/frames_concert/Tomorrowland Belgium 2016 - Steve Aoki (10-19-2016 11-44-05 AM)'
    # img_name = r'Tomorrowland Belgium 2016 - Steve Aoki 0332.jpg'
    # img_name = r'Tomorrowland Belgium 2016 - Steve Aoki 0359.jpg'

    testing_folder = r'/home/falmasri/Desktop/sequential_detection'
    # img_name = r'0_Tomorrowland Belgium 2016 - Steve Aoki 063295.jpg'
    img_name = r'Voldemort.jpg'
    # img_name = r'Tomorrowland B.jpg'
    # img_name = r'Untitled.jpg'
    img_name = r'600-01042430n.jpg'
    # img_name = r'Tomorrowland Belgium 2016 - Steve Aoki 0422_1.jpg'
    # img_name = r'Tomorrowland Belgium 2016 - Steve Aoki 0295.jpg'
    # img_name = r'jpg-2.jpg'
    # img_name = 'man-wheat-field-5478762.jpg'
    # img_name = r'Tomorrowland Belgium 2016 - Steve Aoki 0302_1.jpg'
    # testing_folder = r'/media/falmasri/14BAB2D2BAB2AF9A/www_crowd_dataset/all_first_frame_upgrade'
    # img_name = r'001_1-16_142618281_q.jpg'

    ## This block to check images from DS
    # jpgfile,labels = callDataset(1, testing_folder, img_name)
    # DS = jpgfile, labels
    # for i in range(0,jpgfile.shape[0],1):
    #     Images = jpgfile[i*50 :(i*50)+50], labels[i*50 :(i*50)+50]
        # maps, avg, _ = visualizer('weights/FCNN-comp1',[30,2],Images)
        # for j in range(maps.shape[0]):
        #     plt.figure()
        #     plt.suptitle("avg %f, max %f" %(avg[j],maps[j,0].max()))
        #     plt.subplot(121)
        #     plt.imshow(maps[j,0])
        #     plt.subplot(122)
        #     plt.imshow(jpgfile[(i*50) + j])
        #     bin_map = maps[j, 0]
        #     bin_map[bin_map > 0] = 1
        #     bin_map[bin_map <= 0] = 0
        #     x, y = np.argwhere(bin_map == 1).T
        #     plt.scatter(y, x,s= 3, color='r')
        #     plt.savefig('imgs/comp1/struct3.1/%i.png' %((i * 50) + j))
        #     plt.close()
            # plt.show()


    # # This block to check single wide image
    # jpgfile,labels = callDataset(0, testing_folder, img_name)
    # # jpgfile = jpgfile[:,0:100,0:50,:]
    # DS = jpgfile, labels
    # maps, avg, max, _ = visualizer('weights/FCNN-comp1',[41,4],DS)
    # plt.figure()
    # plt.suptitle("avg %f, max %f" % (avg[0], maps[0, 0].max()))
    # plt.subplot(131)
    # plt.imshow(maps[0,0])
    # plt.subplot(132)
    # plt.imshow(jpgfile[0])
    # bin_map = maps[0, 0]
    # bin_map[bin_map > 0.5] = 1
    # bin_map[bin_map <= 0.5] = 0
    # x, y = np.argwhere(bin_map == 1).T
    # plt.scatter(y, x,s= 3, color='r')
    # plt.subplot(133)
    # blur = cv2.GaussianBlur(maps[0,0], (101, 101), 0)
    # plt.imshow(blur)
    # plt.show()
    #################

    TrainProcess()

    # Test()
