import matplotlib.pyplot as plt
plt.style.use('ggplot')
import theano
from CNN.Visualizer.struct4.deconv_constructor import *
import cv2
from mpl_toolkits.mplot3d.axes3d import Axes3D
import plotly.plotly as py
import plotly.graph_objs as go
import pyqtgraph as pg
import sys


def Attention(classifier, x, nkerns, jpgfile, y):
    inp = T.tensor4('inp')
    # out_label = T.dvector('out_label')
    Forward_Pass = ForwardPass(classifier,x)
    # Classify_pass = Classify(classifier,x)
    main_weights = construct_main_weights(classifier, nkerns)

    maps, avg, max  = Forward_Pass(jpgfile) #confidence,classed
    # print conc2.shape
    # for i in range(conc2.shape[1]):
    #     plt.imshow(conc2[0,i])
    #     plt.show()
    # train_set_y = theano.shared(np.asarray(y, dtype='int32'))
    # cost = Classify_pass(jpgfile,train_set_y)
    # print cost

    # print "flatted shape "
    # print flatted.shape
    # print Levels.shape

    # print map.shape

    #print the Fmap into file
    # for idx in range(maps.shape[2]):
    #     f = open('/home/falmasri/Desktop/test.txt', 'a')
    #     print >> f, maps[0,0,idx]
    #     f.close()
    # flatted_map = maps[0,0].flatten()
    # plt.hist(flatted_map,100)
    # plt.show()


    # fig = plt.figure()
    # x_axis = range(710)
    # y_axis = range(951)
    # X, Y = np.meshgrid(y_axis, x_axis)
    # ax = Axes3D(fig)
    # ax.scatter(X, Y, maps[0, 0])
    # plt.show()

    # view = pg.GraphicsLayoutWidget()
    # view.show()
    # w1 = view.addPlot()
    # w1.plot(maps[0,0])

    #plot historam
    # bins = maps[0,0]
    # activated = bins[bins > 0]
    # print len(activated)
    # plt.hist(activated,10)
    # plt.show()
    # sys.exit()

    return maps,avg, max

    # for i in range(maps.shape[0]):
    #     print "average is %f: " %avg[i]
    #     # print map[i, 0,39:61,57:72]
    #     # print map[i, 0, 114:118, 64:70]
    #     # np.savetxt("/home/falmasri/Desktop/22.1-tosend.csv", maps[i,0], delimiter=",")
    #
    #     plt.imshow(maps[i, 0])
    #
    #     # py.iplot(maps[i,0])
    #     plt.figure()
    #     # plt.subplot(121)
    #     plt.imshow(jpgfile[i])
    #
    #     bin_map = maps[i, 0]
    #     bin_map[bin_map > 0] = 1
    #     bin_map[bin_map <= 0] = 0
    #
    #     # plt.matshow(bin_map)
    #     x, y = np.argwhere(bin_map == 1).T
    #     plt.scatter(y, x)
    #
    #     plt.figure()
    #     blur = cv2.GaussianBlur(maps[i,0], (101, 101), 0)
    #     plt.imshow(blur)
    #
    #     plt.show()

        # plt.figure()
        # plt.gca().invert_yaxis()
        # plt.subplot(132)
        # plt.imshow(Levels[i][0], cmap='gray')
        # plt.subplot(132)
        # plt.imshow(maps[i][0], cmap='gray')
        # plt.subplot(122)
        # plt.imshow(bin_map)
        # plt.show()

        # plt.savefig('imgs/comp1/struct2.2/%i_2.png' %(i))
        # plt.close()
        # cv2.imshow('test',maps[i][0])
        # cv2.waitKey(0)
    # return 0

def CeckAllMaps(classifier, x, jpgfile):
    Pass = ForwardPass_maps(classifier, x)
    AllMaps = Pass(jpgfile)
    return AllMaps


def test(inp, layer0_frames, main_weights, batch):

    frame_size = layer0_frames.shape
    Layer_inp = layer0_frames[0][6].reshape(1, 1, frame_size[2], frame_size[3])
    Visualizer = DeConv_structure_0(inp, main_weights, frame_size, 4)
    out_deconv = Visualizer.layer0.output
    f = theano.function([inp], out_deconv)
    upsampled_img = f(Layer_inp)

    added_img = np.add(np.add(upsampled_img[0][0],upsampled_img[0][1]),upsampled_img[0][2])
    np.putmask(added_img, added_img > 0, 1)

    new_img = np.array(np.multiply(batch[0].transpose(2,0,1), added_img), dtype=np.uint8).transpose(1,2,0)
    return new_img

def ForwardPass(classifier,x):
    output = classifier.get_intermediate_sides()
    Test_model = theano.function(
        inputs=[x],
        outputs=output,
    )

    return Test_model

def ForwardPass_maps(classifier,x):
    output = classifier.get_intermediate_sides_all()
    Test_model = theano.function(
        inputs=[x],
        outputs=output,
    )

    return Test_model

# def Classify(classifier,x):
#     y = T.ivector('y')
#     output = classifier.predictor.CrossEntropy(y)  # layer9.inp #y_pred #
#     Test_model = theano.function(
#         inputs=[x],
#         outputs=output,
#     )
#
#     return Test_model

def construct_main_weights(classifier,nkerns):
    no_params = len(nkerns) *2
    main_weights = []
    for idx in range(no_params,-1,-2):
        main_weights.append(classifier.params[idx].get_value())
    return main_weights

def printFmaps(lmaps, fname):
    for i in range(lmaps.shape[1]):
        plt.imsave(fname %i, lmaps[0][i],cmap='gray')
        plt.close()


