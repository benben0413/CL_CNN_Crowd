import numpy
import theano
import theano.tensor as T

from cnn_structure import CNN_struct
import pickle as cPickle
from utils import tile_raster_images
import matplotlib.pyplot as plt
import PIL.Image as Image
import matplotlib.cm as cm


def cnn_predict(classifier,x, test_datasets, batch_size):

    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    index = T.lscalar()  # index to a [mini]batch


    # params_range = int(numpy.floor(len(numpy.array(test_datasets[0])) / batch_size))

    print "...."

    # construct_W_image(classifier.__getstate__())

    RET = []
    # for it in range(len(test_datasets)):
    #     test_data = test_datasets[it]
    #     N = len(test_data)

        # just zeroes
        # test_labels = T.cast(theano.shared(numpy.asarray(numpy.zeros(batch_size), dtype=theano.config.floatX)), 'int32')

    test_data = theano.shared(numpy.asarray(test_datasets, dtype=theano.config.floatX))

    output = classifier.layer8.get_y_pred()

    ppm = theano.function(
        inputs = [index],
        outputs= output,
        givens={
            x: test_data[index * batch_size: (index + 1) * batch_size]  # ,
            # y: test_labels
            },
        on_unused_input='warn'
    )
    # print ppm(0)
    return ppm(0)

        # # p : predictions, we need to take argmax, p is 3-dim: (# loop iterations x batch_size x 2)
        # p = [ppm(ii) for ii in    xrange(N / batch_size)]
        # # p_one = sum(p, [])
        # xd = numpy.asarray(p)
        # # print p
        # p = numpy.array(p).reshape((xd.shape[0] * xd.shape[1], 2))
        # p = numpy.argmax(p, axis=1)
        # p = p.astype(int)
        # RET.append(p)

    # visualizing images in each kernal
    # test_data = test_datasets[0]  # [500:1000]
    # test_data = theano.shared(numpy.asarray(test_data, dtype=theano.config.floatX))

    # kernals = classifier.layer2.get_conv_value()
    # visualize_imgs = theano.function(
    #     inputs=[index],
    #     outputs=kernals,
    #     givens={
    #         x: test_data[index: (index + 66)]
    #     }
    # )

    # flatted_representaions = classifier.get_flatted_params()
    # visualize_params = theano.function(
    #     inputs=[index],
    #     outputs=flatted_representaions,
    #     givens={
    #         x: test_data[index: (index + 500)]
    #     }
    # )

    # s = visualize_imgs(0)
    # visulaize_cov_images(s)

    # param_h = []
    #
    # for idx in xrange(params_range):
    #     param_h.append(visualize_params(idx * batch_size))
    # param_h.append(visualize_params(500))
    # param_h.append(visualize_params(1000))
    # param_h.append(visualize_params(1500))
    # # param_h.append(visualize_params(2000))
    # # param_h.append(visualize_params(2500))
    # # param_h.append(visualize_params(3000))
    # param_array = numpy.array(param_h).reshape(params_range * batch_size, 450)
    # param_h.append(visualize_params(1000))

    # visualize_flatted_representaion(param_array)

    # return RET, param_array


def visualize_flatted_representaion(h):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.subplot(1,2,1)

    # param_array[498:502,:] = 0
    plt.imshow(h.T)
    plt.gca().invert_yaxis()
    plt.show()
    # return 0


def visulaize_cov_images(s):
    """
    :param s: batch of images in their depth after convolution, constrained to wich layer by the request
    it saves the images in the img folder or plt them on the screen.
    :return:
    """
    # print s[0][0]
    # plt.imshow(s[0][0].reshape((40, 40)), cmap=cm.Greys_r)
    # plt.grid(False)
    # plt.show()
    for i in xrange(len(s[0])):
        #     # img = s[0][i].reshape((8, 8))
        #     # save_image(img,"%.gif" %i)
        plt.imshow(s[0][i].reshape((46, 46)), cmap=cm.Greys_r)
        plt.grid(False)
        filename = "img/%d.png" % i
        plt.savefig(filename)




def save_image(image, file_name):
    output_folder = "img/"
    file_to_save = output_folder + file_name
    image.save(file_to_save)