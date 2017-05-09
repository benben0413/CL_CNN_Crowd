import numpy

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import theano
import theano.tensor as T


def cnn_predict(classifier,x, test_datasets, batch_size):

    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    index = T.lscalar()  # index to a [mini]batch

    RET = []


    test_x, test_y = test_datasets
    test_set_x = theano.shared(numpy.asarray(test_x, dtype=theano.config.floatX), borrow = True)
    test_set_y = theano.shared(numpy.asarray(test_y, dtype='int32'))

    n_test_batches = test_x.shape[0]
    n_test_batches //= batch_size


    output = classifier.predictor.errors(y)

    Test_model = theano.function(
        inputs = [index],
        outputs= output,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
            },
        on_unused_input='warn'
    )

    test_losses = [
        Test_model(i)
        for i in range(n_test_batches)
        ]
    test_score = numpy.mean(test_losses)

    return test_score

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