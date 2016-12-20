import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T



def construct_L0(W0):
    shape = W0.get_value().shape
    Layers = W0.get_value().reshape(shape[0],shape[2],shape[3],shape[1])
    for idx in range(shape[0]):
        Img = Layers[idx]
        plt.imshow(Img)
        plt.show()


def Study_L0(classifier, x, jpgfile):

    batch = jpgfile.reshape(1, jpgfile.shape[0], jpgfile.shape[1], 3)
    test_set_x = np.asarray(batch, dtype=theano.config.floatX)
    print test_set_x.shape

    output = classifier.layer0.output
    Test_model = theano.function(
        inputs=[x],
        outputs=output,
    )

    layer_Fmaps = Test_model(test_set_x)
    print layer_Fmaps.shape
    layers_flatten = layer_Fmaps.flatten()
    binsZeros = layers_flatten[layers_flatten == 0]
    print ("zeros in Activation map: %.2f%%" %((float(binsZeros.shape[0]) / layers_flatten.shape[0]) *100))
    # plt.hist(layers_flatten, bins=30)
    # plt.show()

    # # plot original Image and one feature map from the layer
    # img_test = layer_Fmaps[0][0]
    # print img_test.shape
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(jpgfile)
    # plt.subplot(122)
    # plt.imshow(img_test, cmap='gray')
    # plt.show()
    # print layer_Fmaps.shape