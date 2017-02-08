import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import cv2


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

    output = classifier.layer0.output
    Test_model = theano.function(
        inputs=[x],
        outputs=output,
    )

    layer_Fmaps = Test_model(test_set_x)
    layers_flatten = layer_Fmaps.flatten()
    binsZeros = layers_flatten[layers_flatten == 0]
    print ("zeros in Activation map: %.2f%%" %((float(binsZeros.shape[0]) / layers_flatten.shape[0]) *100))

    # tt = classifier.layer0.W.get_value()[0].T
    # conv_out = T.nnet.conv.conv2d(
    #     input=input,
    #     filters=classifier.layer0.W.T,
    #     filter_shape=(64,3,7,7),
    #     image_shape= output_grad_reshaped.shape,
    #     border_mode=(2,2)
    # )
    #
    # out = theano.function([output_grad_reshaped], conv_out)
    # print out
    #testing grad deconv2d

    fusued = np.zeros((226,226,3))
    for idx in range(layer_Fmaps.shape[1]):
        maps_no = idx
        bch, ch, row, col = layer_Fmaps.shape
        output_grad_reshaped = layer_Fmaps.reshape((-1, 1, row, col))
        output_grad_reshaped = output_grad_reshaped[maps_no].reshape(1,1,row,col)
        print ("activation maps size:   ", output_grad_reshaped.shape)
        # print output_grad_reshaped.shape
        input_shape = (1, 3, 226, 226)
        # print ("kernel size:    ", classifier.layer0.W.get_value().shape)

        W = classifier.layer0.W.get_value()[maps_no].reshape(1,3,7,7)
        kernel = theano.shared(W)
        print ("kernel size:    ", W.shape)

        inp = T.tensor4('inp')
        # deconv size o' = i + (k - 1), o= 220+(7-1) = 226
        deconv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
            output_grad = inp,
            filters=kernel,
            input_shape= input_shape,
            filter_shape=(1,3,7,7),
            border_mode=(0,0), #
            subsample=(1,1)
        )
        f = theano.function(
            inputs = [inp],
            outputs= deconv_out)

        f_out = f(output_grad_reshaped)
        deconved_relu =  T.nnet.relu(f_out)[0].transpose(1,2,0)
        print deconved_relu[7][6:10]
        deconved =  f_out[0].transpose(1,2,0)
        print "\n"
        # print deconved[7][6:10]
        print ("deconv image shape: ", deconved.shape)

        selected_deonv_relu = deconved_relu
        selected_deonv_relu[selected_deonv_relu > 0 ] = 1

        filtered_jpg = jpgfile * selected_deonv_relu
        print filtered_jpg[7][6:10]

        # plt.imshow(filtered_jpg)
        # plt.show()
        # cv2.imshow('img',filtered_jpg)
        # cv2.waitKey(0)

        # plt.imshow(classifier.layer0.W.get_value()[0].transpose(1,2,0))
        # plt.show()

        # print output_grad_reshaped


        # plt.hist(layers_flatten, bins=30)
        # plt.show()

        # plot kernel
        # t = kernel.get_value()[0].transpose(1,2,0)
        # print t.shape
        # plt.imshow(t)
        # plt.show()

        #plot original image, feature map, deconved image
        fname = "imgs/deconv/test_relu_deconv_%i.png" %idx
        img_test = layer_Fmaps[0][maps_no]
        plt.figure()
        plt.subplot(141)
        plt.imshow(jpgfile)
        plt.subplot(142)
        plt.imshow(img_test, cmap='gray')
        plt.subplot(143)
        plt.imshow(deconved)
        plt.subplot(144)
        plt.imshow(filtered_jpg)
        plt.savefig(fname)
        plt.close()

        # plt.show()
        fusued += deconved
    # plt.imshow(fusued)
    # plt.show()
    return fusued

    # print layer_Fmaps.reshape((-1, 1, row, col))[maps_no][0].shape


def Study2_L0(classifier, x, jpgfile):

    batch = jpgfile.reshape(1, jpgfile.shape[0], jpgfile.shape[1], 3)
    test_set_x = np.asarray(batch, dtype=theano.config.floatX)

    output_layer2 = classifier.layer2.convs_out
    Test_model = theano.function(
        inputs=[x],
        outputs=output_layer2,
    )
    Layer2_frames = Test_model(test_set_x)
    switches_2 = switchs(Layer2_frames, 'get', 3)


    output_layer5 = classifier.layer5.convs_out  # layer9.y_pred
    Test_model = theano.function(
        inputs=[x],
        outputs=output_layer5,
    )

    layer5_frames = Test_model(test_set_x)
    switches_5 = switchs(layer5_frames, 'get', 2)


    output_layer6 = classifier.layer6.convs_out  # layer9.y_pred
    Test_model = theano.function(
        inputs=[x],
        outputs=output_layer6,
    )

    layer6_frames = Test_model(test_set_x)
    re_layer6_frames = np.zeros((layer6_frames.shape[0],layer6_frames.shape[1],layer6_frames.shape[2]+1, layer6_frames.shape[3]+1))
    for idx in range(re_layer6_frames.shape[1]):
        a = layer6_frames[0][idx]
        a = np.c_[a, np.zeros(layer6_frames.shape[3])]
        a = np.r_[a, np.zeros((1, layer6_frames.shape[2]+1))]
        re_layer6_frames[0][idx] = a
    switches_6 = switchs(re_layer6_frames, 'get', 2)




    output_layer7 = classifier.layer7.output  # layer9.y_pred
    Test_model = theano.function(
        inputs=[x],
        outputs=output_layer7,
    )

    layer7_frames = Test_model(test_set_x)

    # for idx in range(upsampled_7.shape[1]):
    #     fname = "imgs/tester/l7/u_%i.png" % idx
    #     plt.imsave(fname,upsampled_7[0][idx], cmap='gray' )


    # for i in range(upsampled_3.shape[1]):
    #     plt.figure()
    #     plt.subplot(121)
    #     plt.imshow(upsampled_3[0][i], cmap='gray')
    #     plt.subplot(122)
    #     plt.imshow(unpooled_3[0][i], cmap='gray')
    #     plt.show()


    for i in xrange(layer7_frames.shape[1]):

        bloc_size = layer7_frames.shape[2] * layer7_frames.shape[3]
        bloc = np.zeros(bloc_size)
        bloc[layer7_frames[0][i].argmax()] = np.max(layer7_frames[0][i])
        bloc = bloc.reshape(layer7_frames.shape[2], layer7_frames.shape[3])

        layer7_frames[0][i] = bloc

        upsampled_7 = T.nnet.relu(deconv_L7(classifier, layer7_frames, i))
        unpooled_7 = switchs(upsampled_7, 'set', 2, switches_6)
        unpooled_7 = unpooled_7.reshape(1, unpooled_7.shape[0], unpooled_7.shape[1], unpooled_7.shape[2])

        upsampled_6 = T.nnet.relu(deconv_L6(classifier, unpooled_7, -1))
        unpooled_6 = switchs(upsampled_6, 'set', 2, switches_5)
        unpooled_6 = unpooled_6.reshape(1, unpooled_6.shape[0], unpooled_6.shape[1], unpooled_6.shape[2])

        upsampled_5 = T.nnet.relu(deconv_L5(classifier, unpooled_6, -1))
        upsampled_4 = T.nnet.relu(deconv_L4(classifier, upsampled_5, -1))
        upsampled_3 = T.nnet.relu(deconv_L3(classifier, upsampled_4, -1))
        unpooled_3 = switchs(upsampled_3, 'set', 3, switches_2)
        unpooled_3 = unpooled_3.reshape(1, unpooled_3.shape[0], unpooled_3.shape[1], unpooled_3.shape[2])

        upsampled_2 = T.nnet.relu(deconv_L2(classifier, unpooled_3,-1))
        upsampled_1 = T.nnet.relu(deconv_L1(classifier, upsampled_2,-1))
        upsampled_0 = T.nnet.relu(deconv_L0(classifier, upsampled_1,-1))

        fname = "imgs/tester/l8/full_%i_g_8.png" % i
        plt.imsave(fname,upsampled_0[0].transpose(1,2,0))
        plt.close()

        # if i == 31:
        #     img = upsampled_0[0].transpose(1,2,0)
        #     f = file('imgs/tester/img.pkl', 'wb')
        #     import pickle as cPickle
        #     cPickle.dump(img, f, protocol=cPickle.HIGHEST_PROTOCOL)
        #     f.close()
            # cv2.imshow('%i' %i, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            # cv2.waitKey(0)
        # cv2.imwrite(fname,upsampled_0[0].transpose(1,2,0))
        # plt.imshow(upsampled_0[0].transpose(1,2,0), cmap='gray')
        # plt.show()
        # fname = "imgs/deconv/xlg/L3_deconv_%i.png" % i
        # plt.figure()
        # plt.subplot(131)
        # plt.imshow(jpgfile)
        # plt.subplot(132)
        # plt.imshow(layer3_frames[0][i],cmap='gray')
        # plt.subplot(133)
        # plt.imshow(upsampled_0[0].transpose(1,2,0))
        # plt.savefig(fname)
        # # plt.show()
        # plt.close()


def deconv_L0(classifier, layer_Fmaps,layer_no):

    bch, ch, row, col = layer_Fmaps.shape
    if layer_no > -1:
        output_grad_reshaped = layer_Fmaps[0][layer_no].reshape(1,1,row,col)
    else:
        output_grad_reshaped = layer_Fmaps

    print ("activation maps size:   ", output_grad_reshaped.shape)

    # print output_grad_reshaped.shape

    print ("kernel size:    ", classifier.layer0.W.get_value().shape)
    W = classifier.layer0.W.get_value()
    input_shape = (1, 3, row+6, row+6)
    if layer_no > -1:
        W = W[layer_no].reshape(1, W.shape[1], W.shape[2], W.shape[3])
    else:
        W = classifier.layer0.W.get_value().reshape(ch,W.shape[1], W.shape[2], W.shape[3])
    kernel = theano.shared(W)
    print ("kernel size:    ", W.shape)

    inp = T.tensor4('inp')
    # deconv size o' = i + (k - 1), o= 220+(7-1) = 226
    deconv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
        output_grad = inp,
        filters=kernel,
        input_shape= input_shape,
        filter_shape=(ch,3,7,7),
        border_mode=(0,0), #
        subsample=(1,1)
    )
    f = theano.function(
        inputs = [inp],
        outputs= deconv_out)

    return f(output_grad_reshaped)

def deconv_L1(classifier, layer_Fmaps, layer_no):

    bch, ch, row, col = layer_Fmaps.shape
    if layer_no > -1:
        output_grad_reshaped = layer_Fmaps[0][layer_no].reshape(1,1,row,col)
    else:
        output_grad_reshaped = layer_Fmaps
    print ("selected activation maps size:   ", output_grad_reshaped.shape)

    print ("kernel size:    ", classifier.layer1.W.get_value().shape)
    W = classifier.layer1.W.get_value()
    input_shape = (1, W.shape[1], row+2, col+2)
    if layer_no > -1:
        W = W[layer_no].reshape(1,W.shape[1],W.shape[2],W.shape[3])
    else:
        W = W.reshape(ch, W.shape[1], W.shape[2], W.shape[3])
    kernel = theano.shared(W)
    print ("selected kernel size:    ", W.shape)


    inp = T.tensor4('inp')
    # deconv size o' = i + (k - 1), o= 220+(7-1) = 226
    deconv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
        output_grad = inp,
        filters=kernel,
        input_shape= input_shape,
        filter_shape= W.shape,
        border_mode=(0,0), #
        subsample=(1,1)
    )
    f = theano.function(
        inputs = [inp],
        outputs= deconv_out)

    return f(output_grad_reshaped)

def deconv_L2(classifier, layer_Fmaps, layer_no):

    bch, ch, row, col = layer_Fmaps.shape
    if layer_no > -1:
        output_grad_reshaped = layer_Fmaps[0][layer_no].reshape(1,1,row,col)
    else:
        output_grad_reshaped = layer_Fmaps
    print ("selected activation maps size:   ", output_grad_reshaped.shape)

    print ("kernel size:    ", classifier.layer2.W.get_value().shape)
    W = classifier.layer2.W.get_value()
    input_shape = (1, W.shape[1], row+2, col+2)
    if layer_no > -1:
        W = W[layer_no].reshape(1,W.shape[1],W.shape[2],W.shape[3])
    else:
        W = W.reshape(ch, W.shape[1], W.shape[2], W.shape[3])
    kernel = theano.shared(W)
    print ("selected kernel size:    ", W.shape)


    inp = T.tensor4('inp')
    # deconv size o' = i + (k - 1), o= 220+(7-1) = 226
    deconv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
        output_grad = inp,
        filters=kernel,
        input_shape= input_shape,
        filter_shape= W.shape,
        border_mode=(0,0), #
        subsample=(1,1)
    )
    f = theano.function(
        inputs = [inp],
        outputs= deconv_out)

    return f(output_grad_reshaped)

def deconv_L3(classifier, layer_Fmaps, layer_no):

    bch, ch, row, col = layer_Fmaps.shape
    if layer_no > -1:
        output_grad_reshaped = layer_Fmaps[0][layer_no].reshape(1,1,row,col)
    else:
        output_grad_reshaped = layer_Fmaps
    print ("selected activation maps size:   ", output_grad_reshaped.shape)

    print ("kernel size:    ", classifier.layer3.W.get_value().shape)
    W = classifier.layer3.W.get_value()
    input_shape = (1, W.shape[1], row+2, col+2)
    if layer_no > -1:
        W = W[layer_no].reshape(1,W.shape[1],W.shape[2],W.shape[3])
    else:
        W = W.reshape(ch, W.shape[1], W.shape[2], W.shape[3])
    kernel = theano.shared(W)
    print ("selected kernel size:    ", W.shape)

    inp = T.tensor4('inp')
    deconv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
        output_grad = inp,
        filters=kernel,
        input_shape= input_shape,
        filter_shape= W.shape,
        border_mode=(0,0),
        subsample=(1,1)
    )
    f = theano.function(
        inputs = [inp],
        outputs= deconv_out)

    return f(output_grad_reshaped)

def deconv_L4(classifier, layer_Fmaps, layer_no):

    bch, ch, row, col = layer_Fmaps.shape
    if layer_no > -1:
        output_grad_reshaped = layer_Fmaps[0][layer_no].reshape(1,1,row,col)
    else:
        output_grad_reshaped = layer_Fmaps
    print ("selected activation maps size:   ", output_grad_reshaped.shape)

    print ("kernel size:    ", classifier.layer4.W.get_value().shape)
    W = classifier.layer4.W.get_value()
    input_shape = (1, W.shape[1], row+2, col+2)
    if layer_no > -1:
        W = W[layer_no].reshape(1,W.shape[1],W.shape[2],W.shape[3])
    else:
        W = W.reshape(ch, W.shape[1], W.shape[2], W.shape[3])
    kernel = theano.shared(W)
    print ("selected kernel size:    ", W.shape)

    inp = T.tensor4('inp')
    deconv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
        output_grad = inp,
        filters=kernel,
        input_shape= input_shape,
        filter_shape= W.shape,
        border_mode=(0,0),
        subsample=(1,1)
    )
    f = theano.function(
        inputs = [inp],
        outputs= deconv_out)

    return f(output_grad_reshaped)

def deconv_L5(classifier, layer_Fmaps, layer_no):

    bch, ch, row, col = layer_Fmaps.shape
    if layer_no > -1:
        output_grad_reshaped = layer_Fmaps[0][layer_no].reshape(1,1,row,col)
    else:
        output_grad_reshaped = layer_Fmaps
    print ("selected activation maps size:   ", output_grad_reshaped.shape)

    print ("kernel size:    ", classifier.layer5.W.get_value().shape)
    W = classifier.layer5.W.get_value()
    input_shape = (1, W.shape[1], row+2, col+2)
    if layer_no > -1:
        W = W[layer_no].reshape(1,W.shape[1],W.shape[2],W.shape[3])
    else:
        W = W.reshape(ch, W.shape[1], W.shape[2], W.shape[3])
    kernel = theano.shared(W)
    print ("selected kernel size:    ", W.shape)

    inp = T.tensor4('inp')
    deconv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
        output_grad = inp,
        filters=kernel,
        input_shape= input_shape,
        filter_shape= W.shape,
        border_mode=(0,0),
        subsample=(1,1)
    )
    f = theano.function(
        inputs = [inp],
        outputs= deconv_out)

    return f(output_grad_reshaped)


def deconv_L6(classifier, layer_Fmaps, layer_no):

    bch, ch, row, col = layer_Fmaps.shape
    if layer_no > -1:
        output_grad_reshaped = layer_Fmaps[0][layer_no].reshape(1,1,row,col)
    else:
        output_grad_reshaped = layer_Fmaps
    print ("selected activation maps size:   ", output_grad_reshaped.shape)

    print ("kernel size:    ", classifier.layer6.W.get_value().shape)
    W = classifier.layer6.W.get_value()
    input_shape = (1, W.shape[1], row+2, col+2)
    if layer_no > -1:
        W = W[layer_no].reshape(1,W.shape[1],W.shape[2],W.shape[3])
    else:
        W = W.reshape(ch, W.shape[1], W.shape[2], W.shape[3])
    kernel = theano.shared(W)
    print ("selected kernel size:    ", W.shape)

    inp = T.tensor4('inp')
    deconv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
        output_grad = inp,
        filters=kernel,
        input_shape= input_shape,
        filter_shape= W.shape,
        border_mode=(0,0),
        subsample=(1,1)
    )
    f = theano.function(
        inputs = [inp],
        outputs= deconv_out)

    return f(output_grad_reshaped)


def deconv_L7(classifier, layer_Fmaps, layer_no):

    bch, ch, row, col = layer_Fmaps.shape
    if layer_no > -1:
        output_grad_reshaped = layer_Fmaps[0][layer_no].reshape(1,1,row,col)
    else:
        output_grad_reshaped = layer_Fmaps
    print ("selected activation maps size:   ", output_grad_reshaped.shape)

    print ("kernel size:    ", classifier.layer7.W.get_value().shape)
    W = classifier.layer7.W.get_value()
    input_shape = (1, W.shape[1], row, col)
    if layer_no > -1:
        W = W[layer_no].reshape(1,W.shape[1],W.shape[2],W.shape[3])
    else:
        W = W.reshape(ch, W.shape[1], W.shape[2], W.shape[3])
    kernel = theano.shared(W)
    print ("selected kernel size:    ", W.shape)

    inp = T.tensor4('inp')
    deconv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
        output_grad = inp,
        filters=kernel,
        input_shape= input_shape,
        filter_shape= W.shape,
        border_mode=(0,0),
        subsample=(1,1),
        filter_flip=False
    )
    f = theano.function(
        inputs = [inp],
        outputs= deconv_out)

    return f(output_grad_reshaped)

def switchs(layer_Fmaps, operation= 'get', step = 2, switches = None):
    sw_size = step * step
    if operation == 'get':
        switches = np.zeros((layer_Fmaps.shape[1],layer_Fmaps.shape[2],layer_Fmaps.shape[3]), dtype=theano.config.floatX)
        for idx in range(layer_Fmaps.shape[1]):
            layer = layer_Fmaps[0][idx]
            main_switch = np.zeros(layer.shape)
            for i in range(0,layer.shape[0],step):
                for j in range(0,layer.shape[1],step):
                    switch = np.zeros(sw_size)
                    ds = layer[i:i+step,j:j+step]
                    switch[ds.argmax()] = 1
                    switch = switch.reshape(step,step)
                    main_switch[i:i+step,j:j+step] = switch
            switches[idx] = main_switch
        return switches
    elif operation == 'set':
        for idx in range(switches.shape[0]):
            for i in range(0,switches.shape[1],step):
                for j in range(0,switches.shape[2],step):
                    loc = np.zeros(sw_size)
                    loc[switches[idx][i:i+step, j:j+step].argmax()] = layer_Fmaps[0][idx][i/step,j/step]
                    loc = loc.reshape(step,step)
                    switches[idx][i:i+step, j:j+step] = loc
        return switches


def Study3_L0(classifier, x, jpgfile):

    batch = jpgfile.reshape(1, jpgfile.shape[0], jpgfile.shape[1], 3)
    test_set_x = np.asarray(batch, dtype=theano.config.floatX)

    output = classifier.layer0.convs_out
    Test_model = theano.function(
        inputs=[x],
        outputs=output,
    )

    layer_Fmaps = Test_model(test_set_x)
    print layer_Fmaps.shape

    bch, ch, row, col = layer_Fmaps.shape
    output_grad_reshaped = layer_Fmaps.reshape((-1, 1, row, col))
    output_grad_reshaped = output_grad_reshaped[0].reshape(1,1,row,col)
    print ("activation maps size:   ", output_grad_reshaped.shape)

    # print output_grad_reshaped.shape
    input_shape = (1, 3, 226, 226)
    print ("kernel size:    ", classifier.layer0.W.get_value().shape)

    W = classifier.layer0.W.get_value()[0].reshape(1,3,7,7)
    kernel = theano.shared(W)
    print ("kernel size:    ", W.shape)

    inp = T.tensor4('inp')
    # deconv size o' = i + (k - 1), o= 220+(7-1) = 226
    deconv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
        output_grad = inp,
        filters=kernel,
        input_shape= input_shape,
        filter_shape=(1,3,7,7),
        border_mode=(0,0), #
        subsample=(1,1)
    )
    f = theano.function(
        inputs = [inp],
        outputs= deconv_out)

    deconved =  f(output_grad_reshaped)
    print ("deconv image shape: ", deconved.shape)
    # img = deconv_L0(classifier,deconved)#[0].reshape(226,226,3)
    # print img.shape

    plt.imshow(deconved[0].reshape(226,226,3))
    plt.show()


