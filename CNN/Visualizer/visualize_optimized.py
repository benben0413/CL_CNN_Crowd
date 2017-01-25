import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import cv2
from deconv_constructor import *
# from DeconvHelper import switchs



def Attention_opt(classifier, x, nkerns, jpgfile):
    batch = jpgfile.reshape(1, jpgfile.shape[0], jpgfile.shape[1], 3)
    test_set_x = np.asarray(batch, dtype=theano.config.floatX)

    output_layer2 = classifier.layer2.convs_out
    Test_model = theano.function(
        inputs=[x],
        outputs=output_layer2,
    )
    Layer2_frames = Test_model(test_set_x)
    switches_2 = switchs(Layer2_frames, 'get', 3)
    # switches_2.reshape(1,switches_2.shape[0], switches_2.shape[1], switches_2.shape[2])

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
    re_layer6_frames = np.zeros(
        (layer6_frames.shape[0], layer6_frames.shape[1], layer6_frames.shape[2] + 1, layer6_frames.shape[3] + 1))
    for idx in range(re_layer6_frames.shape[1]):
        a = layer6_frames[0][idx]
        a = np.c_[a, np.zeros(layer6_frames.shape[3])]
        a = np.r_[a, np.zeros((1, layer6_frames.shape[2] + 1))]
        re_layer6_frames[0][idx] = a
    switches_6 = switchs(re_layer6_frames, 'get', 2)




    #backword pass
    output_layer7 = classifier.layer7.output  # layer9.inp #y_pred #
    Test_model = theano.function(
        inputs=[x],
        outputs=output_layer7,
    )

    layer7_frames = Test_model(test_set_x)

    main_weights = construct_main_weights(classifier)

    inp = T.tensor4('inp')
    layer_no = 0
    #frame shape is used to help the function to construct the unsampling shapes for each layer


    for i in xrange(layer7_frames.shape[1]):

        bloc_size = layer7_frames.shape[2] * layer7_frames.shape[3]
        bloc = np.zeros(bloc_size)
        bloc[layer7_frames[0][i].argmax()] = np.max(layer7_frames[0][i])
        bloc = bloc.reshape(layer7_frames.shape[2], layer7_frames.shape[3])

        layer7_frames[0][i] = bloc

        frame_size = layer7_frames.shape
        Layer_inp = layer7_frames[0][i].reshape(1, 1, frame_size[2], frame_size[3])
        Visualizer7 = DeConv_structure_7(inp, main_weights, frame_size, i)
        out_deconv = Visualizer7.layer7.output
        f = theano.function([inp], out_deconv)
        upsampled_7 = f(Layer_inp)
        unpooled_7 = switchs(upsampled_7, 'set', 2, switches_6)

        frame_size = unpooled_7.shape
        Visualizer6 = DeConv_structure_6(inp, main_weights, frame_size)
        out_deconv = Visualizer6.layer6.output
        f = theano.function([inp], out_deconv)
        upsampled_6 = f(unpooled_7)
        unpooled_6 = switchs(upsampled_6, 'set', 2, switches_5)

        frame_size = unpooled_6.shape
        Visualizer543 = DeConv_structure_543(inp, main_weights, frame_size)
        out_deconv = Visualizer543.layer3.output
        f = theano.function([inp], out_deconv)
        upsampled_3 = f(unpooled_6)
        unpooled_3 = switchs(upsampled_3, 'set', 3, switches_2)

        frame_size = unpooled_3.shape
        Visualizer210 = DeConv_structure_210(inp, main_weights, frame_size)
        out_deconv = Visualizer210.layer0.output
        f = theano.function([inp], out_deconv)
        upsampled_img = f(unpooled_3)

        # fname = "imgs/tester/l8/full_%i_g_8.png" % i
        plt.imshow(upsampled_img[0].transpose(1,2,0))
        plt.show()


def construct_main_weights(classifier):
    main_weights = []
    for idx in range(14,-1,-2):
        main_weights.append(classifier.params[idx].get_value())
    return main_weights


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
        return switches.reshape(1, switches.shape[0], switches.shape[1], switches.shape[2])




