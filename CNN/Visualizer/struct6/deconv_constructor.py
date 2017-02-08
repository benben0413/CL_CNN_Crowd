import theano.tensor as T
import theano
import numpy as np
# from DeconvHelper import switchs

class DeConvLayer(object):

    def __init__(self, W, input, input_shape, subsample=(1, 1)):

        self.input = input

        kernel = theano.shared(W)

        deconv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
            output_grad=input,
            filters=kernel,
            input_shape=input_shape,
            filter_shape=W.shape,
            border_mode=(0, 0),
            subsample=subsample
        )

        self.output = T.nnet.relu(deconv_out)


class DeConv_structure(object):
    def __init__(self, Layer_inp, main_weights, frame_size, channel_no):

        bch, ch, row, col = frame_size

        W7 = main_weights[7]
        row = (row * 2) + 2
        col = (col * 2) + 2
        W7 = W7[channel_no].reshape(1, W7.shape[1], W7.shape[2], W7.shape[3])
        input_shape = (1, W7.shape[1], row , col)
        self.layer7 = DeConvLayer(W7, Layer_inp, input_shape, subsample=(2,2))

        W6 = main_weights[6]
        row = (row * 2) + 2
        col = (col * 2) + 2
        input_shape = (1, W6.shape[1], row, col)
        self.layer6 = DeConvLayer(W6, self.layer7.output, input_shape, subsample=(2,2))

        W5 = main_weights[5]
        input_shape = (1, W5.shape[1], row, col)
        self.layer5 = DeConvLayer(W5, self.layer6.output, input_shape)

        W4 = main_weights[4]
        row += 2
        col += 2
        input_shape = (1, W4.shape[1], row, col)
        self.layer4 = DeConvLayer(W4, self.layer5.output, input_shape)


        W3 = main_weights[3]
        row +=2
        col +=2
        input_shape = (1, W3.shape[1], row , col)
        self.layer3 = DeConvLayer(W3, self.layer4.output, input_shape)


        W2 = main_weights[2]
        row +=2
        col +=2
        input_shape = (1, W2.shape[1], row , col)
        self.layer2 = DeConvLayer(W2, self.layer3.output, input_shape)


        W1 = main_weights[1]
        row +=2
        col +=2
        input_shape = (1, W1.shape[1], row, col)
        self.layer1 = DeConvLayer(W1, self.layer2.output, input_shape)


        W0 = main_weights[0]
        row +=6
        col +=6
        input_shape = (1, W0.shape[1], row, col)
        self.layer0 = DeConvLayer(W0, self.layer1.output, input_shape)



class DeConv_structure_0(object):
    def __init__(self, Layer_inp, main_weights, frame_size, channel_no):

        bch, ch, row, col = frame_size
        row +=6
        col +=6
        if channel_no == -1:
            W0 = main_weights[0]
        else:
            W0 = main_weights[0]
            W0 = W0[channel_no].reshape(1, W0.shape[1], W0.shape[2], W0.shape[3])
        input_shape = (1, W0.shape[1], row , col)
        self.layer0 = DeConvLayer(W0, Layer_inp, input_shape)
#
#
#
# class DeConv_structure_1(object):
#     def __init__(self, Layer_inp, main_weights, frame_size, channel_no):
#
#         bch, ch, row, col = frame_size
#         row +=2
#         col +=2
#         if channel_no == -1:
#             W = main_weights[1]
#         else:
#             W = main_weights[1]
#             W = W[channel_no].reshape(1, W.shape[1], W.shape[2], W.shape[3])
#         input_shape = (1, W.shape[1], row , col)
#         self.layer1 = DeConvLayer(W, Layer_inp, input_shape)
#
#         W0 = main_weights[0]
#         row += 6
#         col += 6
#         input_shape = (1, W0.shape[1], row, col)
#         self.layer0 = DeConvLayer(W0, self.layer1.output, input_shape)