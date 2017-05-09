import numpy

import theano
import theano.tensor as T
from CNN.Optimizier.weights_initialize import *
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(inumpyut,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inumpyut: theano.tensor.dmatrix
        :param inumpyut: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of inumpyut

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.inp = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_shape = (n_in,n_out)
            w_values, b_values = generate_weights(rng, W_shape, 0, 0, 'relu', n_out)
            W = theano.shared(value=w_values, name='W', borrow=True)
            b = theano.shared(value=b_values, name='b', borrow=True)


        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

    def get_h_out(self):
        return self.output



class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, subsample=(1,1), poolsize=(2, 2), pool_flag = False, ig_border = True):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))

        # initialize weights with random weights and biases
        w_values, b_values = generate_weights(rng, filter_shape, fan_in, fan_out, 'relu', filter_shape[0])

        self.W = theano.shared(w_values, borrow=True )

        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            input_shape=image_shape,
            filter_shape=filter_shape,
            subsample=subsample
            # ,border_mode='same'
        )
        self.rectified_conv_out = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = self.rectified_conv_out

        # downsample each feature map individually, using maxpooling
        #downsample.max_pool_2d
        if pool_flag:
            pooled_out = pool.pool_2d(
                input= self.rectified_conv_out,
                ds=poolsize,
                ignore_border=ig_border
            )
            self.output = pooled_out

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        # self.output = pooled_out #T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        # apply RELU after pooling to get over negative results.
        # self.output = T.nnet.relu(self.transformed + self.b.dimshuffle('x', 0, 'x', 'x'))
        # store parameters of this layer
        self.params = [self.W, self.b]

        self.convs_out = conv_out

    def get_conv_value(self):
        # self.input = self.input.reshape((1, 1, 28, 28))
        # pp = T.dot(self.input, self.W) + self.b
        return self.convs_out, self.convs_out + self.b.dimshuffle('x', 0, 'x', 'x'), self.output
        # return T.nnet.relu(self.convs_out)


class AvgPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, input, image_shape):
        """
        Allocate a PoolLayer.

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        """

        self.input = input


        # downsample each feature map individually, using maxpooling
        #downsample.max_pool_2d

        poolsize = (image_shape[0], image_shape[1])
        total_cells = image_shape[0] * image_shape[1]
        pooled_out = pool.pool_2d(
            input= self.input,
            ws=poolsize,
            ignore_border=True,
            mode= 'average_exc_pad'
        )
        self.output = pooled_out #* total_cells
