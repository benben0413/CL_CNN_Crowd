import theano.tensor as T
from theano.tensor.nnet.nnet import binary_crossentropy
import numpy as np

class CrossEntropy_classifier(object):
    def __init__(self, input1, input2):

        self.input1 = input1
        self.input2 = input2

        # self.y_pred = T.cast(self.input, 'int32') #
        # self.y_pred = self.input
        # self.y_pred = T.set_subtensor(self.y_pred[self.y_pred >= 0.7], 1)
        # self.y_pred = T.set_subtensor(self.y_pred[self.y_pred < 0.7], 0)

        # idxs = (self.y_pred >= 0.5).nonzero()
        # self.y_pred = T.set_subtensor(self.y_pred[idxs], 1)
        # idxs = (self.y_pred < 0.5).nonzero()
        # self.y_pred = T.set_subtensor(self.y_pred[idxs], 0)

        self.y_pred = T.switch(T.lt(input1, 0.7), 0 , 1)

        # self.sig_input =

    def CrossEntropy(self, y):

        # return binary_crossentropy(self.input, y).mean()
        # return -T.mean( y * T.log(self.input[T.arange((y.shape[0]),0)])+ (1-y) * T.log(1 - self.input[T.arange((y.shape[0]),0)]))

        # return -T.mean(T.log(self.input)[T.arange(y.shape[0]), y])
        return -T.mean(y * T.log(self.input1 + 1e-5) + (1-y) * T.log(1- self.input1 + 1e-5))

    def MSE(self, y):

        return T.mean((y - self.input1) ** 2) #,((y - self.input) ** 2), self.y_pred

    def MSE2(self, y):
        loc = T.eq(y,1).nonzero()[0]
        S = T.clip(self.input2[loc],0,1)
        self.input2 = T.set_subtensor(self.input2[loc], S)
        return T.mean((y - self.input2) ** 2)

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        # if y.ndim != self.input.ndim:
        #     raise TypeError(
        #         'y should have the same shape as self.input',
        #         ('y', y.type, 'y_pred', self.y_pred.type)
        #     )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))  # , self.y_pred
        else:
            raise NotImplementedError()