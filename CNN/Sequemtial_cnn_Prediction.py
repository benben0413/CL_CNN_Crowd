import numpy

import theano
import theano.tensor as T


def Sequemtial_cnn_Prediction(classifier,x, test_datasets, batch_size):

    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    index = T.lscalar()  # index to a [mini]batch

    RET = []


    test_x = test_datasets
    test_set_x = theano.shared(numpy.asarray(test_x, dtype=theano.config.floatX), borrow = True)

    n_test_batches = test_x.shape[0]
    n_test_batches //= batch_size


    output = classifier.layer9.get_p_y() #get_inp() #get_poll_out()

    Test_model = theano.function(
        inputs = [index],
        outputs= output,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
            },
        on_unused_input='warn'
    )


    return Test_model(0)
