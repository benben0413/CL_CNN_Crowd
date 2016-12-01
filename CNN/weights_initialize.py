import numpy
import theano.tensor as T
import theano

def generate_weights(rng, filter_shape, fan_in, fan_out, activation, b_size):
    W_dist = 0
    if activation == "relu":
        if len(filter_shape)==4:

            # W_bound_2 = numpy.sqrt(2. / (fan_in + fan_out))
            # W_dist = numpy.random.randn(filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]) * W_bound_2

            #xavier
            W_bound_6 = numpy.sqrt(6. / (fan_in + fan_out))
            W_dist = rng.uniform(low=-W_bound_6, high=W_bound_6, size=filter_shape)




        else:
            # W_bound_2 = numpy.sqrt(2. / (filter_shape[0] + filter_shape[1]))
            # W_dist = numpy.random.randn(filter_shape[0], filter_shape[1]) * W_bound_2

            #xavier
            W_bound_6 = numpy.sqrt(6. / (filter_shape[0] + filter_shape[1]))
            W_dist = rng.uniform(
                        low=-W_bound_6,
                        high=W_bound_6,
                        size=(filter_shape[0], filter_shape[1])
                )


            if activation == T.nnet.sigmoid:
                W_dist *= 4

    elif activation == "softmax":
        #     value=numpy.zeros(
        #         (n_in, n_out),
        #         dtype=theano.config.floatX
        #     )

        W_bound_6 = numpy.sqrt(6. / (filter_shape[0] + filter_shape[1]))
        W_dist = rng.uniform(
                    low=-W_bound_6,
                    high=W_bound_6,
                    size=(filter_shape[0], filter_shape[1])
            )


    W_values = numpy.asarray(W_dist, dtype=theano.config.floatX)

    # the bias is a 1D tensor -- one bias per output feature map
    # initialize the baises b as a vector of n_out 0s
    b_values = numpy.zeros((b_size,), dtype=theano.config.floatX)

    # else:
    #     print "error"
    #     W_values = numpy.asarray(numpy.zeros(filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]))
    #     b_values = numpy.asarray(numpy.zeros(b_size))

    return W_values,b_values