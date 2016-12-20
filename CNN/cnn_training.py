import numpy
import sys
import time
from math import isnan

import matplotlib.pyplot as plt

from Optimizier.optimization import *


# from sklearn import preprocessing

def fit_predict(classifier, x, data, batch_size, learning_rate, plot_fig= True, n_epochs=300):

    train, valid = data
    train_x, train_y = train
    valid_x, valid_y = valid

    n_train_batches = train_x.shape[0]
    n_valid_batches = valid_x.shape[0]
    n_train_batches //= batch_size
    n_valid_batches //=batch_size

    train_set_x = theano.shared(numpy.asarray(train_x[0:batch_size], dtype=theano.config.floatX), borrow=True)
    train_set_y = theano.shared(numpy.asarray(train_y[0:batch_size], dtype='int32'))

    valid_set_x = theano.shared(numpy.asarray(valid_x[0:batch_size], dtype=theano.config.floatX), borrow=True)
    valid_set_y = theano.shared(numpy.asarray(valid_y[0:batch_size], dtype='int32'))

    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    index = T.lscalar()  # index to a [mini]batch

    lambda_1 = 0.00
    lambda_2 = 1e-3

    NLL = classifier.layer9.negative_log_likelihood(y)
    L1 = T.sum(abs(classifier.layer0.W)) + T.sum(abs(classifier.layer1.W)) + T.sum(abs(classifier.layer2.W)) + T.sum(abs(classifier.layer3.W)) +\
         T.sum(abs(classifier.layer4.W)) + T.sum(abs(classifier.layer5.W)) + T.sum(abs(classifier.layer6.W)) + T.sum(abs(classifier.layer7.W))

    L2 = T.sum(T.sqr(classifier.layer0.W)) + T.sum(T.sqr(classifier.layer1.W)) + T.sum(T.sqr(classifier.layer2.W)) + T.sum(T.sqr(classifier.layer3.W)) + \
         T.sum(T.sqr(classifier.layer4.W)) + T.sum(T.sqr(classifier.layer5.W)) + T.sum(T.sqr(classifier.layer6.W)) + T.sum(T.sqr(classifier.layer7.W))

    cost = NLL + lambda_1 * L1 + lambda_2 * L2
    # create a list of gradients for all model parameters
    grads = T.grad(cost, classifier.params)


    opt = gd_optimizer('nestrov', classifier.params)
    updates = opt.update_param(grads, classifier.params, learning_rate, momentum= 0.9)

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.layer9.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )

    verifier_model = theano.function(
        inputs=[index],
        outputs=classifier.get_poll_out(),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )


    print '... training'

    start_time = time.clock()
    epoch = 0
    done_looping = False
    total_cost = numpy.zeros(n_epochs)
    total_valid_err = numpy.zeros(n_epochs)
    total_train_err = numpy.zeros(n_epochs)

    # print verifier_model(0)
    # sys.exit()

    # start training
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        avg_cost = []

        # layer6_weights_delta = verifier_model(0)[0][0]
        # print layer6_weights_delta
        for minibatch_idx in range(n_train_batches):

            # print ("batch_no: %d;    epoch: %d" %(minibatch_idx,epoch))
            data_x_updated,data_y_updated = update_train_sharedVal(train_x, train_y, minibatch_idx, batch_size)
            train_set_x.set_value(data_x_updated)
            train_set_y.set_value(data_y_updated)

            tr_cost = train_model(0)
            # temp = verifier_model(0)[0][0]
            # print np.array(temp, dtype=theano.config.floatX) - np.array(layer6_weights_delta, dtype=theano.config.floatX)
            # layer6_weights_delta = temp

            # print ("batch_no: %d;    epoch: %d;     tr_cost:%2f" %(minibatch_idx,epoch,tr_cost))
            # print verifier_model(0)
            if isnan(tr_cost):
                print "nan found in cost"
                sys.exit()

            avg_cost.append(tr_cost)

            # learning_rate = learning_rate * (1 / (1 + 0.001 * epoch))
            # print "learning rate: %f" %learning_rate
            # py,log_py = validate_model(0)
            # y_x, p_y = validate_model(0)

        this_training_cost = numpy.mean(avg_cost)

        this_validation_loss = []
        for idx_v in range(n_valid_batches):
            data_xv_updated, data_yv_updated = update_valid_sharedVal(valid_x, valid_y, idx_v, batch_size)
            valid_set_x.set_value(data_xv_updated)
            valid_set_y.set_value(data_yv_updated)
            valid_loss = validate_model(0)
            this_validation_loss.append(valid_loss)

        this_Training_loss = []
        for idx_t in range(n_train_batches):
            data_xt_updated, data_yt_updated = update_train_sharedVal(train_x, train_y, idx_t, batch_size)
            valid_set_x.set_value(data_xt_updated)
            valid_set_y.set_value(data_yt_updated)
            train_loss = validate_model(0)
            this_Training_loss.append(train_loss)

        # this_validation_loss = [validate_model(i) for i
        #                              in range(n_valid_batches)]

        total_validation_loss = numpy.mean(this_validation_loss)
        total_training_loss = numpy.mean(this_Training_loss)


        total_cost[epoch-1] = this_training_cost
        total_train_err[epoch -1] = total_training_loss *100
        total_valid_err[epoch -1] = total_validation_loss *100


        print(('epoch: (%i/%i);   '
               'training cost: %f;    '
               'training loss: %f;  '
               'validation error: %2f;    '
               'learning_rate: %.e')
              %(epoch,n_epochs,
                this_training_cost,
                total_training_loss*100,
                total_validation_loss * 100.,
                learning_rate)
              )

        # if epoch > 30:
        #     learning_rate = 1e-6 /2
        # if epoch > 20:
        #     learning_rate = 5e-7

        # if validation_losses == 0:
        #     break

        # if (epoch) % 5  == 0:
            # compute zero-one loss on validation set


    end_time = time.clock()
    print >> sys.stderr, ('The code ran for %.2fm' % ((end_time - start_time) / 60.))

    #//TODO add training classification beside the cost
    #//TODO move the plot to the class before to add test classification to the plot
    if plot_fig:
        plt.plot(total_cost,'r')
        plt.plot(total_train_err, 'b')
        plt.plot(total_valid_err,'g')
        plt.show()


def update_train_sharedVal(train_x, train_y, x, batch_size):
    st = x * batch_size
    end = st + batch_size
    # print ("start %d, end %d" %(st,end))
    return numpy.asarray(train_x[st:end], dtype=theano.config.floatX), numpy.asarray(train_y[st:end], dtype='int32')

def update_valid_sharedVal(valid_x, valid_y, x, batch_size):
    st = x * batch_size
    end = st + batch_size
    return numpy.asarray(valid_x[st:end], dtype=theano.config.floatX), numpy.asarray(valid_y[st:end], dtype='int32')




def print_weights_values(classifier):
    print "conv 0:"
    print classifier.__getstate__()[16][0][0]

    print "conv 3:"
    print classifier.__getstate__()[10][0][0]

    print "hid 1:"
    print classifier.__getstate__()[4][0]

    print "softmax "
    print classifier.__getstate__()[0][0]