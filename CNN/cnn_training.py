import os
import sys, getopt
import time
import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from math import isnan


# from sklearn import preprocessing
from cnn_structure import CNN_struct
import pickle as cPickle
from logistic_sgd import LogisticRegression
from cnn_Prediction import cnn_predict

def fit_predict(classifier, x, data, batch_size, learning_rate=0.0001, n_epochs=100):

    train, valid = data
    train_x, train_y = train
    valid_x, valid_y = valid

    n_train_batches = train_x.shape[0]
    n_valid_batches = valid_x.shape[0]
    n_train_batches //= batch_size
    n_valid_batches //=batch_size

    train_set_x = theano.shared(numpy.asarray(train_x[0:batch_size], dtype=theano.config.floatX), borrow=True)
    train_set_y = theano.shared(numpy.asarray(train_y[0:batch_size], dtype='int32'))

    valid_set_x = theano.shared(numpy.asarray(valid_x, dtype=theano.config.floatX), borrow=True)
    valid_set_y = theano.shared(numpy.asarray(valid_y, dtype='int32'))

    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    index = T.lscalar()  # index to a [mini]batch

    lambda_1 = 0.00
    lambda_2 = 0.001

    NLL = classifier.layer8.negative_log_likelihood(y)
    L1 = T.sum(abs(classifier.layer0.W)) + T.sum(abs(classifier.layer1.W)) + T.sum(abs(classifier.layer2.W)) + T.sum(abs(classifier.layer3.W)) +\
         T.sum(abs(classifier.layer4.W)) + T.sum(abs(classifier.layer5.W)) + T.sum(abs(classifier.layer6.W)) + T.sum(abs(classifier.layer7.W)) + \
         T.sum(abs(classifier.layer8.W))
    L2 = T.sum(T.sqr(classifier.layer0.W)) + T.sum(T.sqr(classifier.layer1.W)) + T.sum(T.sqr(classifier.layer2.W)) + T.sum(T.sqr(classifier.layer3.W)) + \
         T.sum(T.sqr(classifier.layer4.W)) + T.sum(T.sqr(classifier.layer5.W)) + T.sum(T.sqr(classifier.layer6.W)) + T.sum(T.sqr(classifier.layer7.W)) + \
         T.sum(T.sqr(classifier.layer8.W))

    cost = NLL + lambda_1 * L1 + lambda_2 * L2
    # create a list of gradients for all model parameters
    grads = T.grad(cost, classifier.params)
    grads_clipped = (T.clip(g, -2, 2) for g in grads)

    # specify how to update the parameters of the model as a list of (variable, update expression) pairs
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(classifier.params, grads)
        ]

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
        outputs=classifier.layer8.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )

    verifier_model = theano.function(
        inputs=[index],
        outputs=classifier.layer8.get_NLL(y),
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

    # print verifier_model(0)
    # sys.exit()

    # start training
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        avg_cost = []
        for minibatch_idx in range(n_train_batches):

            # print ("batch_no: %d;    epoch: %d" %(minibatch_idx,epoch))
            data_x_updated,data_y_updated = update_shared_vars(train_x, train_y, minibatch_idx,batch_size)


            train_set_x.set_value(data_x_updated)
            train_set_y.set_value(data_y_updated)
            tr_cost = train_model(0)

            # print ("batch_no: %d;    epoch: %d;     tr_cost:%2f" %(minibatch_idx,epoch,tr_cost))
            if isnan(tr_cost):
                print "nan found in cost"
                sys.exit()

            avg_cost.append(tr_cost)


            # learning_rate = learning_rate * (1 / (1 + 0.001 * epoch))
            # print "learning rate: %f" %learning_rate
            # py,log_py = validate_model(0)
            # y_x, p_y = validate_model(0)

        this_training_cost = numpy.mean(avg_cost)
        validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)


        total_cost[epoch] = this_training_cost
        total_valid_err = this_validation_loss

        print(('epoch: (%i/%i);   '
               'training cost: %f;    '
               'validation error: %2f;    ')
              %(epoch,n_epochs,numpy.mean(avg_cost),
                this_validation_loss * 100.))


        # if validation_losses == 0:
        #     break

        # if (epoch) % 5  == 0:
            # compute zero-one loss on validation set



    end_time = time.clock()
    print >> sys.stderr, ('The code ran for %.2fm' % ((end_time - start_time) / 60.))

    plt.plot(total_cost,'r')
    plt.plot(total_valid_err,'g')
    plt.show()


def update_shared_vars(train_x,train_y, x, batch_size):
    st = x * batch_size
    end = st + batch_size
    # print ("start %d, end %d" %(st,end))
    return numpy.asarray(train_x[st:end], dtype=theano.config.floatX), numpy.asarray(train_y[st:end], dtype='int32')

def print_weights_values(classifier):
    print "conv 0:"
    print classifier.__getstate__()[16][0][0]

    print "conv 3:"
    print classifier.__getstate__()[10][0][0]

    print "hid 1:"
    print classifier.__getstate__()[4][0]

    print "softmax "
    print classifier.__getstate__()[0][0]