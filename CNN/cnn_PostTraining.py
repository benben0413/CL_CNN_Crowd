import numpy
import sys
import time
from math import isnan

import matplotlib.pyplot as plt

from Optimizier.optimization import *
from helpers.utilities import SaveData

# from sklearn import preprocessing

def Postfit(classifier, x, data, batch_size, learning_rate, plot_fig= True, n_epochs=200):

    train_x = data
    train_y = np.zeros(1)

    train_set_x = theano.shared(numpy.asarray(train_x, dtype=theano.config.floatX), borrow=True)
    train_set_y = theano.shared(numpy.asarray(train_y, dtype='int32')) #[0:batch_size]


    y = T.ivector('y')
    index = T.lscalar()

    lambda_1 = 0.00
    lambda_2 = 1e-3

    NLL = classifier.layer9.negative_log_likelihood(y)
    L1 = T.sum(abs(classifier.layer0.W)) + T.sum(abs(classifier.layer1.W)) + T.sum(abs(classifier.layer2.W)) + T.sum(abs(classifier.layer3.W)) +\
         T.sum(abs(classifier.layer4.W)) + T.sum(abs(classifier.layer5.W)) + T.sum(abs(classifier.layer6.W)) + T.sum(abs(classifier.layer7.W))# +\
         # T.sum(abs(classifier.layer8.W))

    L2 = T.sum(T.sqr(classifier.layer0.W)) + T.sum(T.sqr(classifier.layer1.W)) + T.sum(T.sqr(classifier.layer2.W)) + T.sum(T.sqr(classifier.layer3.W)) +\
         T.sum(T.sqr(classifier.layer4.W)) + T.sum(T.sqr(classifier.layer5.W)) + T.sum(T.sqr(classifier.layer6.W)) + T.sum(T.sqr(classifier.layer7.W))#+\
         # T.sum(T.sqr(classifier.layer8.W))

    cost = NLL + lambda_1 * L1 + lambda_2 * L2
    # create a list of gradients for all model parameters
    grads = T.grad(cost, classifier.params)


    opt = gd_optimizer('nestrov', classifier.params)
    updates = opt.update_param(grads, classifier.params, learning_rate, momentum= 0.9)

    train_model = theano.function(
        inputs=[x, y],
        outputs=cost,
        updates=updates#,
        # givens={
        #     x: train_set_x,
        #     y: train_set_y
        # }
    )

    classifier_errorFN = classifier.layer9.errors(y) #p_y_given_x
    Train_model_loss = theano.function(
        inputs=[index],
        outputs=classifier_errorFN,
        givens={
            x: train_set_x,
            y: train_set_y
        },
        on_unused_input='ignore'
    )


    print '... training'

    start_time = time.clock()
    epoch = 0
    done_looping = False
    total_cost = numpy.zeros(n_epochs)
    total_train_err = numpy.zeros(n_epochs)

    # start training
    while (epoch < 50):
        epoch += 1
        avg_cost = []

        tr_cost = 0# train_model(train_set_x, train_set_y)
        train_loss = Train_model_loss(0)


        total_cost[epoch-1] = tr_cost
        total_train_err[epoch -1] = train_loss *100


        print(('epoch: (%i/%i);   '
               'training cost: %f;    '
               'training loss: %f;  '
               )
              %(epoch,n_epochs,
                tr_cost,
                train_loss*100,
                )
              )


    end_time = time.clock()
    print >> sys.stderr, ('The code ran for %.2fm' % ((end_time - start_time) / 60.))

    #//TODO move the plot to the class before to add test classification to the plot
    if plot_fig:
        plt.plot(total_cost,'r')
        plt.plot(total_train_err, 'b')
        plt.show()

        # total_lost_to_save = total_cost + total_train_err
        # SaveData(total_lost_to_save, 'weights/FCNN-noP2/loss_3.pkl')

