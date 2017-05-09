import numpy
import sys
import time
from math import isnan
from os.path import join

import matplotlib.pyplot as plt

from Optimizier.optimization import *
from helpers.utilities import SaveData
import Image

# from sklearn import preprocessing

def fit_predict(classifier, x, data, batch_size, learning_rate, plot_fig= True, n_epochs=200, training_detailes=[0,0]):

    train, valid = data
    train_x, train_y = train
    train_y = numpy.asarray(train_y)
    valid_x, valid_y = valid

    n_train_batches = train_x.shape[0]
    n_valid_batches = valid_x.shape[0]
    n_train_batches //= batch_size
    n_valid_batches //=batch_size

    # data_x = loadImages(train_x[0:batch_size])
    train_set_x = theano.shared(numpy.asarray(train_x, dtype=theano.config.floatX), borrow=True)
    train_set_y = theano.shared(numpy.asarray(train_y, dtype='int32')) #[0:batch_size]

    # data_x = loadImages(valid_x[0:batch_size])
    valid_set_x = theano.shared(numpy.asarray(valid_x, dtype=theano.config.floatX), borrow=True)
    valid_set_y = theano.shared(numpy.asarray(valid_y, dtype='int32')) #[0:batch_size]

    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    index = T.lscalar()  # index to a [mini]batch

    lambda_1 = 0.00
    lambda_2 = 0.00 #1e-3

    NLL = classifier.predictor.MSE(y) #CrossEntropy(y) #negative_log_likelihood2(y) #negative_log_likelihood(y)
    NLL2 = classifier.predictor.MSE2(y)
    # L1 = T.sum(abs(classifier.layer0.W)) + T.sum(abs(classifier.layer1.W))
    #
    # L2 = T.sum(T.sqr(classifier.bloc1.W)) + T.sum(T.sqr(classifier.bloc2_0.W))

    L1 = 0
    L2 = 0
    cost = NLL + NLL2 + lambda_1 * L1 + lambda_2 * L2
    # create a list of gradients for all model parameters
    grads = T.grad(cost, classifier.params)


    opt = gd_optimizer('sgd', classifier.params)
    updates = opt.update_param(grads, classifier.params, learning_rate, momentum= 0.9)

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )

    classifier_errorFN = classifier.predictor.errors(y)
    validate_model = theano.function(
        inputs=[index],
        outputs= classifier_errorFN,
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )

    Train_model_loss = theano.function(
        inputs=[index],
        outputs=classifier_errorFN,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )

    # theano.printing.debugprint(train_model)
    # updates_2 = opt.tester(grads,classifier.params, learning_rate)
    # veri = grads
    veri = classifier.struct_tester()

    # verifier_model = theano.function(
    #     inputs=[index],
    #     outputs=y_pred,
    #     givens={
    #         x: train_set_x[index * batch_size: (index + 1) * batch_size],
    #         y: train_set_y[index * batch_size: (index + 1) * batch_size]
    #     },
    #     on_unused_input='ignore'
    # )

    verifier_model2 = theano.function(
        inputs=[index],
        outputs=veri,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )

    # print verifier_model2(0).shape
    # sys.exit()

    print '... training'

    log_file_name = 'weights/FCNN-comp1/logs_%i.%i.txt' % (training_detailes[0], training_detailes[1] + 1)
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


        # //TODO Add function load images from the name lists Dataset
        for minibatch_idx in range(n_train_batches):

            # print ("batch_no: %d;    epoch: %d" %(minibatch_idx,epoch))

            # data_x_updated, data_y_updated = Update_dataset_from_Images(train_x, train_y, minibatch_idx, batch_size)
            # data_x_updated,data_y_updated = update_train_sharedVal(train_x, train_y, minibatch_idx, batch_size)
            # train_set_x.set_value(data_x_updated)
            # train_set_y.set_value(data_y_updated)

            # print verifier_model2(0)
            tr_cost = train_model(minibatch_idx)
            # print tr_cost
            # sys.exit()


            if isnan(tr_cost):
                print "nan found in cost"
                sys.exit()

            avg_cost.append(tr_cost)

            # if epoch >= 800:
            #     learning_rate = 2e-6

            # learning_rate = learning_rate * (1 / (1 + 0.001 * epoch))
            # print "learning rate: %f" %learning_rate

        this_training_cost = numpy.mean(avg_cost)

        this_validation_loss = []
        for idx_v in range(n_valid_batches):
            # data_xv_updated, data_yv_updated = Update_dataset_from_Images(valid_x, valid_y, idx_v, batch_size)
            # data_xv_updated, data_yv_updated = update_valid_sharedVal(valid_x, valid_y, idx_v, batch_size)
            # valid_set_x.set_value(data_xv_updated)
            # valid_set_y.set_value(data_yv_updated)
            valid_loss = validate_model(idx_v) #0
            this_validation_loss.append(valid_loss)

            # valid_loss = validate_model(idx_v)
            # this_validation_loss.append(valid_loss)

        this_Training_loss = []
        for idx_t in range(n_train_batches):
            # data_xt_updated, data_yt_updated = update_train_sharedVal(train_x, train_y, idx_t, batch_size)
            # valid_set_x.set_value(data_xt_updated)
            # valid_set_y.set_value(data_yt_updated)

            train_loss = Train_model_loss(idx_t) #0
            this_Training_loss.append(train_loss)


        # this_validation_loss = [validate_model(i) for i
        #                              in range(n_valid_batches)]

        total_validation_loss = numpy.mean(this_validation_loss)
        total_training_loss = numpy.mean(this_Training_loss)


        total_cost[epoch-1] = this_training_cost
        total_train_err[epoch -1] = total_training_loss *100
        total_valid_err[epoch -1] = total_validation_loss *100

        # 'training loss: %f;  ' \  total_training_loss*100,
        logs = 'epoch: (%i/%i);   '    \
               'training cost: %f;    ' \
               'training loss: %f;  ' \
               'validation error: %2f;    ' \
               'learning_rate: %.e'    \
              %(epoch,n_epochs,
                this_training_cost,
                total_training_loss * 100,
                total_validation_loss * 100,
                learning_rate)

        print (logs)

        f = open(log_file_name, 'a')
        print >> f, logs
        f.close()

        print "save parameters for this epoch"
        param_toSave = join('weights/FCNN-comp1',(('%i.%i.pkl' % (training_detailes[0], training_detailes[1] + 1))))
        SaveData(classifier.__getstate__(), param_toSave)


    end_time = time.clock()
    timing_log = ('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    print >> sys.stderr, timing_log
    f = open(log_file_name, 'a')
    print >> f, timing_log
    f.close()

    #//TODO move the plot to the class before to add test classification to the plot
    if plot_fig:
        plt.figure()
        plt.subplot(211)
        plt.plot(total_cost,'r')
        plt.subplot(212)
        # plt.plot(total_train_err, 'b')
        plt.plot(total_valid_err,'g')
        plt.show()

        total_lost_to_save = total_cost + total_train_err + total_valid_err
        SaveData(total_lost_to_save, 'weights/FCNN-comp1/loss_%i.%i.pkl' %(training_detailes[0], training_detailes[1] + 1))



def update_train_sharedVal(train_x, train_y, x, batch_size):
    st = x * batch_size
    end = st + batch_size
    return numpy.asarray(train_x[st:end], dtype=theano.config.floatX), numpy.asarray(train_y[st:end], dtype='int32')

def update_valid_sharedVal(valid_x, valid_y, x, batch_size):
    st = x * batch_size
    end = st + batch_size
    return numpy.asarray(valid_x[st:end], dtype=theano.config.floatX), numpy.asarray(valid_y[st:end], dtype='int32')

def loadImages(ImagesList):
    ImagesList = ["/home/falmasri/Desktop/Datasets/Mixed2/" + s for s in ImagesList]
    Imgs = []
    for idx in range(len(ImagesList)):
        jpgfile = np.array(Image.open(ImagesList[idx]))
        Imgs.append(jpgfile)
    return Imgs

def Update_dataset_from_Images(data_x,data_y, minibatch_idx, batch_size):
    str = minibatch_idx * batch_size
    end = str + batch_size
    loaded_data = loadImages(data_x[str: end])
    # if minibatch_idx == 10:
    #     for idx in range(len(loaded_data)):
    #         print data_x[idx + str]
    #         print len(loaded_data[idx])
    #         plt.imshow(loaded_data[idx])
    #         plt.show()
    return numpy.asarray(loaded_data, dtype=theano.config.floatX), numpy.asarray(data_y[str: end], dtype='int32')