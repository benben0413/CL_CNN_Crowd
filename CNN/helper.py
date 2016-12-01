import numpy as np
import theano
import theano.tensor as T

def arrange_training_input(data):
    labels = np.zeros(len(data))

    tr_x = []
    for idx in range(len(data)):
        tr_x.append(data[idx][0])
        if len(data[idx][1]) > 0:
            labels[idx] = 1
    return np.array(tr_x),np.array(labels)



def CNN_tester(classifier,x, test_data, batch_size, layer):

    index = T.lscalar()
    test_data = theano.shared(np.asarray(test_data, dtype=theano.config.floatX))

    if layer == 0:
        output = classifier.layer0.get_conv_value()
    elif layer == 1:
        output = classifier.layer1.get_conv_value()
    elif layer == 2:
        output = classifier.layer2.get_conv_value()
    elif layer == 3:
        output = classifier.layer3.get_conv_value()
    elif layer == 4:
        output = classifier.layer4.get_conv_value()
    elif layer == 5:
        output = classifier.layer5.get_conv_value()


    ppm = theano.function(
        inputs=[index],
        outputs=output,
        givens={
            x: test_data[index * batch_size: (index + 1) * batch_size]  # ,
            # y: test_labels
        },
        on_unused_input='warn'
    )
    # print ppm(0)
    return ppm(0)




# NUM_TRAIN = len(data)
#     if NUM_TRAIN % batch_size != 0: #if the last batch is not full, just don't use the remainder
#         whole = (NUM_TRAIN / batch_size) * batch_size
#         data = data[:whole]
#         NUM_TRAIN = len(data)
#
#     # random permutation
#     indices = rng.permutation(NUM_TRAIN)
#     data, labels = data[indices, :], labels[indices]
#
#     # batch_size == 500, splits (480, 20). We will use 96% of the data for training, and the rest to validate the NN while training
#     is_train = numpy.array( ([0]* (batch_size - 20) + [1] * 20) * (NUM_TRAIN / batch_size))
#
#     # now we split the dataset to train and valid datasets
#     train_set_x, train_set_y = numpy.array(data[is_train==0]), labels[is_train==0]
#     valid_set_x, valid_set_y = numpy.array(data[is_train==1]), labels[is_train==1]
#
#     # #just for test and delete immediatly
#     # for i in xrange(2500):
#     #     train_set_x[i, :], train_set_x[i:] = numpy.random.random((50, 50)).flatten(), 0
#
#     # compute number of minibatches
#     n_train_batches = len(train_set_y) / batch_size
#     if len(valid_set_x) > batch_size:
#         n_valid_batches = len(valid_set_y) / batch_size
#     else:
#         n_active_valid = len(valid_set_x)
#         n_valid_batches = 1
#         noise_valid_set_x, noise_valid_set_y = numpy.zeros((500-n_active_valid, len(data[0]))), numpy.zeros(500-n_active_valid)
#         for i in xrange(500-n_active_valid):
#             noise_valid_set_x[i] = numpy.random.random((50, 50)).flatten()
#         valid_set_x, valid_set_y = numpy.concatenate((valid_set_x, noise_valid_set_x), axis=0), numpy.concatenate((valid_set_y,noise_valid_set_y),axis=0)
#
#         # valid_set_x[n_active_valid:n_active_valid+50,:], valid_set_y[n_active_valid:n_active_valid+50] = train_set_x[0:50,:] , 0
#         # valid_set_x[n_active_valid + 50:, :], valid_set_y[n_active_valid + 50:] = numpy.random.random((50, 50)).flatten(), 0