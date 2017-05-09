import theano


def Sequemtial_cnn_Prediction(classifier,x, test_datasets):

    test_x = test_datasets
    # test_set_x = theano.shared(numpy.asarray(test_x, dtype=theano.config.floatX), borrow = True)

    output = classifier.predictor.y_pred

    Test_model = theano.function(
        inputs = [x],
        outputs= output
    )

    return Test_model(test_x)