from CNN.Structures.AvgPoolSoftmax import AvgPoolSoftmax
from CNN.Structures.Constructor import *


class CNN_struct6_2(object):

    def __getstate__(self):
        weights = [p.get_value() for p in self.params]
        return weights

    def __setstate__(self, weights):
        i = iter(weights)
        for p in self.params:
            p.set_value(i.next())


    def __init__(self, rng, input, nkerns, batch_size, image_size, image_dimension, L_sizes):
        # Reshape matrix of rasterized images of shape (batch_size, size[0] * size[1])
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # self.layer0_input = input.reshape((batch_size, image_dimension, image_size[0], image_size[1]))
        self.layer0_input = input.transpose(0, 3, 1, 2)
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (100-7+1 , 100-7+1) = (94, 94)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 94, 94)
        L0_k = [7,7]
        self.layer0 = LeNetConvPoolLayer(
            rng,
            input=self.layer0_input,
            image_shape=(batch_size, image_dimension, image_size[0], image_size[1]),
            filter_shape=(nkerns[0], image_dimension, L0_k[0], L0_k[1])
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (94-3+1, 94-3+1) = (92, 92)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 92, 92)
        L1_k = [3, 3]
        self.layer1 = LeNetConvPoolLayer(
            rng,
            input= self.layer0.output,
            image_shape=(batch_size, nkerns[0], L_sizes[0][0], L_sizes[0][1]),
            filter_shape=(nkerns[1], nkerns[0],  L1_k[0], L1_k[1])
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (92-3+1, 92-3+1) = (90, 90)
        # maxpooling reduces this further to (90/3, 90/3) = (30, 30)
        # 4D output tensor is thus of shape (batch_size, nkerns[2], 30, 30)
        L2_k = [3, 3]
        self.layer2 = LeNetConvPoolLayer(
            rng,
            input = self.layer1.output,
            image_shape=(batch_size, nkerns[1], L_sizes[1][0], L_sizes[1][1]),
            filter_shape=(nkerns[2], nkerns[1], L2_k[0], L2_k[1])
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (30-3+1, 30-3+1) = (28, 28)
        # 4D output tensor is thus of shape (batch_size, nkerns[3], 28, 28)
        L3_k = [3, 3]
        self.layer3 = LeNetConvPoolLayer(
             rng,
             input=self.layer2.output,
             image_shape=(batch_size, nkerns[2], L_sizes[2][0], L_sizes[2][1]),
             filter_shape=(nkerns[3], nkerns[2], L3_k[0], L3_k[1])
         )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (28-3+1, 28-3+1) = (26, 26)
        # 4D output tensor is thus of shape (batch_size, nkerns[4], 26, 26)
        L4_k = [3, 3]
        self.layer4 = LeNetConvPoolLayer(
            rng,
            input=self.layer3.output,
            image_shape=(batch_size, nkerns[3], L_sizes[3][0], L_sizes[3][1]),
            filter_shape=(nkerns[4], nkerns[3], L4_k[0], L4_k[1])
        )

        L5_k = [3, 3]
        self.layer5 = LeNetConvPoolLayer(
            rng,
            input=self.layer4.output,
            image_shape=(batch_size, nkerns[4], L_sizes[4][0], L_sizes[4][1]),
            filter_shape=(nkerns[5], nkerns[4], L5_k[0], L5_k[1])
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (26-3+1, 26-3+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[5], 12, 12)
        L6_k = [1, 1]
        self.layer6 = LeNetConvPoolLayer(
            rng,
            input=self.layer5.output,
            image_shape=(batch_size, nkerns[5],  L_sizes[5][0], L_sizes[5][1]),
            filter_shape=(nkerns[6], nkerns[5], L6_k[0], L6_k[1])
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-3+1, 12-3+1) = (10, 10)
        # maxpooling reduces this further to (24/2, 24/2) = (5, 5)
        # 4D output tensor is thus of shape (batch_size, nkerns[6], 5, 5)
        L7_k = [3, 3]
        self.layer7 = LeNetConvPoolLayer(
            rng,
            input=self.layer6.output,
            image_shape=(batch_size, nkerns[6],  L_sizes[6][0], L_sizes[6][1]),
            filter_shape=(nkerns[7], nkerns[6], L7_k[0], L7_k[1]),
            subsample=(2,2)
        )

        L8_k = [1, 1]
        self.layer8 = LeNetConvPoolLayer(
            rng,
            input=self.layer7.output,
            image_shape=(batch_size, nkerns[7], L_sizes[7][0], L_sizes[7][1]),
            filter_shape=(nkerns[8], nkerns[7], L8_k[0], L8_k[1]),
        )

        L9_k = [3, 3]
        self.layer9 = LeNetConvPoolLayer(
            rng,
            input=self.layer8.output,
            image_shape=(batch_size, nkerns[8], L_sizes[8][0], L_sizes[8][1]),
            filter_shape=(nkerns[9], nkerns[8], L9_k[0], L9_k[1]),
            subsample=(2, 2)
        )

        self.layer10 = AvgPoolLayer(
            input= self.layer9.output,
            image_shape=L_sizes[9]
        )

        self.layer11_input = self.layer10.output.flatten(2)
        self.layer11 = AvgPoolSoftmax(input = self.layer11_input)

        # create a list of all model parameters to be fit by gradient descent
        self.init_param =  self.layer9.params + self.layer8.params + self.layer7.params + self.layer6.params + \
                           self.layer5.params + self.layer4.params + self.layer3.params +\
                           self.layer2.params + self.layer1.params + self.layer0.params

        self.params = self.init_param

    def get_intermediate_sides(self):
        return  self.layer7.output, self.layer1.output, self.layer0.output
