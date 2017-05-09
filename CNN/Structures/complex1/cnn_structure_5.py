from Softmax_classifier import *
from CrossEntropy_classifier import *
from Constructor import *

class CNN_struct_comp5(object):

    def __getstate__(self):
        weights = [p.get_value() for p in self.params]
        return weights

    def __setstate__(self, weights):
        i = iter(weights)
        for p in self.params:
            p.set_value(i.next())


    def __init__(self, rng, input, nkerns, batch_size, image_size, image_dimension, L_sizes, K_sizes):
        # Reshape matrix of rasterized images of shape (batch_size, size[0] * size[1])
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # self.layer0_input = input.reshape((batch_size, image_dimension, image_size[0], image_size[1]))
        self.layer0_input = input.transpose(0, 3, 1, 2)


        #bloc1
        L1_k = [3,3]
        self.bloc1 = ConvLayer(
            rng,
            input= self.layer0_input,
            image_shape=(batch_size, image_dimension, image_size[0], image_size[1]),
            filter_shape=(16, image_dimension,  L1_k[0], L1_k[1]),
            border_mode='half'
        )

        L1_k = [1, 1]
        self.a_bloc1 = ConvLayer(
            rng,
            input=self.bloc1.output,
            image_shape=(batch_size, 16, image_size[0], image_size[1]),
            filter_shape=(32, 16, L1_k[0], L1_k[1]),
            border_mode='half'
        )


        #bloc2
        L2_0_k = K_sizes[2]
        # bloc2_0_input = self.bloc1.output[:,0:2,:,:]
        self.bloc2_0 = ConvLayer(
            rng,
            input=self.a_bloc1.output,
            image_shape=(batch_size, 32, L_sizes[1][0], L_sizes[1][1]),
            filter_shape=(48, 32, L2_0_k[0], L2_0_k[1]),
            border_mode='half'
        )

        #bloc3
        L3_0_k = K_sizes[3]
        self.bloc3_0 = ConvLayer(
            rng,
            input=self.bloc2_0.output,
            image_shape=(batch_size, 48, L_sizes[2][0], L_sizes[2][1]),
            filter_shape=(96, 48, L3_0_k[0], L3_0_k[1]),
            border_mode='half'
        )


        conc1_size = nkerns[3] * 3
        self.a_Conc1 = ConvLayer(
            rng,
            input=self.bloc3_0.output,
            image_shape=(batch_size, 96, L_sizes[3][0], L_sizes[3][1]),
            filter_shape=(96, 96, 1, 1),
            border_mode='half'
        )

        #bloc4
        conc1_size = nkerns[3] * 3
        L4_0_k = [3,3] #K_sizes[4]
        self.bloc4_0 = ConvLayer(
            rng,
            input=self.a_Conc1.output,
            image_shape=(batch_size, 96, L_sizes[3][0], L_sizes[3][1]),
            filter_shape=(72, 96, L4_0_k[0], L4_0_k[1]),
            border_mode='half'
        )


        #bloc5
        L5_0_k = [3,3] #K_sizes[5]
        self.bloc5_0 = ConvLayer(
            rng,
            input=self.bloc4_0.output,
            image_shape=(batch_size, 72, L_sizes[4][0], L_sizes[4][1]),
            filter_shape=(144, 72, L5_0_k[0], L5_0_k[1]),
            border_mode='half'
        )


        conc1_size = nkerns[5] * 3
        self.a_Conc2 = ConvLayer(
            rng,
            input=self.bloc5_0.output,
            image_shape=(batch_size, 144, L_sizes[3][0], L_sizes[3][1]),
            filter_shape=(144, 144, 1, 1),
            border_mode='half'
        )


        #bloc6
        L6_k = K_sizes[6]
        conc2_size = nkerns[5] * 3
        self.bloc6 = ConvLayer(
            rng,
            input=self.a_Conc2.output,
            image_shape=(batch_size, 144, L_sizes[5][0], L_sizes[5][1]),
            filter_shape=(96, 144, L6_k[0], L6_k[1]),
            border_mode='half'
        )


        L7_1_k = K_sizes[7]
        self.bloc7_1 = ConvLayer(
            rng,
            input=self.bloc6.output,
            image_shape=(batch_size, nkerns[6], L_sizes[6][0], L_sizes[6][1]),
            filter_shape=(nkerns[7], nkerns[6], L7_1_k[0], L7_1_k[1]),
            border_mode='half'
        )


        self.average_bloc8 = AvgPoolLayer(
            input= self.bloc7_1.output,
            image_shape=L_sizes[7]
        )

        self.max_bloc8 = PoolLayer(
            input= self.bloc7_1.output,
            poolsize=L_sizes[7]
        )


        self.costfunction_input_average = self.average_bloc8.output.flatten(1)
        self.costfunction_input_max = self.max_bloc8.output.flatten(1)

        self.predictor = CrossEntropy_classifier(self.costfunction_input_average, self.costfunction_input_max)

        # create a list of all model parameters to be fit by gradient descent
        self.init_param =  self.bloc7_1.params + \
                           self.bloc6.params + \
                           self.a_Conc2.params + \
                           self.bloc5_0.params + \
                           self.bloc4_0.params + \
                           self.a_Conc1.params + \
                           self.bloc3_0.params + \
                           self.bloc2_0.params + \
                           self.bloc1.params + self.a_bloc1.params

        self.params = self.init_param

    def get_intermediate_sides(self):
        return  self.bloc7_1.output, self.costfunction_input_average, self.costfunction_input_max

    def get_intermediate_sides_all(self):
        return self.bloc7_1.output , self.bloc6.output , self.a_Conc2.output, self.bloc5_0.output ,\
               self.bloc4_0.output , self.a_Conc1.output, self.bloc3_0.output ,\
               self.bloc2_0.output , self.a_bloc1.output, self.bloc1.output



    def struct_tester(self):
        return self.costfunction_input_max