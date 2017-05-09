from Softmax_classifier import *
from CrossEntropy_classifier import *
from Constructor import *

class CNN_struct_comp1(object):

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

        #The Reciever
        # LR_k = K_sizes[0]
        # self.Receiver = ConvLayer(
        #     rng,
        #     input=self.layer0_input,
        #     image_shape=(batch_size, image_dimension, image_size[0], image_size[1]),
        #     filter_shape=(nkerns[0], image_dimension, LR_k[0], LR_k[1]),
        # )

        #bloc1
        L1_k = K_sizes[1]
        self.bloc1 = ConvLayer(
            rng,
            input= self.layer0_input,
            image_shape=(batch_size, image_dimension, image_size[0], image_size[1]),
            filter_shape=(nkerns[1], image_dimension,  L1_k[0], L1_k[1]),
            subsample = (2,2)
        )


        #bloc2
        L2_0_k = K_sizes[2]
        bloc2_0_input = self.bloc1.output[:,0:2,:,:]
        self.bloc2_0 = ConvLayer(
            rng,
            input=bloc2_0_input,
            image_shape=(batch_size, nkerns[1]-1, L_sizes[1][0], L_sizes[1][1]),
            filter_shape=(nkerns[2], nkerns[1]-1, L2_0_k[0], L2_0_k[1]),
            border_mode='half'
        )

        L2_1_k = K_sizes[2]
        bloc2_1_input = self.bloc1.output[:, 0:3:2, :, :]
        self.bloc2_1 = ConvLayer(
            rng,
            input=bloc2_1_input,
            image_shape=(batch_size, nkerns[1] -1, L_sizes[1][0], L_sizes[1][1]),
            filter_shape=(nkerns[2], nkerns[1] -1, L2_1_k[0], L2_1_k[1]),
            border_mode='half'
        )

        L2_2_k = K_sizes[2]
        bloc2_2_input = self.bloc1.output[:, 1:, :, :]
        self.bloc2_2 = ConvLayer(
            rng,
            input=bloc2_2_input,
            image_shape=(batch_size, nkerns[1] -1, L_sizes[1][0], L_sizes[1][1]),
            filter_shape=(nkerns[2], nkerns[1] -1, L2_2_k[0], L2_2_k[1]),
            border_mode='half'
        )


        #bloc3
        L3_0_k = K_sizes[3]
        self.bloc3_0 = ConvLayer(
            rng,
            input=self.bloc2_0.output,
            image_shape=(batch_size, nkerns[2], L_sizes[2][0], L_sizes[2][1]),
            filter_shape=(nkerns[3], nkerns[2], L3_0_k[0], L3_0_k[1]),
            border_mode='half'
        )

        L3_1_k = K_sizes[3]
        self.bloc3_1 = ConvLayer(
            rng,
            input=self.bloc2_1.output,
            image_shape=(batch_size, nkerns[2], L_sizes[2][0], L_sizes[2][1]),
            filter_shape=(nkerns[3], nkerns[2], L3_1_k[0], L3_1_k[1]),
            border_mode='half'
        )

        L3_2_k = K_sizes[3]
        self.bloc3_2 = ConvLayer(
            rng,
            input=self.bloc2_2.output,
            image_shape=(batch_size, nkerns[2], L_sizes[2][0], L_sizes[2][1]),
            filter_shape=(nkerns[3], nkerns[2], L3_2_k[0], L3_2_k[1]),
            border_mode='half'
        )

        self.Conc1 = T.concatenate([self.bloc3_0.output, self.bloc3_1.output, self.bloc3_2.output], axis=1)

        #bloc4
        conc1_size = nkerns[3] * 3
        L4_0_k = K_sizes[4]
        self.bloc4_0 = ConvLayer(
            rng,
            input=self.Conc1,
            image_shape=(batch_size, conc1_size, L_sizes[3][0], L_sizes[3][1]),
            filter_shape=(nkerns[4], conc1_size, L4_0_k[0], L4_0_k[1]),
            border_mode='half'
        )

        L4_1_k = K_sizes[4]
        self.bloc4_1 = ConvLayer(
            rng,
            input=self.Conc1,
            image_shape=(batch_size, conc1_size, L_sizes[3][0], L_sizes[3][1]),
            filter_shape=(nkerns[4], conc1_size, L4_0_k[0], L4_0_k[1]),
            border_mode='half'
        )

        L4_2_k = K_sizes[4]
        self.bloc4_2 = ConvLayer(
            rng,
            input=self.Conc1,
            image_shape=(batch_size, conc1_size, L_sizes[3][0], L_sizes[3][1]),
            filter_shape=(nkerns[4], conc1_size, L4_0_k[0], L4_0_k[1]),
            border_mode='half'
        )

        L4_3_k = K_sizes[4]
        self.bloc4_3 = ConvLayer(
            rng,
            input=self.Conc1,
            image_shape=(batch_size, conc1_size, L_sizes[3][0], L_sizes[3][1]),
            filter_shape=(nkerns[4], conc1_size, L4_0_k[0], L4_0_k[1]),
            border_mode='half'
        )

        L4_4_k = K_sizes[4]
        self.bloc4_4 = ConvLayer(
            rng,
            input=self.Conc1,
            image_shape=(batch_size, conc1_size, L_sizes[3][0], L_sizes[3][1]),
            filter_shape=(nkerns[4], conc1_size, L4_0_k[0], L4_0_k[1]),
            border_mode='half'
        )

        L4_5_k = K_sizes[4]
        self.bloc4_5 = ConvLayer(
            rng,
            input=self.Conc1,
            image_shape=(batch_size, conc1_size, L_sizes[3][0], L_sizes[3][1]),
            filter_shape=(nkerns[4], conc1_size, L4_0_k[0], L4_0_k[1]),
            border_mode='half'
        )

        L4_6_k = K_sizes[4]
        self.bloc4_6 = ConvLayer(
            rng,
            input=self.Conc1,
            image_shape=(batch_size, conc1_size, L_sizes[3][0], L_sizes[3][1]),
            filter_shape=(nkerns[4], conc1_size, L4_0_k[0], L4_0_k[1]),
            border_mode='half'
        )

        L4_7_k = K_sizes[4]
        self.bloc4_7 = ConvLayer(
            rng,
            input=self.Conc1,
            image_shape=(batch_size, conc1_size, L_sizes[3][0], L_sizes[3][1]),
            filter_shape=(nkerns[4], conc1_size, L4_0_k[0], L4_0_k[1]),
            border_mode='half'
        )

        L4_8_k = K_sizes[4]
        self.bloc4_8 = ConvLayer(
            rng,
            input=self.Conc1,
            image_shape=(batch_size, conc1_size, L_sizes[3][0], L_sizes[3][1]),
            filter_shape=(nkerns[4], conc1_size, L4_0_k[0], L4_0_k[1]),
            border_mode='half'
        )

        L4_9_k = K_sizes[4]
        self.bloc4_9 = ConvLayer(
            rng,
            input=self.Conc1,
            image_shape=(batch_size, conc1_size, L_sizes[3][0], L_sizes[3][1]),
            filter_shape=(nkerns[4], conc1_size, L4_0_k[0], L4_0_k[1]),
            border_mode='half'
        )


        #bloc5
        L5_0_k = K_sizes[5]
        self.bloc5_0 = ConvLayer(
            rng,
            input=self.bloc4_0.output,
            image_shape=(batch_size, nkerns[4], L_sizes[4][0], L_sizes[4][1]),
            filter_shape=(nkerns[5], nkerns[4], L5_0_k[0], L5_0_k[1]),
            border_mode='half'
        )

        L5_1_k = K_sizes[5]
        self.bloc5_1 = ConvLayer(
            rng,
            input=self.bloc4_1.output,
            image_shape=(batch_size, nkerns[4], L_sizes[4][0], L_sizes[4][1]),
            filter_shape=(nkerns[5], nkerns[4], L5_0_k[0], L5_0_k[1]),
            border_mode='half'
        )

        L5_2_k = K_sizes[4]
        self.bloc5_2 = ConvLayer(
            rng,
            input=self.bloc4_2.output,
            image_shape=(batch_size, nkerns[4], L_sizes[4][0], L_sizes[4][1]),
            filter_shape=(nkerns[5], nkerns[4], L5_0_k[0], L5_0_k[1]),
            border_mode='half'
        )

        L5_3_k = K_sizes[4]
        self.bloc5_3 = ConvLayer(
            rng,
            input=self.bloc4_3.output,
            image_shape=(batch_size, nkerns[4], L_sizes[4][0], L_sizes[4][1]),
            filter_shape=(nkerns[5], nkerns[4], L5_0_k[0], L5_0_k[1]),
            border_mode='half'
        )

        L5_4_k = K_sizes[4]
        self.bloc5_4 = ConvLayer(
            rng,
            input=self.bloc4_4.output,
            image_shape=(batch_size, nkerns[4], L_sizes[4][0], L_sizes[4][1]),
            filter_shape=(nkerns[5], nkerns[4], L5_0_k[0], L5_0_k[1]),
            border_mode='half'
        )

        L5_5_k = K_sizes[4]
        self.bloc5_5 = ConvLayer(
            rng,
            input=self.bloc4_5.output,
            image_shape=(batch_size, nkerns[4], L_sizes[4][0], L_sizes[4][1]),
            filter_shape=(nkerns[5], nkerns[4], L5_0_k[0], L5_0_k[1]),
            border_mode='half'
        )

        L5_6_k = K_sizes[4]
        self.bloc5_6 = ConvLayer(
            rng,
            input=self.bloc4_6.output,
            image_shape=(batch_size, nkerns[4], L_sizes[4][0], L_sizes[4][1]),
            filter_shape=(nkerns[5], nkerns[4], L5_0_k[0], L5_0_k[1]),
            border_mode='half'
        )

        L5_7_k = K_sizes[4]
        self.bloc5_7 = ConvLayer(
            rng,
            input=self.bloc4_7.output,
            image_shape=(batch_size, nkerns[4], L_sizes[4][0], L_sizes[4][1]),
            filter_shape=(nkerns[5], nkerns[4], L5_0_k[0], L5_0_k[1]),
            border_mode='half'
        )

        L5_8_k = K_sizes[4]
        self.bloc5_8 = ConvLayer(
            rng,
            input=self.bloc4_8.output,
            image_shape=(batch_size, nkerns[4], L_sizes[4][0], L_sizes[4][1]),
            filter_shape=(nkerns[5], nkerns[4], L5_0_k[0], L5_0_k[1]),
            border_mode='half'
        )

        L5_9_k = K_sizes[4]
        self.bloc5_9 = ConvLayer(
            rng,
            input=self.bloc4_9.output,
            image_shape=(batch_size, nkerns[4], L_sizes[4][0], L_sizes[4][1]),
            filter_shape=(nkerns[5], nkerns[4], L5_0_k[0], L5_0_k[1]),
            border_mode='half'
        )



        self.Conc2 = T.concatenate([self.bloc5_0.output, self.bloc5_1.output, self.bloc5_2.output, self.bloc5_3.output, self.bloc5_4.output,
                                    self.bloc5_5.output, self.bloc5_6.output, self.bloc5_7.output, self.bloc5_8.output, self.bloc5_9.output], axis=1)

        #pool
        self.pool = PoolLayer(
            input=self.Conc2,
            poolsize=(2,2),
            same= True
        )

        #bloc6
        L6_k = K_sizes[6]
        conc2_size = nkerns[5] * 10
        self.bloc6 = ConvLayer(
            rng,
            input=self.pool.output,
            image_shape=(batch_size, conc2_size, L_sizes[5][0], L_sizes[5][1]),
            filter_shape=(nkerns[6], conc2_size, L6_k[0], L6_k[1]),
            subsample=(2, 2)
        )

        # #bloc7
        L7_0_k = K_sizes[7]
        self.bloc7_0 = ConvLayer(
            rng,
            input=self.bloc6.output,
            image_shape=(batch_size, nkerns[6], L_sizes[6][0], L_sizes[6][1]),
            filter_shape=(nkerns[7], nkerns[6], L7_0_k[0], L7_0_k[1]),
            border_mode='half'
        )

        L7_1_k = K_sizes[7]
        self.bloc7_1 = ConvLayer(
            rng,
            input=self.bloc6.output,
            image_shape=(batch_size, nkerns[6], L_sizes[6][0], L_sizes[6][1]),
            filter_shape=(nkerns[7], nkerns[6], L7_1_k[0], L7_1_k[1]),
            border_mode='half',
            activ_func= 'sigmoid'
        )

        self.Conc3 = T.concatenate([self.bloc7_0.output, self.bloc7_1.output], axis=1)

        # bloc8
        # self.bloc8 = SumPoolLayer(
        #     input=self.bloc7_1.output,
        #     image_shape=L_sizes[7],
        # )

        # self.bloc8 = AvgPoolLayer(
        #     input=self.Conc3,
        #     image_shape=L_sizes[7]
        # )

        self.bloc8 = AvgPoolLayer(
            input= self.bloc7_1.output,
            image_shape=L_sizes[7]
        )



        self.sofmax_input = self.bloc8.output.flatten(1) #1

        # self.predictor = Softmax_classifier(input = self.sofmax_input)
        self.predictor = CrossEntropy_classifier(input= self.sofmax_input)

        # create a list of all model parameters to be fit by gradient descent
        self.init_param =  self.bloc7_1.params + \
                           self.bloc6.params + \
                           self.bloc5_0.params + self.bloc5_1.params + self.bloc5_2.params + self.bloc5_3.params + self.bloc5_4.params + \
                           self.bloc5_5.params + self.bloc5_6.params + self.bloc5_7.params + self.bloc5_8.params + self.bloc5_9.params + \
                           self.bloc4_0.params + self.bloc4_1.params + self.bloc4_2.params + self.bloc4_3.params + self.bloc4_4.params + \
                           self.bloc4_5.params + self.bloc4_6.params + self.bloc4_7.params + self.bloc4_8.params + self.bloc4_9.params + \
                           self.bloc3_0.params + self.bloc3_1.params + self.bloc3_2.params + \
                           self.bloc2_0.params + self.bloc2_1.params + self.bloc2_2.params + \
                           self.bloc1.params #+ \
                           # self.Receiver.params  + self.bloc7_0.params +

        self.params = self.init_param

    def get_intermediate_sides(self):
        return  self.Conc3, self.sofmax_input, self.bloc7_1.output #, self.predictor.p_y_given_x
