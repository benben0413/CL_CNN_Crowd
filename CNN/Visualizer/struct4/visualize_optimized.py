import matplotlib.pyplot as plt
import theano
from CNN.Visualizer.struct4.deconv_constructor import *


def Attention(classifier, x, nkerns, jpgfile):
    inp = T.tensor4('inp')
    Forward_Pass = ForwardPass(classifier,x)
    Classify_pass = Classify(classifier,x)
    main_weights = construct_main_weights(classifier, nkerns)

    batch = jpgfile.reshape(1, jpgfile.shape[0], jpgfile.shape[1], 3)
    test_set_x = np.asarray(batch, dtype=theano.config.floatX)
    forward_frames = Forward_Pass(test_set_x)
    layer7_frames = forward_frames[0]
    layer1_frames = forward_frames[1]
    layer0_frames = forward_frames[2]

    #erase features from the image using deconv of specific layer and recall forward pass
    # new_batch = test(inp, layer0_frames, main_weights, batch)
    # new_batch = new_batch.reshape(1, new_batch.shape[0], new_batch.shape[1], new_batch.shape[2])
    # test_set_x = np.asarray(new_batch, dtype=theano.config.floatX)
    # forward_frames = Forward_Pass(test_set_x)
    # layer1_frames = forward_frames[1]

    #generate deconv full image from layer1
    # frame_size = layer1_frames.shape
    # for idx in range(layer1_frames.shape[1]):
    #     Layer_inp = layer1_frames[0][idx].reshape(1, 1, frame_size[2], frame_size[3])
    #     Visualizer1 = DeConv_structure_1(inp, main_weights, frame_size, 1)
    #     out_deconv = Visualizer1.layer0.output
    #     f = theano.function([inp], out_deconv)
    #     upsampled_img = f(Layer_inp)
    #     fname = "imgs/deconv/struct4/L1/deconv/test_%i_d.png" % idx
    #     plt.imsave(fname, upsampled_img[0].transpose(1, 2, 0))
    #     plt.close()



    # printFmaps(layer7_frames,"imgs/deconv/struct4/L7/t_conv_%i.png")
    c= 0
    for idx in range(layer7_frames.shape[2]):
        for jdx in range(layer7_frames.shape[3]):
            # bloc_size = layer7_frames.shape[2] * layer7_frames.shape[3]
            # bloc = np.zeros(bloc_size)
            # bloc[layer7_frames[0][1].argmax()] = np.max(layer7_frames[0][1])
            c += 1
            if layer7_frames[0][1][idx][jdx] > 0:
                bloc = np.zeros((layer7_frames.shape[2], layer7_frames.shape[3]))
                bloc[idx,jdx] = layer7_frames[0][1][idx][jdx]

                frame_size = layer7_frames.shape
                Layer_inp = np.array(bloc.reshape(1, 1, frame_size[2], frame_size[3]), dtype=theano.config.floatX)
                Visualizer7 = DeConv_structure(inp, main_weights, frame_size, 1)
                out_deconv = Visualizer7.layer0.output
                f = theano.function([inp], out_deconv)
                upsampled_img = f(Layer_inp)

                fname = "imgs/deconv/struct4/L7/test/fullarray/full_%i.png" %c
                plt.imsave(fname,upsampled_img[0].transpose(1,2,0))
                plt.close()
#     for in_idx in range(upsampled_img.shape[1]):
#         fname = "imgs/deconv/struct4/L7/l1_%i.png" % (in_idx)
#         plt.imsave(fname, upsampled_img[0][in_idx], cmap='gray')
#         plt.close()

    # print upsampled_img[0, 0, 130:150, 360:380]
    # np.putmask(upsampled_img, upsampled_img > 0, 1)
    # # print upsampled_img[0,0,130:150,360:380]
    # # plt.imshow(upsampled_img[0][0])
    # # plt.show()
    #
    # layer0_frames= np.multiply(layer0_frames,upsampled_img)
    #
    # for idx in range(layer0_frames.shape[1]):
    #     frame_size = layer0_frames.shape
    #     Layer_inp = layer0_frames[0][idx].reshape(1, 1, frame_size[2], frame_size[3])
    #     Visualizer = DeConv_structure_0(inp, main_weights, frame_size, idx)
    #     out_deconv = Visualizer.layer0.output
    #     f = theano.function([inp], out_deconv)
    #     upsampled_img = f(Layer_inp)
    #
    #     fname = "imgs/deconv/struct4/L7/test_%i_d.png" % idx
    #     plt.imsave(fname, upsampled_img[0].transpose(1, 2, 0))
    #     plt.close()

        # plt.imsave(fname,layer0_frames[0][idx], cmap='gray')
        # plt.close()


def test(inp, layer0_frames, main_weights, batch):

    frame_size = layer0_frames.shape
    Layer_inp = layer0_frames[0][6].reshape(1, 1, frame_size[2], frame_size[3])
    Visualizer = DeConv_structure_0(inp, main_weights, frame_size, 4)
    out_deconv = Visualizer.layer0.output
    f = theano.function([inp], out_deconv)
    upsampled_img = f(Layer_inp)

    added_img = np.add(np.add(upsampled_img[0][0],upsampled_img[0][1]),upsampled_img[0][2])
    np.putmask(added_img, added_img > 0, 1)

    new_img = np.array(np.multiply(batch[0].transpose(2,0,1), added_img), dtype=np.uint8).transpose(1,2,0)
    return new_img

def ForwardPass(classifier,x):
    output_layer7 = classifier.get_intermediate_sides()
    Test_model = theano.function(
        inputs=[x],
        outputs=output_layer7,
    )

    return Test_model

def Classify(classifier,x):
    output_layer9 = classifier.layer9.p_y_given_x  # layer9.inp #y_pred #
    Test_model = theano.function(
        inputs=[x],
        outputs=output_layer9,
    )

    return Test_model

def construct_main_weights(classifier,nkerns):
    no_params = len(nkerns) *2
    main_weights = []
    for idx in range(no_params,-1,-2):
        main_weights.append(classifier.params[idx].get_value())
    return main_weights

def printFmaps(lmaps, fname):
    for i in range(lmaps.shape[1]):
        plt.imsave(fname %i, lmaps[0][i],cmap='gray')
        plt.close()


