import theano
import numpy as np
import theano.tensor as T


def switchs(layer_Fmaps, operation='get', step=2, switches=None):
    sw_size = step * step
    if operation == 'get':
        switches = np.zeros((layer_Fmaps.shape[1], layer_Fmaps.shape[2], layer_Fmaps.shape[3]),
                            dtype=theano.config.floatX)
        for idx in range(layer_Fmaps.shape[1]):
            layer = layer_Fmaps[0][idx]
            main_switch = np.zeros(layer.shape)
            for i in range(0, layer.shape[0], step):
                for j in range(0, layer.shape[1], step):
                    switch = np.zeros(sw_size)
                    ds = layer[i:i + step, j:j + step]
                    switch[ds.argmax()] = 1
                    switch = switch.reshape(step, step)
                    main_switch[i:i + step, j:j + step] = switch
            switches[idx] = main_switch
        return switches
    elif operation == 'set':
        SW_located = np.zeros((96,708,708))
        batch = np.ones((1, 96, 708, 708))
        for idx in range(96):
            x = 0
            y = 0
            for i in range(0, 708, step):
                for j in range(0, 708, step):
                    # loc = np.zeros(sw_size)
                    # arg = switches[idx][i:i + step, j:j + step].argmax()
                    # loc[0] = 9 #layer_Fmaps[0][idx][0, 0]
                    # x+=1
                    # y+=1
                    # loc = loc.reshape(step, step)
                    # switches = T.set_subtensor(switches[0][idx][i:i + step, j:j + step], loc)
                    # loc = np.zeros(sw_size)
                    # # arg = switches[idx][i:i + step, j:j + step].argmax()
                    # loc[0] = 9 #layer_Fmaps[0][idx][x, y]
                    # x+=1
                    # y+=1
                    # loc = loc.reshape(step, step)
                    # SW_located = T.set_subtensor(switches[0][idx][i:i + step, j:j + step], loc)

                    val = layer_Fmaps[0][idx][0,0]
                    batch[0][idx][i:i + step, j:j + step] = batch[0][idx][i:i + step, j:j + step] * 9

                    # switches = T.set_subtensor(switches[0][idx][i:i + 3, j:j + 3],9)

        # SW_located = SW_located.reshape(1, 96, 708, 708)
        return  switches * batch



class switchs_swapper(object):
    def __init__(self,layer_Fmaps, step=2, switches=None):

        sw_size = step * step
        SW_located = np.zeros((96,708,708))
        batch = np.ones((1, 96, 708, 708))

        for idx in range(96):
            for i in range(0, 708, step):
                for j in range(0, 708, step):

                    val = layer_Fmaps[0][idx][0][0]
                    batch[0][idx][i:i + step, j:j + step] = batch[0][idx][i:i + step, j:j + step] * val

                    # switches = T.set_subtensor(switches[0][idx][i:i + 3, j:j + 3],9)
        self.switchs = theano.shared(batch, borrow=True)



img = np.zeros((1,96,236,236))
sswitchs =  np.zeros((1,96,708,708))

inp = T.tensor4('img')
SW = T.tensor4('SW')

# tester = switchs(inp,'set',3,SW)
# f = theano.function([inp, SW], tester)

tester = switchs_swapper(inp,3,SW)
out = tester.switchs

f = theano.function([inp, SW], out, on_unused_input='ignore', allow_input_downcast=True)

d = f(img,sswitchs)
print out.get_value()