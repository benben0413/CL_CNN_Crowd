import theano
import numpy as np
import theano.tensor as T

def grd_nesterov(cost, params, learning_rate, momenteum):
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value()*0, brodcastable=param.brodcastable)
        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, momenteum*param_update + (1-momenteum)* T.grad(cost, param)))
    return updates


class sgd_momentum():
    def __init__(self,params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value())) for p in params]

    def update_param(self,grads, params, learning_rate, momentum):
        updates = []
        grad_clipped = [T.clip(g,-1,1) for g in grads]
        for n,(param_i,grad_i) in enumerate(zip(params,grad_clipped)):
            memory = self.memory_[n]
            # grad_clipped = T.clip(grad_i, -1, 1)
            first_update = momentum * memory - learning_rate * grad_i
            updates.append((memory, first_update))
            updates.append((param_i, param_i+ memory))
            # updates.append((param_i, param_i -learning_rate * grad_i))

        return updates