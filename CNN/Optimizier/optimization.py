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


class gd_optimizer():
    def __init__(self, classop,params):
        self.classop = classop
        if self.classop == 'momentum' or "nestrov":
            self.memory_ = [theano.shared(np.zeros_like(p.get_value())) for p in params]

    def update_param(self,grads, params, learning_rate, momentum):
        updates = []
        if self.classop == 'momentum':
            grad_clipped =  grads #
            for n,(param_i,grad_i) in enumerate(zip(params,grad_clipped)):
                memory = self.memory_[n]
                # grad_clipped = T.clip(grad_i, -1, 1)
                velocity = momentum * memory - learning_rate * grad_i
                updates.append((memory, velocity))
                updates.append((param_i, param_i+ velocity))
                # updates.append((param_i, param_i -learning_rate * grad_i))
        elif self.classop == 'nestrov':
            grad_clipped = grads #[T.clip(g,-1,1) for g in grads]
            for n, (param_i, grad_i) in enumerate(zip(params, grad_clipped)):
                memory = self.memory_[n]
                update1 = momentum * memory - learning_rate * grad_i
                update2 = momentum * momentum * memory - ( 1 + momentum) * learning_rate * grad_i
                updates.append((memory, update1))
                updates.append((param_i, param_i + update2))
        elif self.classop == 'sgd':
        # specify how to update the parameters of the model as a list of (variable, update expression) pairs
        #     for param_i,grad_i in zip(params, grads):
        #         updates.append((param_i, param_i - learning_rate * grad_i))
            updates = [
                (param_i, param_i - learning_rate * grad_i)
                for param_i, grad_i in zip(params, grads)
                ]
        return updates