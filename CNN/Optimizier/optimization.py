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
            for param_i,grad_i in zip(params, grads):
                updates.append((param_i, param_i - learning_rate * grad_i))
            # updates = [
            #     (param_i, param_i - learning_rate * grad_i)
            #     for param_i, grad_i in zip(params, grads)
            #     ]
        return updates
    def tester(self, grads, params, learning_rate):
        updates = []

        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - learning_rate * grad_i))
        # updates = [
        #     (param_i, param_i - learning_rate * grad_i)
        #     for param_i, grad_i in zip(params, grads)
        #     ]

        return  updates #grads[60], grads[61], updates[60], updates[61] #params[60], params[61],
    # def floatX(X):
    #     return np.asarray(X, dtype=theano.config.floatX)
    #
    # def init_weights(shape, factor=0.00001):
    #     return theano.shared(floatX(np.random.randn(*shape) * factor))
    #
    # def rectify(X):
    #     return T.maximum(X, 0.0)
    #
    # def softmax(X):
    #     e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    #     return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
    #
    # def dropout(X, p=0.0):
    #     if p > 0:
    #         retain_prob = 1 - p
    #         X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    #         X /= retain_prob
    #     return X
    #
    # def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    #     grads = T.grad(cost=cost, wrt=params)
    #     updates = []
    #     for p, g in zip(params, grads):
    #         acc = theano.shared(p.get_value() * 0.0)
    #         acc_new = rho * acc + (1 - rho) * g ** 2
    #         gradient_scaling = T.sqrt(acc_new + epsilon)
    #         g = g / gradient_scaling
    #         updates.append((acc, acc_new))
    #         updates.append((p, p - lr * g))
    #     return updates