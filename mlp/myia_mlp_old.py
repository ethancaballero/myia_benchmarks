
from dat import mlp_parameters, mlp_data
import time
from myia.debug import traceback
from myia.api import myia
from myia.composite import grad
from myia.prim.py_implementations import array_to_scalar, array_reduce, \
    scalar_add
import numpy
from dataclasses import dataclass


@dataclass(frozen=True)
class Linear:
    W: 'Array'
    b: 'Array'

    def apply(self, input):
        return input @ self.W + self.b


@dataclass(frozen=True)
class Tanh:
    def apply(self, input):
        return numpy.tanh(input)


@dataclass(frozen=True)
class Sequential:
    layers: 'Tuple'

    def apply(self, x):
        for layer in self.layers:
            x = layer.apply(x)
        return x


def cost(model, x, target):
    y = model.apply(x)
    diff = target - y
    diffsqr = diff * diff
    return array_to_scalar(array_reduce(scalar_add, diff * diff, ()))
    # return array_reduce(scalar_add, diff * diff, (1,))


@myia
def step(model, x, y):
    return grad(cost)(model, x, y)
    # return cost(model, x, y)


@myia
def justamul(m1, m2):
    return m1 @ m2


def run_model():
    i = 0
    layers = []
    for W, b in dat.mlp_parameters():
        layers.append(Linear(W, b))
        layers.append(Tanh())
    model = Sequential(tuple(layers))

    for inp, target in dat.mlp_data():
        t0 = time.time()
        res = step(model, inp, target)
        # res.layers[0].W.asnumpy()
        print(time.time() - t0)
        i += 1


def run_matmul():
    m1 = dat.big_matrix(5000, 5000, seed=56)
    m2 = dat.big_matrix(5000, 5000, seed=57)
    for i in range(10):
        t0 = time.time()
        newm = justamul(m1, m2)
        print(time.time() - t0)


#if __name__ == '__main__':
#run_model()
