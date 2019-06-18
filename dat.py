
import numpy
from numpy.random import RandomState


def param(R, *size):
    return numpy.array(R.rand(*size) * 2 - 1, dtype='float32')


def mlp_data(*, seed=87):
    R = RandomState(seed=seed)
    #return [(param(R, 10, 100), param(R)) for i in range(10)]
    return (param(R, 10, 100), param(R))


def mlp_parameters(*, seed=90909):
    R = RandomState(seed=seed)
    sizes = (100, 1000, 5000, 5000, 10)
    parameters = []
    for i, o in zip(sizes[:-1], sizes[1:]):
        W = param(R, i, o)
        b = param(R, 1, o)
        parameters.append((W, b))
    return parameters


def big_matrix(rows=5000, cols=5000, *, seed=1782):
    R = RandomState(seed=seed)
    return param(R, rows, cols)
