"""Example of an RNN in Myia.

Myia is still a work in progress, and this example may change in the future.
"""


import time
import numpy
from numpy.random import RandomState
from dataclasses import dataclass

from myia import myia, value_and_grad, ArithmeticData
from myia.dtype import Array
# The following import installs custom tracebacks for inference errors
from myia.debug import traceback  # noqa
from myia.api import to_device


###########
# Options #
###########


dtype = 'float32'

backend = 'pytorch'
# backend = 'nnvm'  # Uncomment to use nnvm backend
# backend = 'relay'  # Uncomment to use relay backend

#device_type = 'cpu'
device_type = 'cuda'  # Uncomment to run on the gpu

backend_options_dict = {
    'pytorch': {'device': device_type},
    'nnvm': {'target': device_type, 'device_id': 0},
    'relay': {'target': device_type, 'device_id': 0}
    }

backend_options = backend_options_dict[backend]

########
# Data #
########


# This just generates random data so we don't have to load a real dataset,
# but the model will work just as well on a real dataset.


def param(R, *size):
    """Generates a random array using the generator R."""
    return numpy.array(R.rand(*size) * 2 - 1, dtype=dtype)


def generate_data(n, batch_size, input_size, target_size, sequence_size, backend, backend_options,
                  *, seed=91):
    """Generate inputs and targets.

    Generates n batches of samples of size input_size, matched with
    a single target.
    """
    R = RandomState(seed=seed)

    """
    return ([to_device(param(R, batch_size, input_size), backend, backend_options) for i in range(sequence_size)],
             to_device(param(R, batch_size, target_size), backend, backend_options))
             #"""

    return to_device(([param(R, batch_size, input_size) for i in range(sequence_size)], param(R, batch_size, target_size)), \
                        backend, backend_options)


def lstm_parameters(*layer_sizes, batch_size, seed=6666):
    """Generates parameters for a MLP given a list of layer sizes."""
    R = RandomState(seed=seed)
    i, h, *rest = layer_sizes

    W_i = param(R, i, h)
    W_f = param(R, i, h)
    W_c = param(R, i, h)
    W_o = param(R, i, h)

    R_i = param(R, h, h)
    R_f = param(R, h, h)
    R_c = param(R, h, h)
    R_o = param(R, h, h)

    b_i = param(R, 1, h)
    b_f = param(R, 1, h)
    b_c = param(R, 1, h)
    b_o = param(R, 1, h)

    s0 = numpy.zeros((batch_size, h), dtype=dtype)
    c0 = numpy.zeros((batch_size, h), dtype=dtype)

    parameters = [(
        W_i, W_f, W_c, W_o,
        R_i, R_f, R_c, R_o,
        b_i, b_f, b_c, b_o,
        s0, c0
    )]

    for i, o in zip((h, *rest[:-1]), rest):
        W = param(R, i, o)
        b = param(R, 1, o)
        parameters.append((W, b))

    return parameters

###############
# Hyperparams #
###############


#lr = getattr(numpy, dtype)(0.01)


#########
# Model #
#########


# We generate an LSTM.


def sigmoid(x):
    """Sigmoid activation function."""
    return (numpy.tanh(x) + 1) / 2


@dataclass(frozen=True)
class Linear(ArithmeticData):
    """Linear layer."""

    W: 'Weights array'
    b: 'Biases vector'

    def apply(self, input):
        """Apply the layer."""
        return input @ self.W + self.b


@dataclass(frozen=True)
class Tanh(ArithmeticData):
    """Tanh layer."""

    def apply(self, input):
        """Apply the layer."""
        return numpy.tanh(input)


@dataclass(frozen=True)
class LSTMLayer(ArithmeticData):
    """LSTM layer."""

    W_i: Array
    W_f: Array
    W_c: Array
    W_o: Array

    R_i: Array
    R_f: Array
    R_c: Array
    R_o: Array

    b_i: Array
    b_f: Array
    b_c: Array
    b_o: Array

    s0: Array
    c0: Array

    def step(self, x_t, h_tm1, c_tm1):
        """Run one LSTM step."""
        i_t = sigmoid((x_t @ self.W_i) + (h_tm1 @ self.R_i) + self.b_i)
        f_t = sigmoid((x_t @ self.W_f) + (h_tm1 @ self.R_f) + self.b_f)
        o_t = sigmoid((x_t @ self.W_o) + (h_tm1 @ self.R_o) + self.b_o)

        c_hat_t = numpy.tanh(
            (x_t @ self.W_c) + (h_tm1 @ self.R_c) + self.b_c
        )

        c_t = f_t * c_tm1 + i_t * c_hat_t
        h_t = o_t * numpy.tanh(c_t)

        return h_t, c_t

    def apply(self, x):
        """Apply the layer."""
        s = self.s0
        c = self.c0
        for e in x:
            s, c = self.step(e, s, c)
        # Maybe collect and return the full list of s outputs?
        return s


@dataclass(frozen=True)
class Sequential(ArithmeticData):
    """Sequential layer, applies all sub-layers in order."""

    layers: 'Tuple of layers'

    def apply(self, x):
        """Apply the layer."""
        for layer in self.layers:
            x = layer.apply(x)
        return x


def cost(model, x, target):
    """Square difference loss."""
    y = model.apply(x)
    diff = target - y
    return sum(diff * diff)


# @myia(backend_options={'target': device_type})
@myia(backend='pytorch', backend_options={'device': device_type}, return_backend=True)
#def step(model, x, y, lr):
def step(model, x, y):
    """Returns the loss and parameter gradients."""
    # value_and_grad will return cost(model, x, y) and dcost(...)/dmodel.
    # The 'model' argument can be omitted: by default the derivative wrt
    # the first argument is returned.
    _cost, dmodel = value_and_grad(cost, 'model')(model, x, y)
    #return _cost, model - dmodel * lr
    return _cost, model - dmodel


def run_helper(args, n, batch_size, layer_sizes):
    """Run a model with the specified layer sizes on n random batches.

    The first layer is an LSTM layer, the rest are linear+tanh.

    Arguments:
        iters: How many iters to run.
        n: Number of training batches to generate.
        batch_size: Number of samples per batch.
        layer_sizes: Sizes of the model's layers.
    """
    i = 0
    j = 0
    layers = []
    lstmp, *linp = lstm_parameters(*layer_sizes, batch_size=batch_size)
    layers.append(LSTMLayer(*lstmp))
    for W, b in linp:
        layers.append(Linear(W, b))
        layers.append(Tanh())
    model = Sequential(tuple(layers))

    if args.break_bm:
        print("break_bm")
        return

    model = to_device(model, backend, backend_options)

    if args.break_mod_on_d:
        print("break_mod_on_d")
        return

    lr = getattr(numpy, dtype)(args.lr)
    lr = to_device(lr, backend, backend_options)

    inp, target = generate_data(n, batch_size, layer_sizes[0], layer_sizes[-1], args.timesteps, backend, backend_options)

    if args.break_mod_on_d_and_gen_data:
        print("break_mod_on_d_and_gen_data")
        return

    times = []
    if args.cuda_sync:
        import torch
    for _ in range(args.iters):
        costs = []
        if args.cuda_sync:
            torch.cuda.synchronize()
        t0 = time.time()
        #cost, model = step(model, inp, target, lr)
        cost, model = step(model, inp, target)
        if args.break_cr1:
            print("break_cr1", time.time() - t0)
            break

        if args.break_cr2 and i==1:
            print("break_cr2", time.time() - t0)
            break

        if args.print_all_iters:
            print(i, cost.array)

        if args.cuda_sync:
            torch.cuda.synchronize()
        t_diff = time.time() - t0

        if args.break_cr1 or args.break_cr2:
            break

        i += 1
        if args.print_all_iters:
            print(i, t_diff, cost)

        if i > args.warmup: # first 99 steps are warmup
            times.append(t_diff)

    if not args.break_cr1 and not args.break_cr2:
        print("times", times)
        print("\nMyia LSTM Stats")
        print("Avg Time: ", sum(times)/len(times))
        print("Min Time: ", min(times))
        print("Max Time: ", max(times))

    if args.save_txt:
        filename = ""
        for arg in vars(args):
            filename += str(arg)
            filename += "="
            filename += str(getattr(args, arg))
            filename += "."

        str_out = "Myia LSTM Stats\n" + \
                  "Avg Time: " + str(sum(times)/len(times)) + \
                  "Min Time: " + str(min(times)) + \
                  "Max Time: " + str(max(times)) + \
                  "Times" + str(times)

        f = open(filename+".txt","w+")
        f.write(str_out)
        f.close()



# We do not currently run this test in the test suite because it is too
# expensive to run.

# def test_run():
#     """Run the model.

#     This function is run automatically in the test suite to check against
#     regressions, so keep a low number of narrow layers to make sure it runs
#     quickly.
#     """
#     run_helper(1, 1, 5, (10, 3))


def run_model(args):
    """Run the model."""
    run_helper(args, 10, 5, (args.lstm_input_size, args.lstm_hidden_size, 1))
