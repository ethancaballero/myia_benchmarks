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


def rnn_parameters(*layer_sizes, batch_size, seed=123123):
    """Generates parameters for an RNN given a list of layer sizes."""
    R = RandomState(seed=seed)
    i, h, *rest = layer_sizes

    W = param(R, i, h)
    U = param(R, h, h)
    b = param(R, 1, h)
    h0 = param(R, batch_size, h)
    parameters = [(W, U, b, h0)]

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


# We generate an RNN.


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
class RNNLayer(ArithmeticData):
    """RNN layer."""

    W: 'Input to state weights'
    R: 'State transition weights'
    b: 'Biases vector'
    h0: 'Initial state'

    def step(self, x, h_tm1):
        """Run one RNN step."""
        return numpy.tanh((x @ self.W) + (h_tm1 @ self.R) + self.b)

    def apply(self, x):
        """Apply the layer."""
        h = self.h0
        for e in x:
            h = self.step(e, h)
        # Maybe collect and return the full list of outputs?
        return h


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

    The first layer is an RNN layer, the rest are linear+tanh.

    Arguments:
        iters: How many iters to run.
        n: Number of training batches to generate.
        batch_size: Number of samples per batch.
        layer_sizes: Sizes of the model's layers.
    """
    i = 0
    j = 0
    layers = []
    rnnp, *linp = rnn_parameters(*layer_sizes, batch_size=batch_size)
    layers.append(RNNLayer(*rnnp))
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
        print("\nMyia RNN Stats")
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

        str_out = "Myia RNN Stats\n" + \
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
