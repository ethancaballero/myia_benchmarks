"""Example of an RNN in Myia.

Myia is still a work in progress, and this example may change in the future.
"""


import time

import numpy
from numpy.random import RandomState
import torch
from torch import nn
import time
#import dat


###########
# Options #
###########


#print(torch.cuda.is_available())
#print(torch.cuda.get_device_name(0))

cuda = torch.device('cuda')
cpu = torch.device('cpu')

dev = cuda
#dev = cpu

dtype = 'float32'

########
# Data #
########


# This just generates random data so we don't have to load a real dataset,
# but the model will work just as well on a real dataset.


def param(R, *size):
    """Generates a random array using the generator R."""
    return numpy.array(R.rand(*size) * 2 - 1, dtype=dtype)


def generate_data(args, n, batch_size, input_size, target_size, sequence_size, 
                  torch_tensor_wrap, *, seed=91):
    """Generate inputs and targets.

    Generates n batches of samples of size input_size, matched with
    a single target.
    """
    R = RandomState(seed=seed)
    if torch_tensor_wrap:
        if args.dev is 'cuda':
            return ([torch.Tensor(param(R, batch_size, input_size)).cuda() for i in range(sequence_size)],
                     torch.Tensor(param(R, batch_size, target_size)).cuda())
        else:
            return ([torch.Tensor(param(R, batch_size, input_size)) for i in range(sequence_size)],
                     torch.Tensor(param(R, batch_size, target_size)))
    else:
        return ([param(R, batch_size, input_size) for i in range(sequence_size)],
                 param(R, batch_size, target_size))


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
#lr = 0.01
#lr = 1.00


#########
# Model #
#########


# We generate an RNN.


def sigmoid(x):
    """Sigmoid activation function."""
    return (torch.tanh(x) + 1) / 2


class Linear(nn.Module):
    def __init__(self, W, b):
        super(Linear, self).__init__()
        self.W = nn.Parameter(torch.Tensor(W))
        self.b = nn.Parameter(torch.Tensor(b))

    def forward(self, input):
        return input @ self.W + self.b


class Tanh(nn.Module):
    def forward(self, input):
        return torch.tanh(input)

class RNNLayer(nn.Module):
    def __init__(self, W, R, b, h0):
        super(RNNLayer, self).__init__()

        self.W = nn.Parameter(torch.Tensor(W))
        self.R = nn.Parameter(torch.Tensor(R))
        self.b = nn.Parameter(torch.Tensor(b))
        self.h0 = nn.Parameter(torch.Tensor(h0))

    def step(self, x, h_tm1):
        """Run one RNN step."""
        return torch.tanh((x @ self.W) + (h_tm1 @ self.R) + self.b)

    def forward(self, x):
        """Apply the layer."""
        h = self.h0
        for e in x:
            h = self.step(e, h)
        # Maybe collect and return the full list of outputs?
        return h

class Sequential(nn.Module):
    def __init__(self, layers):
        super(Sequential, self).__init__()
        self.layers = layers
        for i, layer in enumerate(layers):
            self.add_module(f'mod{i}', layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input


def mse(value, target):
    diff = value - target
    return torch.sum(diff * diff)


def step(model, inp, target, optimizer):
    value = model(inp)
    loss = mse(value, target)
    optimizer.zero_grad()
    loss.backward()
    return loss


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

    if args.dev is 'cuda':
        model.cuda()

    if args.break_mod_on_d:
        print("break_mod_on_d")
        return

    inp, target = generate_data(args, n, batch_size, layer_sizes[0], layer_sizes[-1], args.timesteps, True)

    if args.break_mod_on_d_and_gen_data:
        print("break_mod_on_d_and_gen_data")
        return

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0)

    if args.break_after_pt_optim:
        print("break_after_pt_optim")
        return

    times = []
    for _ in range(args.iters):
        costs = []

        if args.cuda_sync:
            torch.cuda.synchronize()
        t0 = time.time()

        loss = step(model, inp, target, optimizer)
        optimizer.step()

        if args.break_cr1:
            print("break_cr1", time.time() - t0)
            break

        #j+=1
        if args.break_cr2 and j==2:
            print("break_cr2", time.time() - t0)
            break

        if args.print_all_iters:
            print(i, loss)

        if args.break_cr1 or args.break_cr2:
            break

        #c = sum(costs) / n
        if args.cuda_sync:
            torch.cuda.synchronize()
        t_diff = time.time() - t0
        i += 1
        if args.print_all_iters:
            print(i, t_diff, loss)

        if i > args.warmup: # first 99 steps are warmup
            times.append(t_diff)

    if not args.break_cr1 and not args.break_cr2:
        print("times", times)
        print("\nPyTorch RNN Stats")
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

        str_out = "PyTorch RNN Stats\n" + \
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

