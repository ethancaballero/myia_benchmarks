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
#lr = 0.01
#lr = 1.00


#########
# Model #
#########


# We generate an LSTM.


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

class LSTMLayer(nn.Module):
    def __init__(self, W_i, W_f, W_c, W_o,
                 R_i, R_f, R_c, R_o,
                 b_i, b_f, b_c, b_o,
                 s0, c0):
        super(LSTMLayer, self).__init__()

        self.W_i = nn.Parameter(torch.Tensor(W_i))
        self.W_f = nn.Parameter(torch.Tensor(W_f))
        self.W_c = nn.Parameter(torch.Tensor(W_c))
        self.W_o = nn.Parameter(torch.Tensor(W_o))

        self.R_i = nn.Parameter(torch.Tensor(R_i))
        self.R_f = nn.Parameter(torch.Tensor(R_f))
        self.R_c = nn.Parameter(torch.Tensor(R_c))
        self.R_o = nn.Parameter(torch.Tensor(R_o))

        self.b_i = nn.Parameter(torch.Tensor(b_i))
        self.b_f = nn.Parameter(torch.Tensor(b_f))
        self.b_c = nn.Parameter(torch.Tensor(b_c))
        self.b_o = nn.Parameter(torch.Tensor(b_o))

        self.s0 = nn.Parameter(torch.Tensor(s0))
        self.c0 = nn.Parameter(torch.Tensor(c0))

    def step(self, x_t, h_tm1, c_tm1):
        """Run one LSTM step."""
        i_t = sigmoid((x_t @ self.W_i) + (h_tm1 @ self.R_i) + self.b_i)
        f_t = sigmoid((x_t @ self.W_f) + (h_tm1 @ self.R_f) + self.b_f)
        o_t = sigmoid((x_t @ self.W_o) + (h_tm1 @ self.R_o) + self.b_o)

        c_hat_t = torch.tanh(
            (x_t @ self.W_c) + (h_tm1 @ self.R_c) + self.b_c
        )

        c_t = f_t * c_tm1 + i_t * c_hat_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

    def forward(self, x):
        """Apply the layer."""
        s = self.s0
        c = self.c0
        for e in x:
            s, c = self.step(e, s, c)
        # Maybe collect and return the full list of s outputs?
        return s

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
        #for __ in range(n):
        loss = step(model, inp, target, optimizer)
        optimizer.step()

        if args.break_cr1:
            print("break_cr1", time.time() - t0)
            break

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
        #torch.cuda.synchronize()
        i += 1
        if args.print_all_iters:
            print(i, t_diff, loss)

        if i > args.warmup: # first 99 steps are warmup
            times.append(t_diff)

    if not args.break_cr1 and not args.break_cr2:
        print("times", times)
        print("\nPyTorch LSTM Stats")
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

        str_out = "PyTorch LSTM Stats\n" + \
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

