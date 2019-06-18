
from dat import mlp_parameters, mlp_data
import time
import numpy
from dataclasses import dataclass

from myia import myia, value_and_grad, ArithmeticData
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


###############
# Hyperparams #
###############


#lr = getattr(numpy, dtype)(0.01)


#########
# Model #
#########


# We generate a MLP model with some arbitrary number of layers and tanh
# activations.


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


@myia(backend=backend, backend_options=backend_options, return_backend=True)
def step(model, x, y):
#def step(model, x, y, _lr):
    """Returns the loss and parameter gradients.

    value_and_grad will return cost(model, x, y) and dcost(...)/dmodel.
    The 'model' argument can be omitted: by default the derivative wrt
    the first argument is returned.
    """
    _cost, dmodel = value_and_grad(cost, 'model')(model, x, y)
    #return _cost, model - dmodel * _lr
    return _cost, model - dmodel


def run_model(args):
    i = 0
    j = 0
    layers = []
    for W, b in mlp_parameters():
        layers.append(Linear(W, b))
        layers.append(Tanh())
    model = Sequential(tuple(layers))
    model = to_device(model, backend, backend_options)

    inp, target = mlp_data()
    inp = to_device(inp, backend, backend_options)
    target = to_device(target, backend, backend_options)

    lr = getattr(numpy, dtype)(args.lr)
    lr = to_device(lr, backend, backend_options)

    if args.break_mod_on_d_and_gen_data:
        print("break_mod_on_d_and_gen_data")
        return

    times = []
    if args.cuda_sync:
        import torch
    for _ in range(args.iters):
        if args.cuda_sync:
            torch.cuda.synchronize()
        t0 = time.time()
        cost, model = step(model, inp, target)
        #cost, model = step(model, inp, target, lr)

        if args.cuda_sync:
            torch.cuda.synchronize()
        t_diff = time.time() - t0

        if args.break_cr1:
            print("break_cr1", time.time() - t0)
            break

        i+=1
        if args.break_cr2 and i==2:
            print("break_cr2", time.time() - t0)
            break

        #"""
        if args.print_all_iters:
            print(i, t_diff, cost.array)
            #"""

        if i > args.warmup: # first 99 steps are warmup
            times.append(t_diff)

    print("times", times)
    print("\nMyia MLP Stats")
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

        str_out = "Myia MLP Stats\n" + \
                  "Avg Time: " + str(sum(times)/len(times)) + \
                  "Min Time: " + str(min(times)) + \
                  "Max Time: " + str(max(times)) + \
                  "Times" + str(times)


        f = open(filename+".txt","w+")
        f.write(str_out)
        f.close()



