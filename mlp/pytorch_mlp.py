
import numpy
import torch
from torch import nn
import time
import dat


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

cuda = torch.device('cuda')
cpu = torch.device('cpu')

dev = cuda
#dev = cpu


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


def run_model(args):
    #import dat

    i = 0
    j = 0
    layers = []
    for W, b in dat.mlp_parameters():
        layers.append(Linear(W, b))
        layers.append(Tanh())
    model = Sequential(tuple(layers))
    if args.dev is 'cuda':
        model.cuda()

    inp, target = dat.mlp_data()
    inp = torch.Tensor(inp)
    target = torch.Tensor(target)

    if args.dev is 'cuda':
        inp = inp.cuda()
        target = target.cuda()

    if args.break_mod_on_d_and_gen_data:
        print("break_mod_on_d_and_gen_data")
        return

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0)
    optimizer.zero_grad()

    if args.break_after_pt_optim:
        print("break_after_pt_optim")
        return

    times = []
    for _ in range(args.iters):
        if args.cuda_sync:
            torch.cuda.synchronize()
        t0 = time.time()
        loss = step(model, inp, target, optimizer)
        optimizer.step()
        
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

        if args.print_all_iters:
            #print(i, t_diff, loss)
            print(i, t_diff)
            pass

        if i > args.warmup: # first 99 steps are warmup
            times.append(t_diff)

    print("times", times)
    print("\nPyTorch MLP Stats")
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

        str_out = "PyTorch MLP Stats\n" + \
                  "Avg Time: " + str(sum(times)/len(times)) + \
                  "Min Time: " + str(min(times)) + \
                  "Max Time: " + str(max(times)) + \
                  "Times" + str(times)

        f = open(filename+".txt","w+")
        f.write(str_out)
        f.close()


def run_matmul():
    torch.cuda.synchronize()
    m1 = dat.big_matrix(5000, 5000, seed=56)
    m2 = dat.big_matrix(5000, 5000, seed=57)
    m1 = torch.Tensor(m1).cuda()
    m2 = torch.Tensor(m2).cuda()
    for i in range(10):
        t0 = time.time()
        newm = m1 @ m2
        torch.cuda.synchronize()
        print(time.time() - t0)


#if __name__ == '__main__':
#    run_model()
