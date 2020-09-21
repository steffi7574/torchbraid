#!/usr/bin/env python
import argparse
import torch
from math import pi, sin
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import gradcheck
import torchbraid
from mpi4py import MPI
# import torchbraid.utils


def root_print(rank,s):
  if rank==0:
    print(s)


class SineDataset(torch.utils.data.Dataset):
    """ Dataset for sine approximation
        x in [-pi,pi], y = sin(x) """

    def __init__(self, filename, size):
        self.x = []
        self.y = []
        self.length = size

        f = open(filename, "r")
        cnt = 1
        for line in f.readlines():

            words = line.split()
            self.x.append(np.float32(float(words[0])))
            self.y.append(np.float32(float(words[1])))

            cnt += 1
            if cnt > size:
                break
        f.close()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Opening layer maps x to network width
class OpenLayer(torch.nn.Module):
    def __init__(self, width):
        super(OpenLayer, self).__init__()
        self.width = width

    def forward(self,x):
        x = torch.repeat_interleave(x, repeats=self.width, dim=1)
        return x

# Closing layer takes the mean over network width
class ClosingLayer(torch.nn.Module):
    def __init__(self):
        super(ClosingLayer, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True) # take the mean of each example
        return x

# ResNet layer
class StepLayer(torch.nn.Module):
    def __init__(self, width):
        super(StepLayer, self).__init__()
        self.linearlayer = torch.nn.Linear(width, width)

    def forward(self, x):
        x = torch.tanh(self.linearlayer(x))
        return x

class SerialNet(torch.nn.Module):
    """ Network definition """
    def __init__(self, width, nlayers, Tstop):
        #Constructor
        super(SerialNet, self).__init__()

        self.stepsize = Tstop / float(nlayers)
        # Layers
        self.layers= torch.nn.ModuleList([StepLayer(width) for i in range(nlayers)])
        self.openlayer = OpenLayer(width)
        self.closinglayer = ClosingLayer()

    def forward(self, x):
        # Opening Layer
        x = self.openlayer(x)
        # Hidden layers
        for i, layer in enumerate(self.layers):
            x = x + self.stepsize * layer(x)
        # Closing layer
        # print("Serial NN(x) = ", x)
        x = self.closinglayer(x)
        return x


class ParallelNet(torch.nn.Module):
    def __init__(self, Tstop=10.0, width=4, local_steps=10, max_levels=1, max_iters=1, fwd_max_iters=0, print_level=0, braid_print_level=0, cfactor=4, fine_fcf=False, skip_downcycle=True, fmg=False):
        super(ParallelNet, self).__init__()

        # Create lambda function for normal step layer
        step_layer = lambda: StepLayer(width)

        # Create and store parallel net
        self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD, step_layer, local_steps, Tstop, max_levels=max_levels, max_iters=max_iters)

        # Set options
        if fwd_max_iters > 0:
            print("FWD_max_iter = ", fwd_max_iters)
            self.parallel_nn.setFwdMaxIters(fwd_max_iters)
        self.parallel_nn.setPrintLevel(print_level,True)
        self.parallel_nn.setPrintLevel(braid_print_level,False)
        self.parallel_nn.setCFactor(cfactor)
        self.parallel_nn.setSkipDowncycle(skip_downcycle)

        if fmg:
            self.parallel_nn.setFMG()
        self.parallel_nn.setNumRelax(1)         # FCF elsewehre
        if not fine_fcf:
            self.parallel_nn.setNumRelax(0,level=0) # F-Relaxation on the fine grid
        else:
            self.parallel_nn.setNumRelax(1,level=0) # F-Relaxation on the fine grid

        # this object ensures that only the LayerParallel code runs on ranks!=0
        compose = self.compose = self.parallel_nn.comp_op()

        # by passing this through 'compose' (mean composition: e.g. OpenFlatLayer o channels)
        # on processors not equal to 0, these will be None (there are no parameters to train there)
        self.openlayer = compose(OpenLayer,width)
        self.closinglayer = compose(ClosingLayer)


    def forward(self, x):
        # by passing this through 'o' (mean composition: e.g. self.open_nn o x)
        # this makes sure this is run on only processor 0
        x = self.compose(self.openlayer,x)
        x = self.parallel_nn(x)
        # print("Parallel NN(x) = ", x)
        x = self.compose(self.closinglayer,x)

        return x
# end ParallelNet


# Compute L-2 norm gradient of model parameters
def gradnorm(parameters):
    norm = 0.0

    # print("Trainable parameters:")
    for p in parameters:
            #print(name, p.data, p.requires_grad)
            param_norm = p.grad.data.norm(2)
            norm += param_norm.item() ** 2
    norm = norm ** (1. / 2)
    return norm


# Parse command line
parser = argparse.ArgumentParser(description='TORCHBRAID Sine Example')
parser.add_argument('--force-lp', action='store_true', default=False, help='Use layer parallel even if there is only 1 MPI rank')
parser.add_argument('--epochs', type=int, default=2, metavar='N', help='number of epochs to train (default: 2)')
parser.add_argument('--batch-size', type=int, default=20, metavar='N', help='batch size for training (default: 50)')
args = parser.parse_args()


# MPI Stuff
rank  = MPI.COMM_WORLD.Get_rank()
procs = MPI.COMM_WORLD.Get_size()


# some logic to default to Serial if on one processor,
# can be overriden by the user to run layer-parallel
if args.force_lp:
    force_lp = True
elif procs>1:
    force_lp = True
else:
    force_lp = False

# Set a seed for reproducability
torch.manual_seed(0)

# Specify network
width = 4
nlayers = 10
Tstop = 10.0

# Specify and training params
batch_size = args.batch_size
max_epochs = args.epochs
learning_rate = 1e-3


# Get sine data
ntraindata = 20
nvaldata = 20
training_set = SineDataset("./xy_train.dat", ntraindata)
training_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)
validation_set = SineDataset("./xy_val.dat", nvaldata)
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)


# Serial network model
torch.manual_seed(0)
if not force_lp:
    root_print(rank, "Building serial net")
    model = SerialNet(width, nlayers, Tstop)
    compose = lambda op,*p: op(*p)    # NO IDEA QHT THAT IS
else:
    root_print(rank, "Building parallel net")
    # Layer-parallel parameters
    lp_max_levels = 1
    lp_max_iter = 10
    lp_printlevel = 1
    lp_braid_printlevel = 1
    lp_cfactor = 2
    # Number of local steps
    local_steps  = int(nlayers / procs)
    if nlayers % procs != 0:
        print(rank,'NLayers must be an even multiple of the number of processors: %d %d' % (nlayers, procs) )
        stop

    # Create layer parallel network
    model = ParallelNet(Tstop=Tstop,
                        width=width,
                        local_steps=local_steps,
                        max_levels=lp_max_levels,
                        max_iters=lp_max_iter,
                        fwd_max_iters=10,
                        print_level=lp_printlevel,
                        braid_print_level=lp_braid_printlevel,
                        cfactor=lp_cfactor,
                        fine_fcf=False,
                        skip_downcycle=False,
                        fmg=False)
    compose = model.compose   # NOT SO SURE WHAT THAT DOES

# Construct loss function
myloss = torch.nn.MSELoss(reduction='sum')

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(max_epochs):

    # TRAINING SET: Train one epoch
    for local_batch, local_labels in training_generator:
        local_batch = local_batch.reshape(len(local_batch),1)
        local_labels= local_labels.reshape(len(local_labels),1)

        # Forward pass
        ypred = model(local_batch)
        loss = compose(myloss, ypred, local_labels)

        # Comput gradient through backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Print gradients
        # for p in model.parameters():
            # print("Serial grad: ", p.grad.data)

        # Update network parameters
        optimizer.step()

    # VALIDATION
    for local_batch, local_labels in validation_generator:
        with torch.no_grad():

            local_batch = local_batch.reshape(len(local_batch),1)
            local_labels= local_labels.reshape(len(local_labels),1)
            ypred = model(local_batch)
            loss_val = compose(myloss, ypred, local_labels).item()


    # Output and stopping
    with torch.no_grad():
        gnorm = gradnorm(model.parameters())
        print(rank,epoch, loss.item(), loss_val, gnorm)

    # Stopping criterion
    if gnorm < 1e-4:
        break



# # plot validation and training
# xtrain = torch.tensor(training_set[0:len(training_set)])[0].reshape(len(training_set),1)
# ytrain = model(xtrain).detach().numpy()
# xval = torch.tensor(validation_set[0:len(validation_set)])[0].reshape(len(validation_set),1)
# yval = model(xval).detach().numpy()
#
# if rank == 0:
#     plt.plot(xtrain, ytrain, 'ro')
#     plt.plot(xval, yval, 'bo')
#     # Groundtruth
#     xtruth = np.arange(-pi, pi, 0.1)
#     plt.plot(xtruth, np.sin(xtruth))
#
#     # Shot the plot
#     plt.legend(['training', 'validation', 'groundtruth'])
#     # plt.show()
