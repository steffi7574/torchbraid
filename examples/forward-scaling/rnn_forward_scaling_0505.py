import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import torchbraid
import time

import getopt,sys
import argparse

from mpi4py import MPI

# only print on rank==0
def root_print(rank,s):
  if rank==0:
    print(s)

# LSTM tutorial: https://pytorch.org/docs/stable/nn.html

class RNN_BasicBlock(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    super(RNN_BasicBlock, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers

    torch.manual_seed(20)
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    # for param in self.lstm.parameters():
    #   print(param.data)

  def __del__(self):
    pass

  def forward(self, x, h_prev, c_prev):
    # Set initial hidden and cell states
    h0 = h_prev
    c0 = c_prev

    output, (hn, cn) = self.lstm(x, (h0, c0))

    return output, (hn, cn)

class RNN_BasicBlock_parallel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    super(RNN_BasicBlock_parallel, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers

    # manual_seed enables different basic_blocks in different procs to use the same initial weights
    torch.manual_seed(20)
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    # for param in self.lstm.parameters():
    #   print(param.data)

  def __del__(self):
    pass

  def forward(self, x):
    # Set initial hidden and cell states
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

    output, (hn, cn) = self.lstm(x, (h0, c0))

    return output, (hn, cn)


class ODEBlock(nn.Module):
  def __init__(self,layer,dt):
    super(ODEBlock, self).__init__()

    self.dt = dt
    self.layer = layer

  def forward(self, x):
    return x + self.dt*self.layer(x)
    

def RNN_build_block_with_dim(input_size, hidden_size, num_layers):
  b = RNN_BasicBlock(input_size, hidden_size, num_layers) # channels = hidden_size
  return b

def preprocess_input_data_serial(num_chunks, num_batch, batch_size, channels, sequence_length, input_size):
  torch.manual_seed(20)
  x = torch.randn(num_batch,batch_size,channels,sequence_length,input_size)

  data_all = []
  seq_split_all = []
  for i in range(len(x)):
    image = x[i].reshape(-1, sequence_length, input_size)
    images_split = torch.chunk(image, num_chunks, dim=1)
    seq_split = []
    for cnk in images_split:
      seq_split.append(cnk)
    seq_split_all.append(seq_split)
    data_all.append(image)

  # print("serial proc 0 seq_split_all - image 0: ", seq_split_all[0][0]) # = x_block[0] on proc 0
  # print("serial proc 1 seq_split_all - image 0: ", seq_split_all[0][1]) # = x_block[0] on proc 1

  return data_all, seq_split_all


def RNN_build_block_with_dim_parallel(input_size, hidden_size, num_layers):
  b = RNN_BasicBlock_parallel(input_size, hidden_size, num_layers) # channels = hidden_size
  return b

def preprocess_distribute_input_data_parallel(rank,num_procs,num_batch,batch_size,channels,sequence_length,input_size):
  if rank == 0:
    torch.manual_seed(20)
    x_parallel = torch.randn(num_batch,batch_size,channels,sequence_length,input_size)

    # x_block_all[total_images][total_blocks]
    x_block_all = []
    for i in range(len(x_parallel)):
      image = x_parallel[i].reshape(-1,sequence_length,input_size)
      data_split = torch.chunk(image, num_procs, dim=1)
      seq_split = []
      for blk in data_split:
        seq_split.append(blk)
      x_block_all.append(seq_split)

    # print("proc 0 x_block_all - image 0: ", x_block_all[0][0]) # = x_block[0] on proc 0
    # print("proc 1 x_block_all - image 0: ", x_block_all[0][1]) # = x_block[0] on proc 1

    x_block = []
    for image_id in range(len(x_block_all)):
      x_block.append(x_block_all[image_id][rank])
    # print("x_block[0] size:",x_block[0].shape)
    # print("x_block[9] size:",x_block[9].shape)

    for block_id in range(1,num_procs):
      x_block_tmp = []
      for image_id in range(len(x_block_all)):
        x_block_tmp.append(x_block_all[image_id][block_id])
      comm.send(x_block_tmp,dest=block_id,tag=20)

    return x_block
  
  else:
    x_block = comm.recv(source=0,tag=20)

    return x_block


# some default input arguments
###########################################

comm = MPI.COMM_WORLD
my_rank   = comm.Get_rank()
last_rank = comm.Get_size()-1

# some default input arguments
###########################################
max_levels      = 3
max_iters       = 1
local_num_steps = 5
num_steps       = int(local_num_steps*comm.Get_size())
# For CNN
###########################################
# channels        = 16
# images          = 10
# image_size      = 256

# For RNN
###########################################
channels        = 1
images          = 10
image_size      = 28

Tf              = 2.0
run_serial      = False
print_level     = 0
nrelax          = 1
cfactor         = 2

# parse the input arguments
###########################################

parser = argparse.ArgumentParser()
parser.add_argument("steps",type=int,help="total number of steps, must be product of proc count (p=%d)" % comm.Get_size())
parser.add_argument("--levels",    type=int,  default=max_levels,   help="maximum number of Layer-Parallel levels")
parser.add_argument("--iters",     type=int,   default=max_iters,   help="maximum number of Layer-Parallel iterations")
parser.add_argument("--channels",  type=int,   default=channels,    help="number of convolutional channels")
parser.add_argument("--images",    type=int,   default=images,      help="number of images")
parser.add_argument("--pxwidth",   type=int,   default=image_size,  help="Width/height of images in pixels")
parser.add_argument("--verbosity", type=int,   default=print_level, help="The verbosity level, 0 - little, 3 - lots")
parser.add_argument("--cfactor",   type=int,   default=cfactor,     help="The coarsening factor")
parser.add_argument("--nrelax",    type=int,   default=nrelax,      help="The number of relaxation sweeps")
parser.add_argument("--tf",        type=float, default=Tf,          help="final time for ODE")
parser.add_argument("--serial",  default=run_serial, action="store_true", help="Run the serial version (1 processor only)")
parser.add_argument("--optstr",  default=False,      action="store_true", help="Output the options string")
args = parser.parse_args()

# the number of steps is not valid, then return
if not args.steps % comm.Get_size()==0:
  if my_rank==0:
    print('error in <steps> argument, must be a multiple of proc count: %d' % comm.Get_size())
    parser.print_help()
  sys.exit(0)
# end if not args.steps

if args.serial==True and comm.Get_size()!=1:
  if my_rank==0:
    print('The <--serial> optional argument, can only be run in serial (proc count: %d)' % comm.Get_size())
    parser.print_help()
  sys.exit(0)
# end if not args.steps
   
# determine the number of steps
num_steps       = args.steps
local_num_steps = int(num_steps/comm.Get_size())

if args.levels:    max_levels  = args.levels
if args.iters:     max_iters   = args.iters
if args.channels:  channels    = args.channels
if args.images:    images      = args.images
if args.pxwidth:   image_size  = args.pxwidth
if args.verbosity: print_level = args.verbosity
if args.cfactor:   cfactor     = args.cfactor
if args.nrelax :   nrelax      = args.nrelax
if args.tf:        Tf          = args.tf
if args.serial:    run_serial  = args.serial

class Options:
  def __init__(self):
    self.num_procs   = comm.Get_size()
    self.num_steps   = args.steps
    self.max_levels  = args.levels
    self.max_iters   = args.iters
    self.channels    = args.channels
    self.images      = args.images
    self.image_size  = args.pxwidth
    self.print_level = args.verbosity
    self.cfactor     = args.cfactor
    self.nrelax      = args.nrelax
    self.Tf          = args.tf
    self.run_serial  = args.serial

  def __str__(self):
    s_net = 'net:ns=%04d_ch=%04d_im=%05d_is=%05d_Tf=%.2e' % (self.num_steps,
                                                             self.channels,
                                                             self.images,
                                                             self.image_size,
                                                             self.Tf)
    s_alg = '__alg:ml=%02d_mi=%02d_cf=%01d_nr=%02d' % (self.max_levels,
                                                       self.max_iters,
                                                       self.cfactor,
                                                       self.nrelax)
    return s_net+s_alg

opts_obj = Options()

if args.optstr==True:
  if comm.Get_rank()==0:
    print(opts_obj)
  sys.exit(0)
    
print(opts_obj)

# set hyper-parameters for RNN
###########################################
sequence_length = 28 # total number of time steps for each sequence
input_size = 28 # input size for each time step in a sequence
hidden_size = 20
num_layers = 2
batch_size = 1

# build parallel information
dt        = Tf/num_steps

# generate randomly initialized data
###########################################
num_batch = int(images / batch_size)

# For MNIST data later
###########################################
# for i, (images, labels) in enumerate(train_loader):
#   images = images.reshape(-1, sequence_length, input_size)
# train_loader.images: torch.Size([batch_size, channels, sequence_length, input_size])
# train_loader.images.reshape(-1, sequence_length, input_size): torch.Size([batch_size, sequence_length, input_size])


root_print(my_rank,'Number of steps: %d' % num_steps)
root_print(my_rank,'Number of local steps: %d' % local_num_steps)

# Sequential version
###########################################
if run_serial:

  root_print(my_rank,'Running PyTorch: %d' % comm.Get_size())

  # build the neural network
  ###########################################
  basic_block = lambda: RNN_build_block_with_dim(input_size, hidden_size, num_layers)

  serial_rnn = basic_block()

  num_chunks = 2 # = num_steps

  # preprocess input data
  ###########################################
  image_all, seq_split_all = preprocess_input_data_serial(num_chunks,num_batch,batch_size,channels,sequence_length,input_size)

  with torch.no_grad():
    t0_parallel = time.time()

    # for i in range(len(x)):
    for i in range(1):

      # Serial version 1
      ###########################################
      # forward pass
      y_serial_hn = torch.zeros(num_layers, image_all[i].size(0), hidden_size)
      y_serial_cn = torch.zeros(num_layers, image_all[i].size(0), hidden_size)

      y_serial_output, (y_serial_hn, y_serial_cn) = serial_rnn(image_all[i],y_serial_hn,y_serial_cn)

      # Serial version 2
      ###########################################
      # assume that there are two steps (blocks)

      # forward pass
      for j in range(num_chunks):
        if j == 0: # in case of the first chunk, use zero values for initial hidden and cell states
          y_serial_prev_hn = torch.zeros(num_layers, seq_split_all[i][j].size(0), hidden_size)
          y_serial_prev_cn = torch.zeros(num_layers, seq_split_all[i][j].size(0), hidden_size)

        y_serial_output_with_chunks, (y_serial_prev_hn, y_serial_prev_cn) = serial_rnn(seq_split_all[i][j],y_serial_prev_hn,y_serial_prev_cn)

      # compare the output from serial version 1 (without chunk) with the output from serial version 2 (with two chunks)
      print(" ")
      print(" ")
      print("Serial version 1 - y_serial_output size: ", y_serial_output.shape) # torch.Size([1, 28, 20])
      print(y_serial_output.data[0][-1])
      print("Serial version 2 - y_serial_output_with_chunks size: ", y_serial_output_with_chunks.shape) # torch.Size([1, 28, 20])
      print(y_serial_output_with_chunks.data[0][-1])

      print(" ")
      print(" ")
      print("Serial version 1 - y_serial_hn size: ", y_serial_hn.shape)
      print(y_serial_hn.data[1])
      print("Serial version 2 - y_serial_prev_hn size: ", y_serial_prev_hn.shape)
      print(y_serial_prev_hn.data[1])

      print(" ")
      print(" ")
      print("Serial version 1 - y_serial_cn size: ", y_serial_cn.shape)
      print(y_serial_cn.data[1])
      print("Serial version 2 - y_serial_prev_cn size: ", y_serial_prev_cn.shape)
      print(y_serial_prev_cn.data[1])

    tf_parallel = time.time()


# Parallel version
###########################################
else:
  root_print(my_rank,'Running TorchBraid: %d' % comm.Get_size())
  # build the parallel neural network
  ###########################################
  # Building a basic_block in each processor
  # In this case, the weights of basic_blocks for every processor are initialied with the same random values
  ###########################################
  basic_block_parallel = lambda: RNN_build_block_with_dim_parallel(input_size, hidden_size, num_layers)
  
  num_procs = comm.Get_size()
  print("num_procs: ",num_procs)

  # preprocess and distribute input data
  ###########################################
  x_block = preprocess_distribute_input_data_parallel(my_rank,num_procs,num_batch,batch_size,channels,sequence_length,input_size)

  max_levels = 1 # for testing parallel rnn
  max_iters = 1 # for testing parallel rnn

  parallel_nn = torchbraid.RNN_Model(comm,basic_block_parallel,num_procs,Tf,max_levels=max_levels,max_iters=max_iters)

  parallel_nn.setPrintLevel(print_level)
  parallel_nn.setCFactor(cfactor)
  parallel_nn.setNumRelax(nrelax)
  t0_parallel = time.time()
  # parallel_nn(x_block)
  # y_parallel_output, (y_parallel_hn, y_parallel_cn) = parallel_nn(x_block)
  y_parallel_output = parallel_nn(x_block)
  comm.barrier()

  tf_parallel = time.time()
  comm.barrier()

# end if not run_serial

root_print(my_rank,'Run    Time: %.6e' % (tf_parallel-t0_parallel))
