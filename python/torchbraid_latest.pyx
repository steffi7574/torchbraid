# cython: profile=True
# cython: linetrace=True

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF

from mpi4py import MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi

import pickle # we need this for building byte packs

ctypedef PyObject _braid_App_struct
ctypedef _braid_App_struct* braid_App

# class BraidVector:
#   def __init__(self,tensor,level):
#     self.tensor_ = tensor
#     self.level_  = level
#     self.time_   = np.nan

#   def tensor(self):
#     return self.tensor_

#   def level(self):
#     return self.level_
  
#   def clone(self):
#     print("BraidVector -> clone() - start")
#     print("BraidVector -> clone() - tensor_ tpye: ",type(self.tensor_))
#     cl = BraidVector(self.tensor().clone(),self.level())
#     print("BraidVector -> clone() - end")
#     return cl

#   def setTime(self,t):
#     self.time_ = t

#   def getTime(self):
#     return self.time_


class BraidVector:

# cdef extern from *:
# cdef class MPIData:
#   cdef MPI.Comm comm
#   cdef int rank
#   cdef int size
  def __init__(self,tensor_tuple,level):
    # self.mpi_data = MPIData(comm)
    self.tensor_tuple_ = tensor_tuple
    self.level_  = level
    self.time_   = np.nan

  def tensors(self):
    return self.tensor_tuple_

  def level(self):
    return self.level_
  
  def clone(self):
    # prefix_rank = self.mpi_data.getRank()
    # print("BraidVector -> clone() - start", " Rank: ", prefix_rank)
    print("BraidVector -> clone() - start")
    # print("BraidVector -> clone() - tensor_ tpye: ",type(self.tensor_tuple_))
    cloned_tuple = tuple([each_tensor.clone() for each_tensor in self.tensors()])
    cl = BraidVector(cloned_tuple,self.level())
    # print("BraidVector -> clone() - end", " Rank: ", prefix_rank)
    print("BraidVector -> clone() - end")
    return cl

  def setTime(self,t):
    self.time_ = t

  def getTime(self):
    return self.time_


# Originally defined place

ctypedef PyObject _braid_Vector_struct
ctypedef _braid_Vector_struct *braid_Vector
##
# Define your Python Braid Vector

# to supress a warning from numpy
cdef extern from *:
  """
  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
  """
include "./braid.pyx"
include "./torchbraid_callbacks.pyx"

#  a python level module
##########################################################
cdef class MPIData:
  cdef MPI.Comm comm
  cdef int rank
  cdef int size

  def __cinit__(self,comm):
    self.comm = comm
    self.rank = self.comm.Get_rank()
    self.size = self.comm.Get_size()

  def getComm(self):
    return self.comm

  def getRank(self):
    return self.rank

  def getSize(self):
    return self.size
# helper class for the MPI communicator

class ODEBlock(nn.Module):
  def __init__(self,layer,dt):
    super(ODEBlock, self).__init__()

    self.dt = dt
    self.layer = layer

  def forward(self, x):
    return x + self.dt*self.layer(x)
# end ODEBlock


class RNN_Model(torch.nn.Module):

  def __init__(self,comm,basic_block,num_steps,hidden_size,num_layers,Tf,max_levels=1,max_iters=10,
                    coarsen=None,
                    refine=None):
    super(RNN_Model,self).__init__()

    # optional parameters
    self.max_levels  = max_levels
    self.max_iters   = max_iters

    self.print_level = 2
    self.nrelax = 0
    self.cfactor = 2

    num_steps = 1 # num_steps changed from num_procs to 1
    self.mpi_data = MPIData(comm)
    self.Tf = Tf
    self.local_num_steps = dict()
    self.local_num_steps[0] = num_steps
    self.num_steps = dict()
    self.num_steps[0] = num_steps*self.mpi_data.getSize()
    # self.num_steps[0] = 1

    self.dt = Tf/self.num_steps[0]
    self.t0_local = self.mpi_data.getRank()*num_steps*self.dt
    self.tf_local = (self.mpi_data.getRank()+1.0)*num_steps*self.dt

    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.basic_block = basic_block
    self.RNN_models = dict()
    self.RNN_models[0] = basic_block()

    # @Eric: why do we need to store the basic_block in index 0 of RNN_models?


    # self.layer_block = layer_block
    # self.layer_models = dict()
    # self.layer_models[0] = [layer_block() for i in range(self.local_num_steps[0])]
    # self.local_layers = torch.nn.Sequential(*self.layer_models[0])




    self.py_core = None
    self.x_final = None

    if coarsen==None or refine==None:
      assert(coarsen==refine) # both should be None
      self.refinement_on = False
    else:
      self.refinement_on = True

    self.coarsen = coarsen
    self.refine  = refine

    self.skip_downcycle = 0
    self.param_size = 0

  # end __init__
 
  def setPrintLevel(self,print_level):
    self.print_level = print_level

  def setNumRelax(self,relax):
    self.nrelax = relax

  def setCFactor(self,cfactor):
    self.cfactor = cfactor

  def setSkipDowncycle(self,skip):
    self.skip_downcycle = skip

  def getMPIData(self):
    return self.mpi_data

  # Which part calls this forword() in RNN_BasicBlock_parallel?
  # Answer: eval() function
  # What is the difference between this forward() and forword() in RNN_BasicBlock_parallel?

  def forward(self,x):
    # forward(x):
    #   set_sequence(x) # self.setInitial(x)
    #   h0 = 0
    #   c0 = 0
    #   setInitial(h0,c0)//
    #   braid_Drive() # my_step->eval
    #   hn,cn = getFinal()
    #   return (hn,cn)

    # Receive h_c and preprocess x HERE?

    prefix_rank = self.mpi_data.getRank()
    total_ranks   = self.mpi_data.getSize()
    comm_ = self.mpi_data.getComm()

    print("RNN_Model -> forward() - start", " Rank: %d" % prefix_rank)

    # self.setInitial(x)
    self.x = x

    h = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    c = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    # h = torch.ones(self.num_layers, x.size(0), self.hidden_size)
    # c = torch.ones(self.num_layers, x.size(0), self.hidden_size)
    # h = torch.rand(self.num_layers, x.size(0), self.hidden_size)
    # c = torch.rand(self.num_layers, x.size(0), self.hidden_size)

    # print("#########################################")
    # print("type of h: ", type(h), " Rank: %d" % prefix_rank)
    # print("type of c: ", type(c), " Rank: %d" % prefix_rank)
    # print("RNN_Model -> forward() - h:", h, " Rank: %d" % prefix_rank)
    # print("RNN_Model -> forward() - c:", c, " Rank: %d" % prefix_rank)
    # print("#########################################")

    # From @Eric
    # self.setInitial(BraidVector(h0,c0))

    # Option 1; Make a tuple of h and c tensor
    self.setInitial_g((h,c))


    print("RNN_Model -> forward() - error-1", " Rank: %d" % prefix_rank)

    if self.py_core==None:
      self.py_core = self.initCore()

    cdef PyBraid_Core py_core = <PyBraid_Core> self.py_core
    cdef braid_Core core = py_core.getCore()

    print("RNN_Model -> forward() - error-2", " Rank: %d" % prefix_rank)

    # Run Braid
    braid_Drive(core) # -> my_step -> eval

    # eval returns BraidVector((t_yh,t_yc),0)

    print("RNN_Model -> forward() - error-3", " Rank: %d" % prefix_rank)

    # Destroy Braid Core C-Struct
    braid_Destroy(core)

    print("RNN_Model -> forward() - error-4", " Rank: %d" % prefix_rank)


    # f = self.getFinal()
    f_h_c  = self.getFinal()
    # f = f_h_c.tensors()
    # hn,cn = f

    print("RNN_Model -> forward() - error-5", " Rank: %d" % prefix_rank)

    # Only process in case of my_rank = num_procs - 1

    # if prefix_rank == total_ranks-1:
    # hn,cn = f_h_c

    # print("#########################################")
    # print("type of f_h_c: ", type(f_h_c), " Rank: %d" % prefix_rank)
    # print("type of hn: ", type(hn), " Rank: %d" % prefix_rank)
    # print("type of cn: ", type(cn), " Rank: %d" % prefix_rank)
    # print("RNN_Model -> forward() - f_h_c:", f_h_c, " Rank: %d" % prefix_rank)
    # print("RNN_Model -> forward() - hn:", hn, " Rank: %d" % prefix_rank)
    # print("RNN_Model -> forward() - cn:", cn, " Rank: %d" % prefix_rank)
    # print("#########################################")

    print("RNN_Model -> forward() - end", " Rank: %d" % prefix_rank)

    f_h_c = comm_.bcast(f_h_c,root=total_ranks-1)

    # return (hn,cn)
    return f_h_c

  # end forward

  # def setInitial(self,x0):
  #   self.x0 = BraidVector(x0,0) # @Eric: level should be 0 or self.level()? answer: just 0 for now

  def setInitial_g(self,g0):
    self.g0 = BraidVector(g0,0) # @Eric: level should be 0 or self.level()?

  def buildInit(self,t):

    prefix_rank = self.mpi_data.getRank()

    print("RNN_Model -> buildInit - start", " Rank: %d" % prefix_rank)
    # print("RNN_Model -> buildInit - before clone() - g0 type: ",type(self.g0))
    g = self.g0.clone()
    # print("RNN_Model -> buildInit - after clone() - g type: ",type(g))
    if t>0:
      # t_g = g.tensor()
      # t_g[:] = 0.0
      t_h,t_c = g.tensors()
      t_h[:] = 0.0
      t_c[:] = 0.0
    print("RNN_Model -> buildInit - end", " Rank: %d" % prefix_rank)
    return g
    # return x

  # evaluate a single layer
  def eval(self,g0,tstart,tstop): # Currently g0 points to x0. First change it to g0. Where does my_step() call?
    # eval(h0,c0,time_step):
    #   x = lookup_seq(x,time_step) # time_step corresponds to sequence index
    #   hn,cn = basic_block(h0,c0,x)
    #   return hn,cn
    # Connecting h and c to g
    # g_1 := (h_1,c_1)
    # g_2 := (h_2,c_2)
    # g_3 := (h_3,c_3)
    # Braid's view point
    # g_2 = f(g_1,1)
    # g_3 = f(g_2,2)
    # We must define 'f'
    # f(g,i) = LSTM(d_i,(h_i,c_i))
    # In the code f == eval

    prefix_rank = self.mpi_data.getRank()

    with torch.no_grad():
      print("RNN_Model -> eval - start", " tstart: ", tstart, " Rank: %d" % prefix_rank)
      # ver4

      # print("#########################################")
      # print("type of g0: ", type(g0), " Rank: %d" % prefix_rank)

      t_g = g0.tensors()
      # print("#########################################")
      # print("type of t_g: ", type(t_g), " Rank: %d" % prefix_rank)

      t_h,t_c = t_g

      print("#########################################")
      print("type of t_h: ", type(t_h), " Rank: %d" % prefix_rank)
      print("type of t_c: ", type(t_c), " Rank: %d" % prefix_rank)
      print("RNN_Model -> eval - t_h:", t_h, " Rank: %d" % prefix_rank)
      print("RNN_Model -> eval - t_c:", t_c, " Rank: %d" % prefix_rank)
      print("#########################################")


      print("RNN_Model -> eval - error-1", " tstart: ", tstart, " Rank: %d" % prefix_rank)
      t_x = self.x
      # print("RNN_Model -> eval() - t_x: ", t_x) # checked it is the same as x_block[i] in the main script

      print("RNN_Model -> eval - error-2", " tstart: ", tstart, " Rank: %d" % prefix_rank)
      # _, (t_yh,t_yc) = self.basic_block(t_x,t_h,t_c)

      _, (t_yh,t_yc) = self.RNN_models[0](t_x,t_h,t_c)

      print("#########################################")
      print("type of t_yh: ", type(t_yh), " Rank: %d" % prefix_rank)
      print("type of t_yc: ", type(t_yc), " Rank: %d" % prefix_rank)
      print("RNN_Model -> eval - t_yh:", t_yh, " Rank: %d" % prefix_rank)
      print("RNN_Model -> eval - t_yc:", t_yc, " Rank: %d" % prefix_rank)
      print("#########################################")

      # print("RNN_Model -> eval - t_yh[-1]:", t_yh[-1], " Rank: %d" % prefix_rank)
      # print("RNN_Model -> eval - t_yc[-1]:", t_yc[-1], " Rank: %d" % prefix_rank)

      print("RNN_Model -> eval - end", " tstart: ", tstart, " Rank: %d" % prefix_rank)
      # return BraidVector((t_yh,t_yc),0) # TODO: @Eric: The returned value directly pass to x_final?
      # return BraidVector((t_yh[-1],t_yc[-1]),0)
      return BraidVector((t_yh,t_yc),0)
  #end eval()

  def access(self,t,u):
    prefix_rank = self.mpi_data.getRank()
    print("RNN_Model -> access - start", " Rank: %d" % prefix_rank)

    print("RNN_Model -> access - t", t, " Rank: %d" % prefix_rank)
    print("RNN_Model -> access - Tf", self.Tf, " Rank: %d" % prefix_rank)
    if t==self.Tf:
      self.x_final = u.clone()

    # print("RNN_Model -> access - type of self.x_final: ", type(self.x_final), " Rank: %d" % prefix_rank)
    # x_final_tensors = self.x_final.tensors()
    # print("RNN_Model -> access - type of x_final_tensors: ", type(x_final_tensors), " Rank: %d" % prefix_rank)
    
    # print("RNN_Model -> access - x_final_tensors: ",x_final_tensors, " Rank: %d" % prefix_rank)

    print("RNN_Model -> access - end", " Rank: %d" % prefix_rank)

  # x_final is not matched with BraidVector((t_yh[-1],t_yc[-1]),0) from eval()
  def getFinal(self):
    prefix_rank = self.mpi_data.getRank()
    # if prefix_rank > 0:
    print("RNN_Model -> getFinal - start", " Rank: %d" % prefix_rank)
    if self.x_final==None:
      return None
    # assert the level
    assert(self.x_final.level()==0)
    # print("#########################################")
    # print("RNN_Model -> getFinal - type of self.x_final: ", type(self.x_final), " Rank: %d" % prefix_rank)
    x_final_tensors = self.x_final.tensors()
    print("RNN_Model -> getFinal - type of x_final_tensors: ", type(x_final_tensors), " Rank: %d" % prefix_rank)
    print("RNN_Model -> getFinal - x_final_tensors: ",x_final_tensors, " Rank: %d" % prefix_rank)

    print("RNN_Model -> getFinal - end", " Rank: %d" % prefix_rank)
    # else:
     # x_final_tensors = self.x_final
    
    return x_final_tensors
    # return self.x_final


  def initCore(self):
    cdef braid_Core core
    cdef double tstart
    cdef double tstop
    cdef int ntime
    cdef MPI.Comm comm = self.mpi_data.getComm()
    cdef int rank = self.mpi_data.getRank()
    cdef braid_App app = <braid_App> self
    cdef braid_PtFcnStep  b_step  = <braid_PtFcnStep> my_step
    cdef braid_PtFcnInit  b_init  = <braid_PtFcnInit> my_init
    cdef braid_PtFcnClone b_clone = <braid_PtFcnClone> my_clone
    cdef braid_PtFcnFree  b_free  = <braid_PtFcnFree> my_free
    cdef braid_PtFcnSum   b_sum   = <braid_PtFcnSum> my_sum
    cdef braid_PtFcnSpatialNorm b_norm = <braid_PtFcnSpatialNorm> my_norm
    cdef braid_PtFcnAccess b_access = <braid_PtFcnAccess> my_access
    cdef braid_PtFcnBufSize b_bufsize = <braid_PtFcnBufSize> my_bufsize
    cdef braid_PtFcnBufPack b_bufpack = <braid_PtFcnBufPack> my_bufpack
    cdef braid_PtFcnBufUnpack b_bufunpack = <braid_PtFcnBufUnpack> my_bufunpack
    # cdef braid_PtFcnSCoarsen b_coarsen = <braid_PtFcnSCoarsen> my_coarsen
    # cdef braid_PtFcnSRefine  b_refine  = <braid_PtFcnSRefine> my_refine

    ntime = self.num_steps[0]
    # ntime = 1
    tstart = 0.0
    tstop = self.Tf

    braid_Init(comm.ob_mpi, comm.ob_mpi, 
               tstart, tstop, ntime, 
               app,
               b_step, b_init, 
               b_clone, b_free, 
               b_sum, b_norm, b_access, 
               b_bufsize, b_bufpack, b_bufunpack, 
               &core)

    # if self.refinement_on:
    #   braid_SetSpatialCoarsen(core,b_coarsen)
    #   braid_SetSpatialRefine(core,b_refine)
    # end if refinement_on

    # Set Braid options
    braid_SetMaxLevels(core, self.max_levels)
    braid_SetMaxIter(core, self.max_iters)
    braid_SetPrintLevel(core,self.print_level)
    braid_SetNRelax(core,-1,self.nrelax)
    braid_SetCFactor(core,-1,self.cfactor) # -1 implies chage on all levels
    braid_SetSkip(core,self.skip_downcycle)

    # store the c pointer
    py_core = PyBraid_Core()
    py_core.setCore(core)

    return py_core
  # end initCore

  def maxParameterSize(self):
    print("maxParameterSize starts")
    print("self.param_size: ",self.param_size)
    if self.param_size==0:
      # walk through the sublayers and figure
      # out the largeset size
      print("maxParameterSize error-1")
      # Check why for loop is not running
      for lm in self.RNN_models[0]:
        print("lm: ",lm)
        local_size = len(pickle.dumps(lm))
        print("local_size: ",local_size)
        self.param_size = max(local_size,self.param_size)
    
    return self.param_size
  # end maxParameterSize

# end RNN_Model

# Other helper functions (mostly for testing)
#################################

# This frees a an initial vector
# using the `my_free` function. 
def freeVector(app,u):
  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector c_u = <braid_Vector> u

  my_free(c_app,c_u)

# This builds a close of the initial vector
# using the `my_init` function called from 'c'
# def cloneInitVector(app):
#   cdef braid_App c_app = <PyObject*>app
#   cdef braid_Vector v_vec
#   my_init(c_app,0.0,&v_vec)
#   return (<object> v_vec).tensor()

# # This builds a close of the initial vector
# # using the `my_clone` 
# def cloneVector(app,x):
#   b_vec = BraidVector(x,0)

#   cdef braid_App c_app = <PyObject*>app
#   cdef braid_Vector c_x = <braid_Vector> b_vec
#   cdef braid_Vector v
#   my_clone(c_app,c_x,&v)

#   return (<object> v).tensor()

def addVector(app,alpha,ten_x,beta,ten_y):
  x = BraidVector(ten_x,0)
  y = BraidVector(ten_y,0)

  cdef braid_App c_app = <PyObject*>app
  cdef double dalpha = alpha
  cdef braid_Vector c_x = <braid_Vector>x
  cdef double dbeta  = beta
  cdef braid_Vector c_y = <braid_Vector>y

  my_sum(c_app,dalpha,c_x,dbeta,c_y)

def vectorNorm(app,ten_x):
  x = BraidVector(ten_x,0)

  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector c_x = <braid_Vector>x
  cdef double [1] norm = [ 0.0 ]
  
  my_norm(c_app,c_x,norm)

  return norm[0]

def bufSize(app):
  cdef braid_App c_app = <PyObject*>app
  cdef int [1] sz = [0]
  cdef braid_BufferStatus status = NULL
  
  my_bufsize(c_app,sz,status)

  # subtract the int size (for testing purposes)
  return sz[0]

def allocBuffer(app):
  cdef void * buffer = PyMem_Malloc(bufSize(app))
  return <object> buffer

def freeBuffer(app,obuffer):
  cdef void * buffer = <void*> obuffer
  PyMem_Free(buffer)

def pack(app,ten_vec,obuffer,level):
  vec = BraidVector(ten_vec,level)

  cdef braid_App c_app    = <PyObject*>app
  cdef braid_Vector c_vec = <braid_Vector> vec
  cdef braid_BufferStatus status = NULL
  cdef void * buffer = <void*> obuffer

  my_bufpack(c_app, c_vec, buffer,status)

def unpack(app,obuffer):
  cdef braid_App c_app    = <PyObject*>app
  cdef braid_Vector c_vec    
  cdef void * buffer = <void*> obuffer
  cdef braid_BufferStatus status = NULL
  
  my_bufunpack(c_app,buffer,&c_vec,status)

  vec = <object> c_vec
  tensors_vec = vec.tensors()
  vec_h, vec_c = tensors_vec
  return ((vec_h, vec_c), vec.level())
  # return (vec.tensor(),vec.level())



  # Serial RNN
  ###########################################
  # def forward(self, x, h_prev, c_prev):
  #   h0 = h_prev
  #   c0 = c_prev
  #   output, (hn, cn) = self.lstm(x, (h0, c0))
  #   return output, (hn, cn)


    # print(" ")
    # print(" ")
    # print("Inside of RNN Model")
    # print("num_ranks: ", self.mpi_data.getSize())
    # print("my_rank: ", self.mpi_data.getRank())
    # print("max_levels: ", self.max_levels)
    # print("max_iters: ", self.max_iters)
    # print(" ")
    # print(" ")


    # self.local_num_steps = dict()
    # self.local_num_steps[0] = num_steps
    # self.num_steps = dict()
    # self.num_steps[0] = num_steps*self.mpi_data.getSize()

    # self.dt = Tf/self.num_steps[0]
    # self.t0_local = self.mpi_data.getRank()*num_steps*self.dt
    # self.tf_local = (self.mpi_data.getRank()+1.0)*num_steps*self.dt
  
    # self.layer_block = layer_block
    # self.layer_models = dict()
    # self.layer_models[0] = [layer_block() for i in range(self.local_num_steps[0])]
    # self.local_layers = torch.nn.Sequential(*self.layer_models[0])
  
    # ResNet
    ###########################################
    # @Eric: The below variables (basic_block, layer_models, local_layers) are required only for layer-parallel ResNet?
    #        In RNN, we don't need local_layers (local blocks)
    #        Then which part of the code is related to each processor is taking each block
    # self.basic_block = basic_block
    # self.layer_models = dict()
    # self.layer_models[0] = [basic_block() for i in range(self.local_num_steps[0])]
    # self.local_layers = torch.nn.Sequential(*self.layer_models[0])

    # RNN
    ###########################################
    ###########################################
    # @Eric: In RNN, we don't need local_layers (a sequence of local blocks)
    #        Then which part of the code is related to each processor is taking each block

    # In RNN, there is only a single layer but every processor has to use the same weights (see getLayer())
    
    # Is RNN_models[0] needed for connecting sub-sequences (consecutive blocks)? Or is this only needed for each block? If so we don't need for loop
    # In RNN, we don't need RNN_models[0]
    # In RNN, we don't need local_blocks
    ###########################################
    # IMPORTANT: All the processors will execute the same code, so every processor has to define only one basic_block.
    ###########################################

  # # TODO: distribute data
  # ###########################################
  # on root
  #   MPI_Send (data, to all procs)
  # else
  #   MPI_Recv (data)
  # ###########################################

  # def distributeData(self,x_block_all):
  #   comm          = self.mpi_data.getComm()
  #   my_rank       = self.mpi_data.getRank()
  #   num_ranks     = self.mpi_data.getSize()
  #   build_seq_tag = 20
  #   if my_rank == 0:
  #     # x_block_all[image id][proc id]
  #     x = x_block_all[0][0]
  #     for i in range(1,comm.Get_size()):
  #       x = x_block_all[0][i] # torch.Size([1, 14, 28])
  #       comm.send(x,dest=i,tag=build_seq_tag) # what is tag=build_seq_tag for?
  #     return None
  #   else:
  #     x = comm.recv(source=0,tag=build_seq_tag)
  #     return x