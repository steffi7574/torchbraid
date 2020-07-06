import torch
import numpy as np
import math
cimport numpy as np

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF

from mpi4py import MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi

# cdef class MPIData_:
#   cdef MPI.Comm comm_

##
# Define your Python Braid Vector as a C-struct

cdef int my_access(braid_App app,braid_Vector u,braid_AccessStatus status):
  # pyApp = <object> app

  cdef int prefix_rank
  cdef int prefix_size

  pyApp = <object> app

  mpi_data_ = pyApp.getMPIData()
  prefix_rank = mpi_data_.getRank()
  prefix_size = mpi_data_.getSize()

  print("callbakcs.pyx -> my_access - start", " Rank: %d" % prefix_rank)

  cdef double t

  # Create Numpy wrapper around u.v
  ten_u = <object> u

  braid_AccessStatusGetT(status, &t)

  pyApp.access(t,ten_u)

  print("callbakcs.pyx -> my_access - end", " Rank: %d" % prefix_rank)
  return 0


cdef int my_step(braid_App app, braid_Vector ustop, braid_Vector fstop, braid_Vector vec_u, braid_StepStatus status):
  
  # cdef MPI.Comm comm_
  # rank_ = comm_.Get_rank()
  # size_ = comm_.Get_size()

  cdef int prefix_rank
  cdef int prefix_size

  pyApp = <object> app

  mpi_data_ = pyApp.getMPIData()
  prefix_rank = mpi_data_.getRank()
  prefix_size = mpi_data_.getSize()

  # print("callbakcs.pyx -> my_step - size: ", prefix_size)
  # print("callbakcs.pyx -> my_step - Rank: ", prefix_rank)

  print("callbakcs.pyx -> my_step - start", " Rank: %d" % prefix_rank)
  print("callbakcs.pyx -> my_step - error-1", " Rank: %d" % prefix_rank)
  u =  <object> vec_u # vec_u was originally x0 but now it should be g0.
  print("**************************************************")
  print("callbakcs.pyx -> my_step - u type: ",type(u), " Rank: %d" % prefix_rank)
  print("**************************************************")
  print("callbakcs.pyx -> my_step - error-2", " Rank: %d" % prefix_rank)
  cdef double tstart
  cdef double tstop
  tstart = 0.0
  tstop = 5.0
  braid_StepStatusGetTstartTstop(status, &tstart, &tstop)
  print("callbakcs.pyx -> my_step - error-3", " Rank: %d" % prefix_rank)
  temp = pyApp.eval(u,tstart,tstop) # u was originally x0 but now it should be g0.
  print("callbakcs.pyx -> my_step - error-4", " Rank: %d" % prefix_rank)
  print("**************************************************")
  print("callbakcs.pyx -> my_step - temp type: ",type(temp), " Rank: %d" % prefix_rank)
  print("**************************************************")

  # AttributeError: 'tuple' object has no attribute 'copy_'
  # u.tensors().copy_(temp.tensors())

  # Why we copy this and where do we store these?
  u_h,u_c = u.tensors()
  temp_h,temp_c = temp.tensors()
  u_h.copy_(temp_h)
  u_c.copy_(temp_c)

  print("callbakcs.pyx -> my_step - end", " Rank: %d" % prefix_rank)

  return 0
  
cdef int my_init(braid_App app, double t, braid_Vector *u_ptr):

  cdef int prefix_rank
  cdef int prefix_size

  pyApp = <object> app

  mpi_data_ = pyApp.getMPIData()
  prefix_rank = mpi_data_.getRank()
  prefix_size = mpi_data_.getSize()

  print("callbakcs.pyx -> my_init - start", " Rank: %d" % prefix_rank)
  # pyApp = <object> app
  print("callbakcs.pyx -> my_init - error-1", " Rank: %d" % prefix_rank)
  u_mem = pyApp.buildInit(t)
  print("callbakcs.pyx -> my_init - error-2", " Rank: %d" % prefix_rank)
  Py_INCREF(u_mem) # why do we need this?
  print("callbakcs.pyx -> my_init - error-3", " Rank: %d" % prefix_rank)
  u_ptr[0] = <braid_Vector> u_mem
  print("callbakcs.pyx -> my_init - end", " Rank: %d" % prefix_rank)

  return 0

cdef int my_free(braid_App app, braid_Vector u):
  # Cast u as a PyBraid_Vector
  pyU = <object> u
  # Decrement the smart pointer
  Py_DECREF(pyU)
  del pyU
  return 0

# TODO
# cdef int braid_my_sum():
cdef int my_sum(braid_App app, double alpha, braid_Vector x, double beta, braid_Vector y):
  # Cast x and y as a PyBraid_Vector
  # cdef np.ndarray[float,ndim=1] np_X = (<object> x).tensor().numpy().ravel()
  # cdef np.ndarray[float,ndim=1] np_Y = (<object> y).tensor().numpy().ravel()

  cdef int prefix_rank
  cdef int prefix_size

  pyApp = <object> app

  mpi_data_ = pyApp.getMPIData()
  prefix_rank = mpi_data_.getRank()
  prefix_size = mpi_data_.getSize()

  print("callbakcs.pyx -> my_sum - start", " Rank: %d" % prefix_rank)

  tensors_X = (<object> x).tensors()
  tensors_Y = (<object> y).tensors()

  print("callbakcs.pyx -> my_sum - tensors_X tpye: ",type(tensors_X), " Rank: %d" % prefix_rank)
  print("callbakcs.pyx -> my_sum - tensors_Y tpye: ",type(tensors_Y), " Rank: %d" % prefix_rank)

  print("====================================================")

  ten_X_h, ten_X_c = tensors_X
  ten_Y_h, ten_Y_c = tensors_Y

  print("callbakcs.pyx -> my_sum - ten_X_h tpye: ",type(ten_X_h), " Rank: %d" % prefix_rank)
  print("callbakcs.pyx -> my_sum - ten_X_c tpye: ",type(ten_X_c), " Rank: %d" % prefix_rank)
  print("callbakcs.pyx -> my_sum - ten_Y_h tpye: ",type(ten_Y_h), " Rank: %d" % prefix_rank)
  print("callbakcs.pyx -> my_sum - ten_Y_c tpye: ",type(ten_Y_c), " Rank: %d" % prefix_rank)

  print("====================================================")
  cdef np.ndarray[float,ndim=1] np_X_h = ten_X_h.numpy().ravel()
  cdef np.ndarray[float,ndim=1] np_X_c = ten_X_c.numpy().ravel()
  cdef np.ndarray[float,ndim=1] np_Y_h = ten_Y_h.numpy().ravel()
  cdef np.ndarray[float,ndim=1] np_Y_c = ten_Y_c.numpy().ravel()


  # tensors().numpy() won't be workable. please modify it. unpack tensors first. and then use numpy() for each tensor.
  # cdef np.ndarray[float,ndim=1] np_X_h = (<object> x).tensor().numpy().ravel()
  # cdef np.ndarray[float,ndim=1] np_X_c = (<object> x).tensor().numpy().ravel()

  # cdef np.ndarray[float,ndim=1] np_Y_h = (<object> y).tensor().numpy().ravel()
  # cdef np.ndarray[float,ndim=1] np_Y_c = (<object> y).tensor().numpy().ravel()

  # do operation with (np_X_h and np_Y_h), and (np_X_c and np_Y_c) separtely.

  # in place copy (this is inefficient because of the copy/allocation to ten_T
  # cdef int sz = len(np_X)
  # for k in range(sz):
  #   np_Y[k] = alpha*np_X[k]+beta*np_Y[k]

  cdef int sz = len(np_X_h)
  for k in range(sz):
    np_Y_h[k] = alpha*np_X_h[k]+beta*np_Y_h[k]
    np_Y_c[k] = alpha*np_X_c[k]+beta*np_Y_c[k]

  print("callbakcs.pyx -> my_sum - end", " Rank: %d" % prefix_rank)

  return 0

cdef int my_clone(braid_App app, braid_Vector u, braid_Vector *v_ptr):

  cdef int prefix_rank
  cdef int prefix_size

  pyApp = <object> app

  mpi_data_ = pyApp.getMPIData()
  prefix_rank = mpi_data_.getRank()
  prefix_size = mpi_data_.getSize()


  print("callbakcs.pyx -> my_clone - start", " Rank: %d" % prefix_rank)
  ten_U = <object> u
  print("callbakcs.pyx -> my_clone - error-1", " Rank: %d" % prefix_rank)
  v_mem = ten_U.clone()
  print("callbakcs.pyx -> my_clone - error-2", " Rank: %d" % prefix_rank)
  Py_INCREF(v_mem) # why do we need this?
  print("callbakcs.pyx -> my_clone - error-3", " Rank: %d" % prefix_rank)
  v_ptr[0] = <braid_Vector> v_mem
  print("callbakcs.pyx -> my_clone - end", " Rank: %d" % prefix_rank)

  return 0

# TODO
# cdef int braid_my_norm():
cdef int my_norm(braid_App app, braid_Vector u, double *norm_ptr):


  cdef int prefix_rank
  cdef int prefix_size

  pyApp = <object> app

  mpi_data_ = pyApp.getMPIData()
  prefix_rank = mpi_data_.getRank()
  prefix_size = mpi_data_.getSize()

  print("callbakcs.pyx -> my_norm - start", " Rank: %d" % prefix_rank)


  # Compute norm 
  # ten_U = (<object> u).tensor()
  # norm_ptr[0] = torch.norm(ten_U)
  tensors_U = (<object> u).tensors()
  norm_ptr[0] = 0.0
  for ten_U in tensors_U:
    norm_ptr[0] += torch.norm(ten_U)**2

  print("callbakcs.pyx -> my_norm - error-1", " Rank: %d" % prefix_rank)

  math.sqrt(norm_ptr[0])

  print("callbakcs.pyx -> my_norm - end", " Rank: %d" % prefix_rank)

  return 0

cdef int my_bufsize(braid_App app, int *size_ptr, braid_BufferStatus status):

  # pyApp = <object> app

  cdef int prefix_rank
  cdef int prefix_size

  pyApp = <object> app

  mpi_data_ = pyApp.getMPIData()
  prefix_rank = mpi_data_.getRank()
  prefix_size = mpi_data_.getSize()

  print("callbakcs.pyx -> my_bufsize - start", " Rank: %d" % prefix_rank)

  # cdef int cnt = pyApp.x0.tensor().size().numel()

  cdef int cnt = 0
  tensors_g0 = (<object> pyApp.g0).tensors()
  for ten_g0 in tensors_g0:
    cnt += ten_g0.size().numel()

  print("callbakcs.pyx -> my_bufsize - error-1", " Rank: %d" % prefix_rank)

  # just maintain one size_ptr containing full buffer size for h and c
  # cdef int total_cnt = pyApp.x0.tensor().size().numel()

  # Note size_ptr is an integer array of size 1, and we index in at location [0]
  # the int size encodes the level
  size_ptr[0] = sizeof(double)*cnt + sizeof(double) + sizeof(int)
              # vector                 time             level
  print("callbakcs.pyx -> my_bufsize - error-2", " Rank: %d" % prefix_rank)

  # size_ptr[0] += pyApp.maxParameterSize()

  print("callbakcs.pyx -> my_bufsize - end", " Rank: %d" % prefix_rank)

  return 0

cdef int my_bufpack(braid_App app, braid_Vector u, void *buffer,braid_BufferStatus status):
    # Cast u as a PyBraid_Vector
    # ten_U = (<object> u).tensor()
    # cdef np.ndarray[float,ndim=1] np_U  = ten_U.numpy().ravel() # ravel provides a flatten accessor to the array

    # # Convert void * to a double array (note dbuffer is a C-array, so no bounds checking is done) 
    # cdef int * ibuffer = <int *> buffer
    # cdef double * dbuffer = <double *>(buffer+4)

    # ibuffer[0] = (<object> u).level()
    # dbuffer[0] = (<object> u).getTime()

    # # Pack buffer
    # cdef int sz = len(np_U)
    # for k in range(sz):
    #   dbuffer[k+1] = np_U[k]
    # end for item

    ####################################################################

  cdef int prefix_rank
  cdef int prefix_size

  pyApp = <object> app

  mpi_data_ = pyApp.getMPIData()
  prefix_rank = mpi_data_.getRank()
  prefix_size = mpi_data_.getSize()

  print("callbakcs.pyx -> my_bufpack - start", " Rank: %d" % prefix_rank)

  tensors_U = (<object> u).tensors()
  ten_U_h, ten_U_c = tensors_U
  cdef np.ndarray[float,ndim=1] np_U_h  = ten_U_h.numpy().ravel()
  cdef np.ndarray[float,ndim=1] np_U_c  = ten_U_c.numpy().ravel()

  print("callbakcs.pyx -> my_bufpack - error-1", " Rank: %d" % prefix_rank)

  cdef int * ibuffer = <int *> buffer
  cdef double * dbuffer = <double *>(buffer+4)

  print("callbakcs.pyx -> my_bufpack - error-2", " Rank: %d" % prefix_rank)

  ibuffer[0] = (<object> u).level()
  dbuffer[0] = (<object> u).getTime()

  print("callbakcs.pyx -> my_bufpack - error-3", " Rank: %d" % prefix_rank)

  cdef int sz = len(np_U_h)
  for k in range(sz):
    dbuffer[k+1] = np_U_h[k]
  cdef int sz_c = len(np_U_c)
  for k in range(sz_c):
    dbuffer[sz+k+1] = np_U_c[k]

  print("callbakcs.pyx -> my_bufpack - end", " Rank: %d" % prefix_rank)

  return 0

cdef int my_bufunpack(braid_App app, void *buffer, braid_Vector *u_ptr,braid_BufferStatus status):

  cdef int prefix_rank
  cdef int prefix_size

  pyApp = <object> app

  mpi_data_ = pyApp.getMPIData()
  prefix_rank = mpi_data_.getRank()
  prefix_size = mpi_data_.getSize()

  # py_app = <object>app

  # mpi_data_ = py_app.getMPIData()
  # prefix_rank = mpi_data_.getRank()
  # prefix_size = mpi_data_.getSize()

  print("callbakcs.pyx -> my_bufunpack - start", " Rank: %d" % prefix_rank)

  cdef int * ibuffer = <int *> buffer
  cdef double * dbuffer = <double *>(buffer+4)
  cdef braid_Vector c_x = <braid_Vector> pyApp.g0
  # cdef braid_Vector c_x = <braid_Vector> pyApp.x0

  print("callbakcs.pyx -> my_bufunpack - error-1", " Rank: %d" % prefix_rank)

  my_clone(app,c_x,u_ptr)

  (<object> u_ptr[0]).level_ = ibuffer[0]
  (<object> u_ptr[0]).setTime(dbuffer[0])

  print("callbakcs.pyx -> my_bufunpack - error-2", " Rank: %d" % prefix_rank)

  # ten_U = (<object> u_ptr[0]).tensor()
  # cdef np.ndarray[float,ndim=1] np_U = ten_U.numpy().ravel() # ravel provides a flatten accessor to the array

  tensors_U = (<object> u_ptr[0]).tensors()
  ten_U_h, ten_U_c = tensors_U
  cdef np.ndarray[float,ndim=1] np_U_h  = ten_U_h.numpy().ravel()
  cdef np.ndarray[float,ndim=1] np_U_c  = ten_U_c.numpy().ravel()

  print("callbakcs.pyx -> my_bufunpack - error-3", " Rank: %d" % prefix_rank)

  # this is almost certainly slow
  # cdef int sz = len(np_U)
  # for k in range(sz):
  #   np_U[k] = dbuffer[k+1]

  cdef int sz = len(np_U_h)
  for k in range(sz):
    np_U_h[k] = dbuffer[k+1]

  cdef int sz_c = len(np_U_c)
  for k in range(sz_c):
    np_U_c[k] = dbuffer[sz+k+1]

  print("callbakcs.pyx -> my_bufunpack - error-4", " Rank: %d" % prefix_rank)

  
  temp_u = <object> u_ptr[0]
  print("temp_u type: ", type(temp_u))
  print("temp_u ", temp_u)

  print("callbakcs.pyx -> my_bufunpack - end", " Rank: %d" % prefix_rank)
  return 0

# Eric removed the below code
# cdef int my_coarsen(braid_App app, braid_Vector vec_fu, braid_Vector *cu_ptr, braid_CoarsenRefStatus status):
#   pyApp  = <object> app
#   fu =  <object> vec_fu

#   cdef int level = -1

#   cu_mem = pyApp.coarsen(fu.tensor(),fu.level())

#   cu = BraidVector(cu_mem,fu.level()+1)
#   Py_INCREF(cu) # why do we need this?

#   cu_ptr[0] = <braid_Vector> cu

#   return 0

# cdef int my_refine(braid_App app, braid_Vector cu_vec, braid_Vector *fu_ptr, braid_CoarsenRefStatus status):
#   pyApp  = <object> app
#   cu =  <object> cu_vec

#   cdef int level = -1

#   fu_mem = pyApp.refine(cu.tensor(),cu.level())
#   fu = BraidVector(fu_mem,cu.level()-1)
#   Py_INCREF(fu) # why do we need this?

#   fu_ptr[0] = <braid_Vector> fu

#   return 0
