import torch
import numpy as np
import math
cimport numpy as np

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF

from mpi4py import MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi

##
# Define your Python Braid Vector as a C-struct

cdef int my_access(braid_App app,braid_Vector u,braid_AccessStatus status):

  pyApp = <object> app
  cdef double t
  # Create Numpy wrapper around u.v
  ten_u = <object> u

  braid_AccessStatusGetT(status, &t)
  pyApp.access(t,ten_u)

  return 0

cdef int my_step(braid_App app, braid_Vector ustop, braid_Vector fstop, braid_Vector vec_u, braid_StepStatus status):
  
  pyApp = <object> app
  u =  <object> vec_u
  cdef double tstart
  cdef double tstop
  tstart = 0.0
  tstop = 5.0
  braid_StepStatusGetTstartTstop(status, &tstart, &tstop)
  temp = pyApp.eval(u,tstart,tstop)
  u_h,u_c = u.tensors()
  temp_h,temp_c = temp.tensors()
  u_h.copy_(temp_h)
  u_h.copy_(temp_c)

  return 0
  
cdef int my_init(braid_App app, double t, braid_Vector *u_ptr):

  pyApp = <object> app
  u_mem = pyApp.buildInit(t)
  Py_INCREF(u_mem) # why do we need this?
  u_ptr[0] = <braid_Vector> u_mem

  return 0

cdef int my_free(braid_App app, braid_Vector u):

  pyU = <object> u
  # Decrement the smart pointer
  Py_DECREF(pyU)
  del pyU
  return 0

cdef int my_sum(braid_App app, double alpha, braid_Vector x, double beta, braid_Vector y):

  pyApp = <object> app
  tensors_X = (<object> x).tensors()
  tensors_Y = (<object> y).tensors()

  ten_X_h, ten_X_c = tensors_X
  ten_Y_h, ten_Y_c = tensors_Y

  cdef np.ndarray[float,ndim=1] np_X_h = ten_X_h.numpy().ravel()
  cdef np.ndarray[float,ndim=1] np_X_c = ten_X_c.numpy().ravel()
  cdef np.ndarray[float,ndim=1] np_Y_h = ten_Y_h.numpy().ravel()
  cdef np.ndarray[float,ndim=1] np_Y_c = ten_Y_c.numpy().ravel()

  cdef int sz = len(np_X_h)
  for k in range(sz):
    np_Y_h[k] = alpha*np_X_h[k]+beta*np_Y_h[k]
    np_Y_c[k] = alpha*np_X_c[k]+beta*np_Y_c[k]

  return 0

cdef int my_clone(braid_App app, braid_Vector u, braid_Vector *v_ptr):

  pyApp = <object> app
  ten_U = <object> u
  v_mem = ten_U.clone()
  Py_INCREF(v_mem) # why do we need this?
  v_ptr[0] = <braid_Vector> v_mem

  return 0

cdef int my_norm(braid_App app, braid_Vector u, double *norm_ptr):

  pyApp = <object> app
  # Compute norm 
  tensors_U = (<object> u).tensors()
  norm_ptr[0] = 0.0
  for ten_U in tensors_U:
    norm_ptr[0] += torch.norm(ten_U)**2

  math.sqrt(norm_ptr[0])

  return 0

cdef int my_bufsize(braid_App app, int *size_ptr, braid_BufferStatus status):

  pyApp = <object> app
  cdef int cnt = 0
  tensors_g0 = (<object> pyApp.g0).tensors()
  for ten_g0 in tensors_g0:
    cnt += ten_g0.size().numel()

  # Note size_ptr is an integer array of size 1, and we index in at location [0]
  # the int size encodes the level
  size_ptr[0] = sizeof(double)*cnt + sizeof(double) + sizeof(int)
              # vector                 time             level
  size_ptr[0] += pyApp.maxParameterSize()

  return 0

cdef int my_bufpack(braid_App app, braid_Vector u, void *buffer,braid_BufferStatus status):

  pyApp = <object> app
  tensors_U = (<object> u).tensors()
  ten_U_h, ten_U_c = tensors_U
  cdef np.ndarray[float,ndim=1] np_U_h  = ten_U_h.numpy().ravel()
  cdef np.ndarray[float,ndim=1] np_U_c  = ten_U_c.numpy().ravel()

  cdef int * ibuffer = <int *> buffer
  cdef double * dbuffer = <double *>(buffer+4)

  ibuffer[0] = (<object> u).level()
  dbuffer[0] = (<object> u).getTime()

  cdef int sz = len(np_U_h)
  for k in range(sz):
    dbuffer[k+1] = np_U_h[k]
  cdef int sz_c = len(np_U_c)
  for k in range(sz_c):
    dbuffer[sz+k+1] = np_U_c[k]

  return 0

cdef int my_bufunpack(braid_App app, void *buffer, braid_Vector *u_ptr,braid_BufferStatus status):

  pyApp = <object> app
  cdef int * ibuffer = <int *> buffer
  cdef double * dbuffer = <double *>(buffer+4)
  cdef braid_Vector c_x = <braid_Vector> pyApp.g0

  my_clone(app,c_x,u_ptr)

  (<object> u_ptr[0]).level_ = ibuffer[0]
  (<object> u_ptr[0]).setTime(dbuffer[0])

  tensors_U = (<object> u_ptr[0]).tensors()
  ten_U_h, ten_U_c = tensors_U
  cdef np.ndarray[float,ndim=1] np_U_h  = ten_U_h.numpy().ravel()
  cdef np.ndarray[float,ndim=1] np_U_c  = ten_U_c.numpy().ravel()

  cdef int sz = len(np_U_h)
  for k in range(sz):
    np_U_h[k] = dbuffer[k+1]

  cdef int sz_c = len(np_U_c)
  for k in range(sz_c):
    np_U_c[k] = dbuffer[sz+k+1]

  temp_u = <object> u_ptr[0]

  return 0

# The updated code removed the code below
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
