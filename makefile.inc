MPIRUN=mpirun
PYTHON=python

# For my Linux
#####################################
#XBRAID_ROOT=/Users/xxxx/Desktop/Layer-parallel/xbraid-cython-adjoint/braid
#POSTFIX=cpython-36m-x86_64-linux-gnu.so  
#CC=mpicc
#LDSHARED=mpicc -shared
#EXTRA_FLAGS=-O3 -fPIC

# for my MAC
#####################################
POSTFIX=cpython-36m-darwin.so
XBRAID_ROOT=/Users/xxxx/Desktop/Layer-parallel/xbraid-cython-adjoint/braid
CC=mpicc
LDSHARED=
EXTRA_FLAGS=

#export TORCHBRAID_DIR=/Users/xxxx/Desktop/Layer-parallel/torchbraid-github/
#export PYTHONPATH=${TORCHBRAID_DIR}/torchbraid:${TORCHBRAID_DIR}:${PYTHONPATH}