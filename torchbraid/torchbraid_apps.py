# cython: profile=True
# cython: linetrace=True

import torch
from torchbraid_app import BraidApp
from torchbraid_app import BraidVector

import sys

from mpi4py import MPI

class ForwardBraidApp(BraidApp):

  def __init__(self,comm,layer_models,local_num_steps,Tf,max_levels,max_iters,timer_manager):
    BraidApp.__init__(self,comm,local_num_steps,Tf,max_levels,max_iters)

    # note that a simple equals would result in a shallow copy...bad!
    self.layer_models = [l for l in layer_models]

    comm          = self.getMPIData().getComm()
    my_rank       = self.getMPIData().getRank()
    num_ranks     = self.getMPIData().getSize()

    # send everything to the left (this helps with the adjoint method)
    if my_rank>0:
      comm.send(self.layer_models[0],dest=my_rank-1,tag=22)
    if my_rank<num_ranks-1:
      neighbor_model = comm.recv(source=my_rank+1,tag=22)
      self.layer_models.append(neighbor_model)

    # build up the core
    self.py_core = self.initCore()

    self.timer_manager = timer_manager
    self.use_deriv = False
  # end __init__

  def run(self,x):
    self.soln_store = dict()

    # run the braid solver
    with self.timer("runBraid"):
      # turn on derivative path (as requried)
      self.use_deriv = x.requires_grad

      y = self.runBraid(x)

      # reset derivative papth
      self.use_deriv = False

    return y
  # end forward

  def timer(self,name):
    return self.timer_manager.timer("ForWD::"+name)

  def getLayer(self,t,tf,level):
    index = self.getLocalTimeStepIndex(t,tf,level)
    return self.layer_models[index]

  def parameters(self):
    return [list(l.parameters()) for l in self.layer_models]

  def eval(self,x,tstart,tstop,level,force_deriv=False):
    """
    Method called by "my_step" in braid. This is
    required to propagate from tstart to tstop, with the initial
    condition x. The level is defined by braid
    """

    # there are two paths by which eval is called:
    #  1. x is a BraidVector: my step has called this method
    #  2. x is a torch tensor: called internally (probably at the behest
    #                          of the adjoint)

    with self.timer("eval(level=%d)" % level):
      # this determines if a derivative computation is required
      require_derivatives = force_deriv or self.use_deriv
  
      # determine if braid or tensor version is called
      t_x = x
      use_braidvec = False
      if isinstance(x,BraidVector):
        t_x = x.tensor()
        use_braidvec = True

      # get some information about what to do
      dt = tstop-tstart
      layer = self.getLayer(tstart,tstop,level)

      if require_derivatives:
        # slow path requires derivatives

        if use_braidvec:
          t_x = x.tensor().clone()

        with torch.enable_grad():
          if level==0:
            t_x.requires_grad = True 
    
          t_y = t_x+dt*layer(t_x)
    
        # store off the solution for later adjoints
        if level==0 and use_braidvec:
          ts_index = self.getGlobalTimeStepIndex(tstart,tstop,0)
          self.soln_store[ts_index] = (t_y,t_x)
      else:
        # fast pure forwrard mode
        with torch.no_grad():
          t_y = t_x+dt*layer(t_x)
      # end if require_derivatives 
    
      # return a braid or tensor depending on what came in
      if use_braidvec:
        return BraidVector(t_y,level) 
      else:
        return t_y 
  # end eval

  def getPrimalWithGrad(self,tstart,tstop,level):
    """ 
    Get the forward solution associated with this
    time step and also get its derivative. This is
    used by the BackkwardApp in computation of the
    adjoint (backprop) state and parameter derivatives.
    Its intent is to abstract the forward solution
    so it can be stored internally instead of
    being recomputed.
    """
    
    ts_index = self.getGlobalTimeStepIndex(tstart,tstop,level)
    layer = self.getLayer(tstart,tstop,level)

    # the idea here is store it internally, failing
    # that the values need to be recomputed locally. This may be
    # because you are at a processor boundary, or decided not
    # to start the value 
    if ts_index in self.soln_store:
      return self.soln_store[ts_index],layer

    # value wasn't found, recompute it and return.
    x_old = self.soln_store[ts_index-1][0].clone()
    return (self.eval(x_old,tstart,tstop,0,force_deriv=True),x_old), layer

  # end getPrimalWithGrad

# end ForwardBraidApp

##############################################################

class BackwardBraidApp(BraidApp):

  def __init__(self,fwd_app,timer_manager):
    # call parent constructor
    BraidApp.__init__(self,fwd_app.getMPIData().getComm(),
                           fwd_app.local_num_steps,
                           fwd_app.Tf,
                           fwd_app.max_levels,
                           fwd_app.max_iters)

    self.fwd_app = fwd_app

    # build up the core
    self.py_core = self.initCore()

    # setup adjoint specific stuff
    self.fwd_app.setStorage(0)

    # reverse ordering for adjoint/backprop
    self.setRevertedRanks(1)

    self.timer_manager = timer_manager
  # end __init__

  def timer(self,name):
    return self.timer_manager.timer("BckWD::"+name)

  def run(self,x):

    with self.timer("runBraid"):
      f = self.runBraid(x)

    with self.timer("run::extra"):
      my_params = self.fwd_app.parameters()

      # this code is due to how braid decomposes the backwards problem
      # The ownership of the time steps is shifted to the left (and no longer balanced)
      first = 1
      if self.getMPIData().getRank()==0:
        first = 0

      # preserve the layerwise structure, to ease communication
      self.grads = [ [item.grad.clone() for item in sublist] for sublist in my_params[first:]]
      for m in self.fwd_app.layer_models:
         m.zero_grad()

    return f
  # end forward

  def eval(self,x,tstart,tstop,level):
    with self.timer("eval(level=%d)" % level):
      # we need to adjust the time step values to reverse with the adjoint
      # this is so that the renumbering used by the backward problem is properly adjusted
      (t_py,t_px),layer = self.fwd_app.getPrimalWithGrad(self.Tf-tstop,self.Tf-tstart,level)
  
      # t_px should have no gradient
      if not t_px.grad is None:
        t_px.grad.data.zero_()
  
      # play with the layers gradient to make sure they are on apprpriately
      for p in layer.parameters(): 
        if level==0:
          if not p.grad is None:
            p.grad.data.zero_()
        else:
          # if you are not on the fine level, compute n gradients
          for p in layer.parameters():
            p.requires_grad = False
  
      # perform adjoint computation
      t_x = x.tensor()
      t_py.backward(t_x,retain_graph=True)
  
      for p in layer.parameters():
        p.requires_grad = True

    return BraidVector(t_px.grad,level) 
  # end eval
# end BackwardBraidApp
