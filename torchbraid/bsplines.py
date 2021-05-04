#!/usr/bin/env python
import matplotlib.pyplot as plt
from numpy import *


# Evaluate the splines at time t:
# This returns the values of d+1 spline basis functions, and the interval k s.t. t \in [tau_k, \tau_k+1] for spline knots \tau
def evalBsplines(degree, deltaKnots, time):

    # Get interval index l s.t. t \in [t_l, t_l+1]
    k = int(time / deltaKnots)   # this will round down to next smaller integer

    # Start with coefficient vector to unit vector 1, 0, 0, 0...
    spline = []
    spline.append(1.0)

    # Recursive loop to update splines
    for i in range(1,degree+1):        # i = 1,2,...,degree
        spline.append(0.0)
        for r in range(i,0,-1):        # r = i, i-1, ..., 1
            coeff1 = (time - (k-i+r)*deltaKnots)  / ( (k+r)*deltaKnots - (k-i+r)*deltaKnots )
            coeff2 = ( (k+r+1)*deltaKnots - time) / ( (k+r+1)*deltaKnots - (k-i+r+1)*deltaKnots )
            spline[r] = coeff1 * spline[r-1] + coeff2 * spline[r]
        spline[0] = spline[0] * ((k+1)*deltaKnots - time) / ((k+1)*deltaKnots - (k-i+1)*deltaKnots)
    
    return spline, k


def spline_test(degree, nSplines, Tfinal, deltax):
    print("Testing ", nSplines, " Bsplines of degree ", degree)

    # Init grid and splines
    n = int(Tfinal / deltax)
    xgrid = linspace(0.0, Tfinal, n+1)

    # Initialize splines
    nKnots = nSplines - degree + 1
    deltaKnots = Tfinal / (nKnots-1)
    spline = zeros((nSplines+1)*(n+1)).reshape(nSplines+1, n+1)

    # Loop over time domain and compute spline coefficients
    for i in range(len(xgrid)):

        time = xgrid[i]
        l = int(time / deltaKnots)
        spline[l:l+degree+1,i], k = evalBsplines(degree, deltaKnots, time)
        # print(spline[l:l+degree+1,i])

    # Plot
    for i in range(nSplines):
        plt.plot(xgrid, spline[i,:], label="B_"+str(i))
        # print(i, " ", spline[i,:])
    plt.legend()
    plt.show()


# # Test now:
# degree = 2
# nSplines = 10
# Tfinal = 1.0
# deltax = 0.01
# spline_test(degree, nSplines, Tfinal, deltax)
