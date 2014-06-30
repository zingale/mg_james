#!/usr/bin/env python

"""

an example of using the multigrid class to solve Laplace's equation.  Here, we
solve

u_xx + u_yy + u_zz = 2[(1-6x**2)y**2(1-y**2)z**2(1-z**2) + 
                       (1-6y**2)x**2(1-x**2)z**2(1-z**2) +
                       (1-6z**2)x**2(1-x**2)y**2(1-y**2)]
u = 0 on the boundary

this is a 3-d extension of the example from page 64 of the book `A
Multigrid Tutorial, 2nd Ed.'

The analytic solution is u(x,y) = (x**2 - x**4)(y**2 - y**4)(z**2 - z**4)

"""

import numpy
import patch
import multigrid
import pylab

# the analytic solution
def true(x,y,z):
    return (x**2 - x**4)*(y**2 - y**4)*(z**2 - z**4)


# the L2 error norm
def error(myg, r):

    # L2 norm of elements in r, multiplied by dx to
    # normalize
    return numpy.sqrt(myg.dx*myg.dy*myg.dz*numpy.sum((r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1,myg.klo:myg.khi+1]**2).flat))


# the righthand side
def f(x,y,z):
    return 2.0*((1.0-6.0*x**2) * y**2*(1.0-y**2) * z**2*(1.0-z**2) + 
                (1.0-6.0*y**2) * x**2*(1.0-x**2) * z**2*(1.0-z**2) + 
                (1.0-6.0*z**2) * x**2*(1.0-x**2) * y**2*(1.0-y**2))

                
# test the multigrid solver
nx = 64
ny = 64
nz = 64


# create the multigrid object
a = multigrid.ccMG3d(nx, ny, nz,
                     xlBCtype="dirichlet", xrBCtype="dirichlet",
                     ylBCtype="dirichlet", yrBCtype="dirichlet",
                     zlBCtype="dirichlet", zrBCtype="dirichlet",
                     verbose=1)

# initialize the solution to 0
init = a.solnGrid.scratchArray()

a.initSolution(init)

# initialize the RHS using the function f
rhs = f(a.x3d, a.y3d, a.z3d)
a.initRHS(rhs)

# solve to a relative tolerance of 1.e-11
a.solve(rtol=1.e-11)

# alternately, we can just use smoothing by uncommenting the following
#a.smooth(a.nlevels-1,50000)

# get the solution 
v = a.getSolution()

# compute the error from the analytic solution
b = true(a.x3d,a.y3d,a.z3d)
e = v - b

print " L2 error from true solution = %g\n rel. err from previous cycle = %g\n num. cycles = %d" % \
      (error(a.solnGrid, e), a.relativeError, a.numCycles)


