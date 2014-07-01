#!/usr/bin/env python

"""

Test inhomogeneous Dirichlet boundary conditions

u_xx + u_yy + u_zz = f

where u = (cos(pi*x) + cos(pi*y) + cos(pi*z))*(1-x)*(1-y)*(1-z)

is the exact solution.  This has boundary conditions of u = 0 on the
upper boundaries, but a functional form on the lower boundaries.

The RHS is:

f = pi*(x - 1)*(y - 1)*(pi*(z - 1)*cos(pi*z) + 
    2*sin(pi*z)) + pi*(x - 1)*(z - 1)*(pi*(y - 1)*cos(pi*y) + 
    2*sin(pi*y)) + pi*(y - 1)*(z - 1)*(pi*(x - 1)*cos(pi*x) + 2*sin(pi*x))

"""

import numpy
import patch
import multigrid
import pylab

pi = numpy.pi
cos = numpy.cos
sin = numpy.sin

# the analytic solution
def true(x,y,z):
    return (cos(pi*x) + cos(pi*y) + cos(pi*z))*(1-x)*(1-y)*(1-z)     


# the L2 error norm
def error(myg, r):

    # L2 norm of elements in r, multiplied by dx to
    # normalize
    return numpy.sqrt(myg.dx*myg.dy*myg.dz*numpy.sum((r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1,myg.klo:myg.khi+1]**2).flat))


# the righthand side
def f(x,y,z):
    return pi*(x - 1)*(y - 1)*(pi*(z - 1)*cos(pi*z) + \
      2*sin(pi*z)) + pi*(x - 1)*(z - 1)*(pi*(y - 1)*cos(pi*y) + \
      2*sin(pi*y)) + pi*(y - 1)*(z - 1)*(pi*(x - 1)*cos(pi*x) + 2*sin(pi*x))


# the boundary condtions
def x_low_BC(y, z):
    return (-y + 1)*(-z + 1)*(cos(pi*y) + cos(pi*z) + 1)

def x_hi_BC(y, z):
    return numpy.zeros_like(y)

def y_low_BC(x, z):
    return (-x + 1)*(-z + 1)*(cos(pi*x) + cos(pi*z) + 1)

def y_hi_BC(x, z):
    return numpy.zeros_like(x)

def z_low_BC(x, y):
    return (-x + 1)*(-y + 1)*(cos(pi*x) + cos(pi*y) + 1)

def z_hi_BC(x, y):
    return numpy.zeros_like(x)
                

# test the multigrid solver
nx = 64
ny = nz = nx

# a dummy grid to make initialization easy -- this is a hack 
g = patch.grid3d(nx, ny, nz, ng=1)

# create the multigrid object
a = multigrid.ccMG3d(nx, ny, nz, 
                     xlBCtype="dirichlet", xrBCtype="dirichlet",
                     ylBCtype="dirichlet", yrBCtype="dirichlet",
                     zlBCtype="dirichlet", zrBCtype="dirichlet",
                     xlBC=x_low_BC(g.y3d[g.ilo,:,:], g.z3d[g.ilo,:,:]),
                     xrBC=x_hi_BC( g.y3d[g.ihi,:,:], g.z3d[g.ihi,:,:]),
                     ylBC=y_low_BC(g.x3d[:,g.jlo,:], g.z3d[:,g.jlo,:]),
                     yrBC=y_hi_BC( g.x3d[:,g.jhi,:], g.z3d[:,g.jhi,:]),
                     zlBC=z_low_BC(g.x3d[:,:,g.klo], g.y3d[:,:,g.klo]),
                     zrBC=z_hi_BC( g.x3d[:,:,g.khi], g.y3d[:,:,g.khi]),
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


