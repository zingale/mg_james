#!/usr/bin/env python

"""

Solve the Poisson problem with isolated BCs with a collection of point masses

Superposition tells us the analytic solution

"""

import math
import numpy
import patch
import multigrid
import pylab

G = 1.0

#nx = 4
#nx = 8
#nx = 32
nx = 64
#nx = 128
#nx = 256
ny = nx
nz = nx

#-----------------------------------------------------------------------------
# define the point masses -- we specify the x,y,z and mass
# store mass objects in a list
class mass:
    def __init__ (self,x,y,z,m):
        self.x = x
        self.y = y
        self.z = z
        self.m = m

masses = []
#masses.append(mass(0.25,0.5,0.25,1.0))
#masses.append(mass(0.5 ,0.5 ,0.5 ,2.0))
#masses.append(mass(0.75,0.75,0.75,1.0))
masses.append(mass(0.25,0.25,0.25,1.0))
masses.append(mass(0.25,0.25,0.75,1.0))
masses.append(mass(0.25,0.75,0.25,1.0))
masses.append(mass(0.25,0.75,0.75,1.0))
masses.append(mass(0.75,0.25,0.25,1.0))
masses.append(mass(0.75,0.25,0.75,1.0))
masses.append(mass(0.75,0.75,0.25,1.0))
masses.append(mass(0.75,0.75,0.75,1.0))

#-----------------------------------------------------------------------------
# the analytic solution
def true(masses,x,y,z):

    phi = 0.0

    for m in masses:
        r = numpy.sqrt((x - m.x)**2 + (y - m.y)**2 + (z - m.z)**2)

        phi += green(r) * m.m

    return phi
        

#-----------------------------------------------------------------------------
# the righthand side
def f(masses, grid):
    # Find the zone that contains each mass and then compute the density accordingly

    f = grid.scratchArray()

    xc = (grid.xmax - grid.xmin) / 2.0
    yc = (grid.ymax - grid.ymin) / 2.0
    zc = (grid.zmax - grid.zmin) / 2.0

    for m in masses:
        ii = numpy.nonzero(numpy.logical_and(grid.xl <= m.x, grid.xr > m.x))[0][0]
        if (m.x <= xc): ii -= 1
        jj = numpy.nonzero(numpy.logical_and(grid.yl <= m.y, grid.yr > m.y))[0][0]
        if (m.y <= yc): jj -= 1
        kk = numpy.nonzero(numpy.logical_and(grid.zl <= m.z, grid.zr > m.z))[0][0]
        if (m.z <= zc): kk -= 1

        f[ii,jj,kk] += 4.0 * math.pi * G * (m.m / (grid.dx*grid.dy*grid.dz))

        # Update the locations on the grid for the masses, since it won't be exactly
        # where we requested, due to discretization.
   
        m.x = grid.x[ii]
        m.y = grid.y[jj]
        m.z = grid.z[kk]

    return f


#-----------------------------------------------------------------------------
# the L2 error norm
def error(myg, r):

    # L2 norm of elements in r, multiplied by dx to
    # normalize
    return numpy.sqrt(myg.dx*myg.dy*myg.dz*numpy.sum((r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1,myg.klo:myg.khi+1]**2).flat))
                

#-----------------------------------------------------------------------------
# the Green's function
def green(r):

    gf = numpy.zeros(r.shape)

    if (numpy.size(r) == 1):
        if (r != 0.0):
            gf = -G / r

    else:

        gf[numpy.where(r != 0.0)] = -G / r[numpy.where(r != 0.0)]

    return gf

#-----------------------------------------------------------------------------
# convolve the Green's functions with the masses
def convolve(mass_list, r_list):

    phi = 0.0

    for i in range(0,6):
 
        phi += numpy.sum(green(r_list[i]) * mass_list[i]) 

    return phi

#-----------------------------------------------------------------------------
# Create the distance functions that we're going to use for convolving over
def create_r(x,y,z,a):

        # we are at a fixed x
        r_xl = numpy.sqrt( (a.xmin - x)**2           + (a.y3d[a.ilo,:,:] - y)**2 + (a.z3d[a.ilo,:,:] - z)**2 )
        r_xr = numpy.sqrt( (a.xmax - x)**2           + (a.y3d[a.ihi,:,:] - y)**2 + (a.z3d[a.ihi,:,:] - z)**2 )

        # we are at a fixed y
        r_yl = numpy.sqrt( (a.x3d[:,a.jlo,:] - x)**2 + (a.ymin - y)**2           + (a.z3d[:,a.jlo,:] - z)**2 )
        r_yr = numpy.sqrt( (a.x3d[:,a.jhi,:] - x)**2 + (a.ymax - y)**2           + (a.z3d[:,a.jhi,:] - z)**2 )

        # we are at a fixed z
        r_zl = numpy.sqrt( (a.x3d[:,:,a.klo] - x)**2 + (a.y3d[:,:,a.klo] - y)**2 + (a.zmin - z)**2           )
        r_zr = numpy.sqrt( (a.x3d[:,:,a.khi] - x)**2 + (a.y3d[:,:,a.khi] - y)**2 + (a.zmax - z)**2           )

        return (r_xl, r_xr, r_yl, r_yr, r_zl, r_zr)

#-----------------------------------------------------------------------------
# the main solve -- James algorithm
#-----------------------------------------------------------------------------

# 1. solve the problem with homogeneous Dirichlet BCs

# create the multigrid object
a = multigrid.ccMG3d(nx, ny, nz,
                     xlBCtype="dirichlet", xrBCtype="dirichlet",
                     ylBCtype="dirichlet", yrBCtype="dirichlet",
                     zlBCtype="dirichlet", zrBCtype="dirichlet",
                     verbose=0)

# initialize the solution to 0
a.initZeros()

# initialize the RHS using the function f
rhs = f(masses, a.solnGrid)
a.initRHS(rhs)

# solve to a relative tolerance of 1.e-11
a.solve(rtol=1.e-11)

# get the solution -- this is the homogeneous potential
phi_h = a.getSolution()


# 2. compute surface masses

const = 1.0 / (4.0 * math.pi * G)

# sigma = grad phi_h . n / (4pi G)
# mass  = sigma . dA

# Make sure that we only sum over masses adjacent to the domain faces. Edges and corners don't count, so we leave those entries zeroed out.

dA = a.dy * a.dz

mass_xl = numpy.zeros(phi_h[a.ilo,:,:].shape)
mass_xr = numpy.zeros(phi_h[a.ihi,:,:].shape)

mass_xl[a.jlo:a.jhi+1,a.klo:a.khi+1] = -0.5 * ( (7.0 * phi_h[a.ilo,a.jlo:a.jhi+1,a.klo:a.khi+1] - phi_h[a.ilo+1,a.jlo:a.jhi+1,a.klo:a.khi+1]) / a.dx ) * const * dA
mass_xr[a.jlo:a.jhi+1,a.klo:a.khi+1] = -0.5 * ( (7.0 * phi_h[a.ihi,a.jlo:a.jhi+1,a.klo:a.khi+1] - phi_h[a.ihi-1,a.jlo:a.jhi+1,a.klo:a.khi+1]) / a.dx ) * const * dA

dA = a.dx * a.dz

mass_yl = numpy.zeros(phi_h[:,a.jlo,:].shape)
mass_yr = numpy.zeros(phi_h[:,a.jhi,:].shape)

mass_yl[a.ilo:a.ihi+1,a.klo:a.khi+1] = -0.5 * ( (7.0 * phi_h[a.ilo:a.ihi+1,a.jlo,a.klo:a.khi+1] - phi_h[a.ilo:a.ihi+1,a.jlo+1,a.klo:a.khi+1]) / a.dy ) * const * dA
mass_yr[a.ilo:a.ihi+1,a.klo:a.khi+1] = -0.5 * ( (7.0 * phi_h[a.ilo:a.ihi+1,a.jhi,a.klo:a.khi+1] - phi_h[a.ilo:a.ihi+1,a.jhi-1,a.klo:a.khi+1]) / a.dy ) * const * dA

dA = a.dx * a.dy

mass_zl = numpy.zeros(phi_h[:,:,a.klo].shape)
mass_zr = numpy.zeros(phi_h[:,:,a.khi].shape)

mass_zl[a.ilo:a.ihi+1,a.jlo:a.jhi+1] = -0.5 * ( (7.0 * phi_h[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo] - phi_h[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo+1]) / a.dz ) * const * dA
mass_zr[a.ilo:a.ihi+1,a.jlo:a.jhi+1] = -0.5 * ( (7.0 * phi_h[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.khi] - phi_h[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.khi-1]) / a.dz ) * const * dA

mass_list = (mass_xl, mass_xr, mass_yl, mass_yr, mass_zl, mass_zr)

totalMass = numpy.sum(mass_list)
actualMass = 0.0
for m in masses:
    actualMass += m.m

print "Total surface mass = ", totalMass
print "Total mass in the domain =", actualMass
print "Relative error in mass = ", abs(totalMass - actualMass) / actualMass

# 3. compute the boundary conditions on phi

# -x face
Phi_xl = numpy.zeros(mass_xl.shape)
true_xl = numpy.zeros(mass_xl.shape)

for kk in range(a.klo, a.khi+1):
    for jj in range(a.jlo, a.jhi+1):

        # we are in the x = xmin plane

        x = a.xmin
        y = a.y[jj]
        z = a.z[kk]

        r_list = create_r(x,y,z,a)

        Phi_xl[jj,kk] = convolve(mass_list, r_list)
        true_xl[jj,kk] = true(masses,x,y,z)

print numpy.min(Phi_xl[a.jlo:a.jhi+1,a.klo:a.khi+1]), numpy.max(Phi_xl[a.jlo:a.jhi+1,a.klo:a.khi+1])

# +x face
Phi_xr = numpy.zeros(mass_xr.shape)
true_xr = numpy.zeros(mass_xr.shape)

for kk in range(a.klo, a.khi+1):
    for jj in range(a.jlo, a.jhi+1):

        # we are in the x = xmax plane

        x = a.xmax
        y = a.y[jj]
        z = a.z[kk]

        r_list = create_r(x,y,z,a)

        Phi_xr[jj,kk] = convolve(mass_list, r_list)
        true_xr[jj,kk] = true(masses,x,y,z)

print numpy.min(Phi_xr[a.jlo:a.jhi+1,a.klo:a.khi+1]), numpy.max(Phi_xr[a.jlo:a.jhi+1,a.klo:a.khi+1])

# -y face
Phi_yl = numpy.zeros(mass_yl.shape)
true_yl = numpy.zeros(mass_yl.shape)

for kk in range(a.klo, a.khi+1):
    for ii in range(a.ilo, a.ihi+1):

        # we are in the y = ymin plane

        x = a.x[ii]
        y = a.ymin
        z = a.z[kk]

        r_list = create_r(x,y,z,a)

        Phi_yl[ii,kk] = convolve(mass_list, r_list)
        true_yl[ii,kk] = true(masses,x,y,z)

print numpy.min(Phi_yl[a.ilo:a.ihi+1,a.klo:a.khi+1]), numpy.max(Phi_yl[a.ilo:a.ihi+1,a.klo:a.khi+1])

# +y face
Phi_yr = numpy.zeros(mass_yr.shape)
true_yr = numpy.zeros(mass_yr.shape)

for kk in range(a.klo, a.khi+1):
    for ii in range(a.ilo, a.ihi+1):

        # we are in the y = ymax plane

        x = a.x[ii]
        y = a.ymax
        z = a.z[kk]

        r_list = create_r(x,y,z,a)

        Phi_yr[ii,kk] = convolve(mass_list, r_list)
        true_yr[ii,kk] = true(masses,x,y,z)

print numpy.min(Phi_yr[a.ilo:a.ihi+1,a.klo:a.khi+1]), numpy.max(Phi_yr[a.ilo:a.ihi+1,a.klo:a.khi+1])

# -z face
Phi_zl = numpy.zeros(mass_zl.shape)
true_zl = numpy.zeros(mass_zl.shape)

for jj in range(a.jlo, a.jhi+1):
    for ii in range(a.ilo, a.ihi+1):

        # we are in the z = zmin plane

        x = a.x[ii]
        y = a.y[jj]
        z = a.zmin

        r_list = create_r(x,y,z,a)

        Phi_zl[ii,jj] = convolve(mass_list, r_list)
        true_zl[ii,jj] = true(masses,x,y,z)

print numpy.min(Phi_zl[a.ilo:a.ihi+1,a.jlo:a.jhi+1]), numpy.max(Phi_zl[a.ilo:a.ihi+1,a.ilo:a.ihi+1])

# +z face
Phi_zr = numpy.zeros(mass_zr.shape)
true_zr = numpy.zeros(mass_zr.shape)

for jj in range(a.jlo, a.jhi+1):
    for ii in range(a.ilo, a.ihi+1):

        # we are in the z = zmax plane

        x = a.x[ii]
        y = a.y[jj]
        z = a.zmax

        r_list = create_r(x,y,z,a)

        Phi_zr[ii,jj] = convolve(mass_list, r_list)
        true_zr[ii,jj] = true(masses,x,y,z)
                
print numpy.min(Phi_zr[a.ilo:a.ihi+1,a.ilo:a.ihi+1]), numpy.max(Phi_zr[a.ilo:a.ihi+1,a.ilo:a.ihi+1])


# 4. solve for the isolated potential

# we do inhomogeneous BCs for this, and solve Laplace's equation.  The
# BCs are simply the Phi's on the surfaces constructed above.
b = multigrid.ccMG3d(nx, ny, nz,
                     xlBCtype="dirichlet", xrBCtype="dirichlet",
                     ylBCtype="dirichlet", yrBCtype="dirichlet",
                     zlBCtype="dirichlet", zrBCtype="dirichlet",
                     xlBC=Phi_xl, xrBC=Phi_xr,
                     ylBC=Phi_yl, yrBC=Phi_yr,
                     zlBC=Phi_zl, zrBC=Phi_zr,
                     verbose=0)

# initialize the solution to 0
b.initZeros()

# initialize the RHS
b.initRHS(rhs)

# solve to a relative tolerance of 1.e-11
b.solve(rtol=1.e-11)

# get the solution -- this is the isolated potential
phi = b.getSolution()

#-----------------------------------------------------------------------------
# compute the error from the analytic solution
c = multigrid.ccMG3d(nx, ny, nz,
                     xlBCtype="dirichlet", xrBCtype="dirichlet",
                     ylBCtype="dirichlet", yrBCtype="dirichlet",
                     zlBCtype="dirichlet", zrBCtype="dirichlet",
                     xlBC=true_xl, xrBC=true_xr,
                     ylBC=true_yl, yrBC=true_yr,
                     zlBC=true_zl, zrBC=true_zr,
                     verbose=0)

c.initZeros()
c.initRHS(rhs)
c.solve(rtol=1.e-11)
t = c.getSolution()

e = phi - t

print " L2 error from true solution = %g\n rel. err from previous cycle = %g\n num. cycles = %d" % \
      (error(a.solnGrid, e) / error(a.solnGrid,t), a.relativeError, a.numCycles)

print "Min (phi) = ", numpy.min(phi[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo:a.khi+1]), "Max (phi) = ", numpy.max(phi[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo:a.khi+1])
print "Min (true) = ", numpy.min(t[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo:a.khi+1]), "Max (true) = ", numpy.max(t[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo:a.khi+1])

#-----------------------------------------------------------------------------
# visualize -- slices through the center
v = phi
pylab.subplot(231)
pylab.imshow(v[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo+a.nz/2])

pylab.subplot(232)
pylab.imshow(v[a.ilo:a.ihi+1,a.jlo+a.ny/2,a.klo:a.khi+1])

pylab.subplot(233)
pylab.imshow(v[a.ilo+a.nx/2,a.jlo:a.jhi+1,a.klo:a.khi+1])

pylab.subplot(234)
pylab.imshow(t[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo+a.nz/2])

pylab.subplot(235)
pylab.imshow(t[a.ilo:a.ihi+1,a.jlo+a.ny/2,a.klo:a.khi+1])

pylab.subplot(236)
pylab.imshow(t[a.ilo+a.nx/2,a.jlo:a.jhi+1,a.klo:a.khi+1])

pylab.show()
