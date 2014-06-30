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

nx = 64
ny = 64
nz = 64


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
masses.append(mass(0.5 ,0.5 ,0.5 ,2.0))
#masses.append(mass(0.75,0.75,0.75,1.0))


#-----------------------------------------------------------------------------
# the analytic solution
def true(masses,x,y,z):

    phi = None

    for m in masses:
        r = numpy.sqrt((x - m.x)**2 + (y - m.y)**2 + (z - m.z)**2)
        if phi == None:
            phi = -G*m.m/r
        else:
            phi += -G*m.m/r

    return phi
        

#-----------------------------------------------------------------------------
# the righthand side
def f(masses, grid):
    # find the zone that contains each mass and then compute the density accordingly

    f = grid.scratchArray()

    for m in masses:
        ii = numpy.nonzero(numpy.logical_and(grid.xl <= m.x, grid.xr > m.x))[0][0]
        jj = numpy.nonzero(numpy.logical_and(grid.yl <= m.y, grid.yr > m.y))[0][0]
        kk = numpy.nonzero(numpy.logical_and(grid.zl <= m.z, grid.zr > m.z))[0][0]

        f[ii,jj,kk] += m.m

    f /= (grid.dx*grid.dy*grid.dz)
    f = 4.0*math.pi*f

    return f


#-----------------------------------------------------------------------------
# the L2 error norm
def error(myg, r):

    # L2 norm of elements in r, multiplied by dx to
    # normalize
    return numpy.sqrt(myg.dx*myg.dy*myg.dz*numpy.sum((r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1,myg.klo:myg.khi+1]**2).flat))


                


#-----------------------------------------------------------------------------
# the main solve -- James algorithm
#-----------------------------------------------------------------------------

# 1. solve the problem with homogeneous Dirichlet BCs

# create the multigrid object
a = multigrid.ccMG3d(nx, ny, nz,
                     xlBCtype="dirichlet", xrBCtype="dirichlet",
                     ylBCtype="dirichlet", yrBCtype="dirichlet",
                     zlBCtype="dirichlet", zrBCtype="dirichlet",
                     verbose=1)

# initialize the solution to 0
a.initZeros()

# initialize the RHS using the function f
a.initRHS(f(masses, a.solnGrid))

# solve to a relative tolerance of 1.e-11
a.solve(rtol=1.e-11)

# get the solution -- this is the homogeneous potential
phi_h = a.getSolution()


# 2. compute sigma -- the surface charge

# sigma = grad phi_h . n / (4pi G)
sigma_xl = ( (phi_h[a.ilo,:,:] - 0.0)/a.dx) / (4.0*math.pi*G)
sigma_xr = ( (0.0 - phi_h[a.ihi,:,:])/a.dx) / (4.0*math.pi*G)

sigma_yl = ( (phi_h[:,a.jlo,:] - 0.0)/a.dy) / (4.0*math.pi*G)
sigma_yr = ( (0.0 - phi_h[:,a.jhi,:])/a.dy) / (4.0*math.pi*G)

sigma_zl = ( (phi_h[:,:,a.klo] - 0.0)/a.dz) / (4.0*math.pi*G)
sigma_zr = ( (0.0 - phi_h[:,:,a.khi])/a.dz) / (4.0*math.pi*G)


# 3. compute the boundary conditions on "opposite-charge" potential, Phi

A = 4.0*math.pi*G
A = 1.0

# -x face
Phi_xl = numpy.zeros(sigma_xl.shape)

for kk in range(a.klo, a.khi+1):
    for jj in range(a.jlo, a.jhi+1):

        # we are in the x = xmin plane

        # sigma coords - Phi coords

        # sum over sigma_xl and sigma_xr -- we are at a fixed x
        r_xl = numpy.sqrt(                        (a.y3d[a.ilo,:,:] - a.y[jj])**2 + (a.z3d[a.ilo,:,:] - a.z[kk])**2 )
        r_xr = numpy.sqrt( (a.xmax - a.xmin)**2 + (a.y3d[a.ihi,:,:] - a.y[jj])**2 + (a.z3d[a.ihi,:,:] - a.z[kk])**2 )

        Phi_xl[jj,kk] += -numpy.sum(numpy.where( r_xl == 0.0, 0.0, sigma_xl/(A*r_xl) ))
        Phi_xl[jj,kk] += -numpy.sum(sigma_xr/(A*r_xr))

        # sum over sigma_yl and sigma_yr -- we are at a fixed y
        r_yl = numpy.sqrt( (a.x3d[:,a.jlo,:] - a.xmin)**2 + (a.ymin - a.y[jj])**2 + (a.z3d[:,a.jlo,:] - a.z[kk])**2 )
        r_yr = numpy.sqrt( (a.x3d[:,a.jhi,:] - a.xmin)**2 + (a.ymax - a.y[jj])**2 + (a.z3d[:,a.jhi,:] - a.z[kk])**2 )

        Phi_xl[jj,kk] += -numpy.sum(sigma_yl/(A*r_yl))
        Phi_xl[jj,kk] += -numpy.sum(sigma_yr/(A*r_yr))

        # sum over sigma_zl and sigma_zr -- we are at a fixed z
        r_zl = numpy.sqrt( (a.x3d[:,:,a.klo] - a.xmin)**2 + (a.y3d[:,:,a.klo] - a.y[jj])**2 + (a.zmin - a.z[kk])**2 )
        r_zr = numpy.sqrt( (a.x3d[:,:,a.khi] - a.xmin)**2 + (a.y3d[:,:,a.khi] - a.y[jj])**2 + (a.zmin - a.z[kk])**2 )

        Phi_xl[jj,kk] += -numpy.sum(sigma_zl/(A*r_zl))
        Phi_xl[jj,kk] += -numpy.sum(sigma_zr/(A*r_zr))

Phi_xl *= a.dy*a.dz

print numpy.min(Phi_xl[a.jlo:a.jhi+1,a.klo:a.khi+1]), numpy.max(Phi_xl[a.jlo:a.jhi+1,a.klo:a.khi+1])

# +x face
Phi_xr = numpy.zeros(sigma_xr.shape)

for kk in range(a.klo, a.khi+1):
    for jj in range(a.jlo, a.jhi+1):

        # we are in the x = xmax plane

        # sigma coords - Phi coords

        # sum over sigma_xl and sigma_xr -- we are at a fixed x
        r_xl = numpy.sqrt( (a.xmin - a.xmax)**2 + (a.y3d[a.ilo,:,:] - a.y[jj])**2 + (a.z3d[a.ilo,:,:] - a.z[kk])**2 )
        r_xr = numpy.sqrt(                        (a.y3d[a.ihi,:,:] - a.y[jj])**2 + (a.z3d[a.ihi,:,:] - a.z[kk])**2 )

        Phi_xr[jj,kk] += -numpy.sum(sigma_xl/(A*r_xl))
        Phi_xr[jj,kk] += -numpy.sum(numpy.where( r_xr == 0.0, 0.0, sigma_xr/(A*r_xr) ))

        # sum over sigma_yl and sigma_yr -- we are at a fixed y
        r_yl = numpy.sqrt( (a.x3d[:,a.jlo,:] - a.xmax)**2 + (a.ymin - a.y[jj])**2 + (a.z3d[:,a.jlo,:] - a.z[kk])**2 )
        r_yr = numpy.sqrt( (a.x3d[:,a.jhi,:] - a.xmax)**2 + (a.ymax - a.y[jj])**2 + (a.z3d[:,a.jhi,:] - a.z[kk])**2 )

        Phi_xr[jj,kk] += -numpy.sum(sigma_yl/(A*r_yl))
        Phi_xr[jj,kk] += -numpy.sum(sigma_yr/(A*r_yr))

        # sum over sigma_zl and sigma_zr -- we are at a fixed z
        r_zl = numpy.sqrt( (a.x3d[:,:,a.klo] - a.xmax)**2 + (a.y3d[:,:,a.klo] - a.y[jj])**2 + (a.zmin - a.z[kk])**2 )
        r_zr = numpy.sqrt( (a.x3d[:,:,a.khi] - a.xmax)**2 + (a.y3d[:,:,a.khi] - a.y[jj])**2 + (a.zmin - a.z[kk])**2 )

        Phi_xr[jj,kk] += -numpy.sum(sigma_zl/(A*r_zl))
        Phi_xr[jj,kk] += -numpy.sum(sigma_zr/(A*r_zr))

Phi_xr *= a.dy*a.dz

print numpy.min(Phi_xr[a.jlo:a.jhi+1,a.klo:a.khi+1]), numpy.max(Phi_xr[a.jlo:a.jhi+1,a.klo:a.khi+1])

# -y face
Phi_yl = numpy.zeros(sigma_yl.shape)

for kk in range(a.klo, a.khi+1):
    for ii in range(a.ilo, a.ihi+1):

        # we are in the y = ymin plane

        # sigma coords - Phi coords

        # sum over sigma_xl and sigma_xr -- we are at a fixed x
        r_xl = numpy.sqrt( (a.xmin - a.x[ii])**2 + (a.y3d[a.ilo,:,:] - a.ymin)**2 + (a.z3d[a.ilo,:,:] - a.z[kk])**2 )
        r_xr = numpy.sqrt( (a.xmax - a.x[ii])**2 + (a.y3d[a.ihi,:,:] - a.ymin)**2 + (a.z3d[a.ihi,:,:] - a.z[kk])**2 )

        Phi_yl[ii,kk] += -numpy.sum(sigma_xl/(A*r_xl))
        Phi_yl[ii,kk] += -numpy.sum(sigma_xr/(A*r_xr))

        # sum over sigma_yl and sigma_yr -- we are at a fixed y
        r_yl = numpy.sqrt( (a.x3d[:,a.jlo,:] - a.x[ii])**2 +                        (a.z3d[:,a.jlo,:] - a.z[kk])**2 )
        r_yr = numpy.sqrt( (a.x3d[:,a.jhi,:] - a.x[ii])**2 + (a.ymax - a.ymin)**2 + (a.z3d[:,a.jhi,:] - a.z[kk])**2 )

        Phi_yl[ii,kk] += -numpy.sum(numpy.where( r_yl == 0.0, 0.0, sigma_yl/(A*r_yl) ))
        Phi_yl[ii,kk] += -numpy.sum(sigma_yr/(A*r_yr))

        # sum over sigma_zl and sigma_zr -- we are at a fixed z
        r_zl = numpy.sqrt( (a.x3d[:,:,a.klo] - a.x[ii])**2 + (a.y3d[:,:,a.klo] - a.ymin)**2 + (a.zmin - a.z[kk])**2 )
        r_zr = numpy.sqrt( (a.x3d[:,:,a.klo] - a.x[ii])**2 + (a.y3d[:,:,a.khi] - a.ymin)**2 + (a.zmax - a.z[kk])**2 )

        Phi_yl[ii,kk] += -numpy.sum(sigma_zl/(A*r_zl))
        Phi_yl[ii,kk] += -numpy.sum(sigma_zr/(A*r_zr))

Phi_yl *= a.dx*a.dy

print numpy.min(Phi_yl[a.jlo:a.jhi+1,a.klo:a.khi+1]), numpy.max(Phi_yl[a.jlo:a.jhi+1,a.klo:a.khi+1])

# +y face
Phi_yr = numpy.zeros(sigma_yr.shape)

for kk in range(a.klo, a.khi+1):
    for ii in range(a.ilo, a.ihi+1):

        # we are in the y = ymax plane

        # sigma coords - Phi coords

        # sum over sigma_xl and sigma_xr -- we are at a fixed x
        r_xl = numpy.sqrt( (a.xmin - a.x[ii])**2 + (a.y3d[a.ilo,:,:] - a.ymax)**2 + (a.z3d[a.ilo,:,:] - a.z[kk])**2 )
        r_xr = numpy.sqrt( (a.xmax - a.x[ii])**2 + (a.y3d[a.ihi,:,:] - a.ymax)**2 + (a.z3d[a.ihi,:,:] - a.z[kk])**2 )

        Phi_yr[ii,kk] += -numpy.sum(sigma_xl/(A*r_xl))
        Phi_yr[ii,kk] += -numpy.sum(sigma_xr/(A*r_xr))

        # sum over sigma_yl and sigma_yr -- we are at a fixed y
        r_yl = numpy.sqrt( (a.x3d[:,a.jlo,:] - a.x[ii])**2 + (a.ymin - a.ymax)**2 + (a.z3d[:,a.jlo,:] - a.z[kk])**2 )
        r_yr = numpy.sqrt( (a.x3d[:,a.jhi,:] - a.x[ii])**2 +                        (a.z3d[:,a.jhi,:] - a.z[kk])**2 )

        Phi_yr[ii,kk] += -numpy.sum(sigma_yl/(A*r_yl))
        Phi_yr[ii,kk] += -numpy.sum(numpy.where( r_yr == 0.0, 0.0, sigma_yr/(A*r_yr) ))

        # sum over sigma_zl and sigma_zr -- we are at a fixed z
        r_zl = numpy.sqrt( (a.x3d[:,:,a.klo] - a.x[ii])**2 + (a.y3d[:,:,a.klo] - a.ymax)**2 + (a.zmin - a.z[kk])**2 )
        r_zr = numpy.sqrt( (a.x3d[:,:,a.klo] - a.x[ii])**2 + (a.y3d[:,:,a.khi] - a.ymax)**2 + (a.zmax - a.z[kk])**2 )

        Phi_yr[ii,kk] += -numpy.sum(sigma_zl/(A*r_zl))
        Phi_yr[ii,kk] += -numpy.sum(sigma_zr/(A*r_zr))

Phi_yr *= a.dx*a.dy

print numpy.min(Phi_yr[a.jlo:a.jhi+1,a.klo:a.khi+1]), numpy.max(Phi_yr[a.jlo:a.jhi+1,a.klo:a.khi+1])

# -z face
Phi_zl = numpy.zeros(sigma_zl.shape)

for jj in range(a.jlo, a.jhi+1):
    for ii in range(a.ilo, a.ihi+1):

        # we are in the z = zmin plane

        # sigma coords - Phi coords

        # sum over sigma_xl and sigma_xr -- we are at a fixed x
        r_xl = numpy.sqrt( (a.xmin - a.x[ii])**2 + (a.y3d[a.ilo,:,:] - a.y[jj])**2 + (a.z3d[a.ilo,:,:] - a.zmin)**2 )
        r_xr = numpy.sqrt( (a.xmax - a.x[ii])**2 + (a.y3d[a.ihi,:,:] - a.y[jj])**2 + (a.z3d[a.ihi,:,:] - a.zmin)**2 )

        Phi_zl[ii,jj] += -numpy.sum(sigma_xl/(A*r_xl))
        Phi_zl[ii,jj] += -numpy.sum(sigma_xr/(A*r_xr))

        # sum over sigma_yl and sigma_yr -- we are at a fixed y
        r_yl = numpy.sqrt( (a.x3d[:,a.jlo,:] - a.x[ii])**2 + (a.ymin - a.y[jj])**2 + (a.z3d[:,a.jlo,:] - a.zmin)**2 )
        r_yr = numpy.sqrt( (a.x3d[:,a.jhi,:] - a.x[ii])**2 + (a.ymax - a.y[jj])**2 + (a.z3d[:,a.jhi,:] - a.zmin)**2 )

        Phi_zl[ii,jj] += -numpy.sum(sigma_yl/(A*r_yl))
        Phi_zl[ii,jj] += -numpy.sum(sigma_yr/(A*r_yr))

        # sum over sigma_zl and sigma_zr -- we are at a fixed z
        r_zl = numpy.sqrt( (a.x3d[:,:,a.klo] - a.x[ii])**2 + (a.y3d[:,:,a.klo] - a.y[jj])**2                        )
        r_zr = numpy.sqrt( (a.x3d[:,:,a.khi] - a.x[ii])**2 + (a.y3d[:,:,a.khi] - a.y[jj])**2 + (a.zmax - a.zmin)**2 )

        Phi_zl[ii,jj] += -numpy.sum(numpy.where( r_zl == 0.0, 0.0, sigma_zl/(A*r_zl) )) 
        Phi_zl[ii,jj] += -numpy.sum(sigma_zr/(A*r_zr))

Phi_zl *= a.dx*a.dy

print numpy.min(Phi_zl[a.jlo:a.jhi+1,a.klo:a.khi+1]), numpy.max(Phi_zl[a.jlo:a.jhi+1,a.klo:a.khi+1])

# +z face
Phi_zr = numpy.zeros(sigma_zr.shape)

for jj in range(a.jlo, a.jhi+1):
    for ii in range(a.ilo, a.ihi+1):

        # we are in the z = zmax plane

        # sigma coords - Phi coords

        # sum over sigma_xl and sigma_xr -- we are at a fixed x
        r_xl = numpy.sqrt( (a.xmin - a.x[ii])**2 + (a.y3d[a.ilo,:,:] - a.y[jj])**2 + (a.z3d[a.ilo,:,:] - a.zmax)**2 )
        r_xr = numpy.sqrt( (a.xmax - a.x[ii])**2 + (a.y3d[a.ihi,:,:] - a.y[jj])**2 + (a.z3d[a.ihi,:,:] - a.zmax)**2 )

        Phi_zr[ii,jj] += -numpy.sum(sigma_xl/(A*r_xl))
        Phi_zr[ii,jj] += -numpy.sum(sigma_xr/(A*r_xr))

        # sum over sigma_yl and sigma_yr -- we are at a fixed y
        r_yl = numpy.sqrt( (a.x3d[:,a.jlo,:] - a.x[ii])**2 + (a.ymin - a.y[jj])**2 + (a.z3d[:,a.jlo,:] - a.zmax)**2 )
        r_yr = numpy.sqrt( (a.x3d[:,a.jhi,:] - a.x[ii])**2 + (a.ymax - a.y[jj])**2 + (a.z3d[:,a.jhi,:] - a.zmax)**2 )

        Phi_zr[ii,jj] += -numpy.sum(sigma_yl/(A*r_yl))
        Phi_zr[ii,jj] += -numpy.sum(sigma_yr/(A*r_yr))

        # sum over sigma_zl and sigma_zr -- we are at a fixed z
        r_zl = numpy.sqrt( (a.x3d[:,:,a.klo] - a.x[ii])**2 + (a.y3d[:,:,a.klo] - a.y[jj])**2 + (a.zmin - a.zmax)**2 )
        r_zr = numpy.sqrt( (a.x3d[:,:,a.khi] - a.x[ii])**2 + (a.y3d[:,:,a.khi] - a.y[jj])**2                        )

        Phi_zr[ii,jj] += -numpy.sum(sigma_zl/(A*r_zl)) 
        Phi_zr[ii,jj] += -numpy.sum(numpy.where( r_zr == 0.0, 0.0, sigma_zr/(A*r_zr) ))
                
Phi_zr *= a.dx*a.dy

print numpy.min(Phi_zr[a.jlo:a.jhi+1,a.klo:a.khi+1]), numpy.max(Phi_zr[a.jlo:a.jhi+1,a.klo:a.khi+1])
                    

# 4. solve for Phi

# we do inhomogeneous BCs for this, and solve Laplace's equation.  The
# BCs are simply the Phi's on the surfaces constructed above.
a = multigrid.ccMG3d(nx, ny, nz,
                     xlBCtype="dirichlet", xrBCtype="dirichlet",
                     ylBCtype="dirichlet", yrBCtype="dirichlet",
                     zlBCtype="dirichlet", zrBCtype="dirichlet",
                     xlBC=Phi_xl, xrBC=Phi_xr,
                     ylBC=Phi_yl, yrBC=Phi_yr,
                     zlBC=Phi_zl, zrBC=Phi_zr,
                     verbose=1)

# initialize the solution to 0
a.initZeros()

# initialize the RHS to 0 -- Laplace's equation
a.initRHS(a.solnGrid.scratchArray())

# solve to a relative tolerance of 1.e-11
a.solve(rtol=1.e-11)

# get the solution -- this is the homogeneous potential
Phi = a.getSolution()


# 5. compute the isolated potential
phi = phi_h - Phi


#-----------------------------------------------------------------------------
# compute the error from the analytic solution
b = true(masses,a.x3d,a.y3d,a.z3d)
e = phi - b

print " L2 error from true solution = %g\n rel. err from previous cycle = %g\n num. cycles = %d" % \
      (error(a.solnGrid, e), a.relativeError, a.numCycles)


#-----------------------------------------------------------------------------
# visualize -- slices through the center
v = phi
pylab.subplot(131)
pylab.imshow(v[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo+a.nz/2])

pylab.subplot(132)
pylab.imshow(v[a.ilo:a.ihi+1,a.jlo+a.ny/2,a.klo:a.khi+1])

pylab.subplot(133)
pylab.imshow(v[a.ilo+a.nx/2,a.jlo:a.jhi+1,a.klo:a.khi+1])

pylab.show()


