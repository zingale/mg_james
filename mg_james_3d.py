#!/usr/bin/env python

"""

Solve the Poisson problem with isolated BCs with a collection of point masses

Superposition tells us the analytic solution

"""

import math
import numpy as np
import patch
import multigrid
import pylab

G = 1.0

nx = 64

ny = nx
nz = nx

xmin = -1.0
ymin = -1.0
zmin = -1.0

xmax = 1.0
ymax = 1.0
zmax = 1.0

#-----------------------------------------------------------------------------
# the analytic solution
def true(grid, x, y, z):

    phi = np.zeros(np.shape(x))

    c = np.zeros( (2, 3) )

    for kk in range(0, grid.nz + 2):
        z = grid.z[kk]
        c[:,2] = [ -0.5 - z, 0.5 - z ]
        for jj in range(0, grid.ny + 2):
            y = grid.y[jj]
            c[:,1] = [ -0.5 - y, 0.5 - y ]
            for ii in range(0, grid.nx + 2):
                x = grid.x[ii]
                c[:,0] = [ -0.5 - x, 0.5 - x ]

                phi[ii,jj,kk] = 0.0

                for i in range(0,2):
                    for j in range(0,2):
                        for l in range(0,3):

                            num1 = ( (c[i,l]**2 + c[j,(l+1)%3]**2 + c[1,(l+2)%3]**2)**(0.5) + c[1,(l+2)%3] )**3
                            num2 = ( (c[i,l]**2 + c[j,(l+1)%3]**2 + c[0,(l+2)%3]**2)**(0.5) - c[0,(l+2)%3] )
                            den1 = ( (c[i,l]**2 + c[j,(l+1)%3]**2 + c[1,(l+2)%3]**2)**(0.5) - c[1,(l+2)%3] )
                            den2 = ( (c[i,l]**2 + c[j,(l+1)%3]**2 + c[0,(l+2)%3]**2)**(0.5) + c[0,(l+2)%3] )**3

                            phi[ii,jj,kk] += 0.5 * (-1)**(i+j) * ( c[i,l] * c[j,(l+1)%3] * 
                                             math.log( num1 * num2 / (den1 * den2) ) )

                for i in range(0,2):
                    for j in range(0,2):
                        for k in range(0,2):
                            for l in range(0,3):
                                phi[ii,jj,kk] += (-1)**(i+j+k+1) * (c[i,l]**2 * 
                                                 math.atan2( c[i,l] * c[k,(l+2)%3], c[i,l]**2 + c[j,(l+1)%3]**2 + 
                                                             c[j,(l+1)%3]*(c[i,l]**2 + c[j,(l+1)%3]**2 + c[k,(l+2)%3]**2)**(0.5) ) )

                phi[ii,jj,kk] *= -0.5 * G

    return phi

#-----------------------------------------------------------------------------
# the righthand side -- this initializes a cube in the center of the domain.
# this assumes that our domain runs from -1 to 1 in each coordinate direction.
def f(grid):

    f = grid.scratchArray()

    rho = 1.0

    # indices of a cube from -0.5 to 0.5 in the domain
    index = np.logical_and(abs(grid.x3d) < 0.5,
                           np.logical_and(abs(grid.y3d) < 0.5, 
                                          abs(grid.z3d) < 0.5))

    f[index] = 4.0*np.pi*G*rho

    return f


#-----------------------------------------------------------------------------
# the L2 error norm
def error(g, r):

    # L2 norm of the *faces*

    err = 0.0
    err += np.sum((r[g.ilo,        g.jlo:g.jhi+1,g.klo:g.khi+1]**2).flat)
    err += np.sum((r[g.ihi,        g.jlo:g.jhi+1,g.klo:g.khi+1]**2).flat)
    err += np.sum((r[g.ilo:g.ihi+1,g.jlo,        g.klo:g.khi+1]**2).flat)
    err += np.sum((r[g.ilo:g.ihi+1,g.jhi,        g.klo:g.khi+1]**2).flat)
    err += np.sum((r[g.ilo:g.ihi+1,g.jlo:g.jhi+1,g.klo        ]**2).flat)
    err += np.sum((r[g.ilo:g.ihi+1,g.jlo:g.jhi+1,g.khi        ]**2).flat)

    return np.sqrt(g.dx*g.dy*g.dz*err)


#-----------------------------------------------------------------------------
# the Green's function
def green(r):

    if (np.size(r) == 1):
        if (r != 0.0):
            gf = -G / r
        else:
            gf = -2.38008 * G / a.dx

    else:

        x = r[:,:,1]
        y = r[:,:,2]
        z = r[:,:,3]
        r = r[:,:,0]

        gf = np.zeros(r.shape)

        gf[:,:] = -2.38008 * G / a.dx

        loc = np.where(r != 0.0)

        x = x[loc]
        y = y[loc]
        z = z[loc]
        r = r[loc]

        u = x / r
        v = y / r
        w = z / r

        eta1 = 1.0

        U20 = a.dx**2 * u**0 + a.dy**2 * v**0 + a.dz**2 * w**0
        U22 = a.dx**2 * u**2 + a.dy**2 * v**2 + a.dz**2 * w**2
        U24 = a.dx**2 * u**4 + a.dy**2 * v**4 + a.dz**2 * w**4

        eta3 = 1.0 / 8.0 * (U20 - 6 * U22 + 5 * U24)

        U40 = a.dx**4 * u**0 + a.dy**4 * v**0 + a.dz**4 * w**0
        U42 = a.dx**4 * u**2 + a.dy**4 * v**2 + a.dz**4 * w**2
        U44 = a.dx**4 * u**4 + a.dy**4 * v**4 + a.dz**4 * w**4
        U46 = a.dx**4 * u**6 + a.dy**4 * v**6 + a.dz**4 * w**6

        eta5 = 1.0 / 128.0 * ( U20 * (3 * U20 - 60 * U22 + 70 * U24) + \
                               U22 * (420 * U22 - 1260 * U24) + 1155 * U24**2 + \
                               24 * U40 - 520 * U42 + 1680 * U44 - 1512 * U46 )

        gf[loc] = -G * ( eta1 * r**(-1) + eta3 * r**(-3) + eta5 * r**(-5) )

    return gf

#-----------------------------------------------------------------------------
# convolve the Green's functions with the masses
def convolve(mass_list, r_list):

    phi = 0.0

    # loop over the 6 faces
    for i in range(6):
 
        phi += np.sum(green(r_list[i]) * mass_list[i]) 

    return phi


#-----------------------------------------------------------------------------
# Create the distance functions that we're going to use for convolving over
def create_r(x,y,z,a):

    # we are at a fixed x

    r_xl = np.zeros((a.qy, a.qz, 4))
    r_xr = np.zeros((a.qy, a.qz, 4))

    #r_xl[:,:,1] = a.xmin - x
    r_xl[:,:,1] = a.x3d[a.ilo,:,:] - x
    r_xl[:,:,2] = a.y3d[a.ilo,:,:] - y
    r_xl[:,:,3] = a.z3d[a.ilo,:,:] - z

    r_xl[:,:,0] = np.sqrt(r_xl[:,:,1]**2 + r_xl[:,:,2]**2 + r_xl[:,:,3]**2)

    #r_xr[:,:,1] = a.xmax - x
    r_xr[:,:,1] = a.x3d[a.ihi,:,:] - x
    r_xr[:,:,2] = a.y3d[a.ihi,:,:] - y
    r_xr[:,:,3] = a.z3d[a.ihi,:,:] - z

    r_xr[:,:,0] = np.sqrt(r_xr[:,:,1]**2 + r_xr[:,:,2]**2 + r_xr[:,:,3]**2)

    # we are at a fixed y
    
    r_yl = np.zeros((a.qx, a.qz, 4))
    r_yr = np.zeros((a.qx, a.qz, 4))
    
    r_yl[:,:,1] = a.x3d[:,a.jlo,:] - x
    #r_yl[:,:,2] = a.ymin - y
    r_yl[:,:,2] = a.y3d[:,a.jlo,:] - y
    r_yl[:,:,3] = a.z3d[:,a.jlo,:] - z

    r_yl[:,:,0] = np.sqrt(r_yl[:,:,1]**2 + r_yl[:,:,2]**2 + r_yl[:,:,3]**2)

    r_yr[:,:,1] = a.x3d[:,a.jhi,:] - x
    #r_yr[:,:,2] = a.ymax - y
    r_yr[:,:,2] = a.y3d[:,a.jhi,:] - y
    r_yr[:,:,3] = a.z3d[:,a.jhi,:] - z

    r_yr[:,:,0] = np.sqrt(r_yr[:,:,1]**2 + r_yr[:,:,2]**2 + r_yr[:,:,3]**2)
       
    # we are at a fixed z

    r_zl = np.zeros((a.qx, a.qy, 4))
    r_zr = np.zeros((a.qx, a.qy, 4))

    r_zl[:,:,1] = a.x3d[:,:,a.klo] - x
    r_zl[:,:,2] = a.y3d[:,:,a.klo] - y
    #r_zl[:,:,3] = a.zmin - z
    r_zl[:,:,3] = a.z3d[:,:,a.klo] - z

    r_zl[:,:,0] = np.sqrt(r_zl[:,:,1]**2 + r_zl[:,:,2]**2 + r_zl[:,:,3]**2)

    r_zr[:,:,1] = a.x3d[:,:,a.khi] - x
    r_zr[:,:,2] = a.y3d[:,:,a.khi] - y
    #r_zr[:,:,3] = a.zmax - z
    r_zr[:,:,3] = a.z3d[:,:,a.khi] - z
    
    r_zr[:,:,0] = np.sqrt(r_zr[:,:,1]**2 + r_zr[:,:,2]**2 + r_zr[:,:,3]**2)

    return (r_xl, r_xr, r_yl, r_yr, r_zl, r_zr)


#-----------------------------------------------------------------------------
# the main solve -- James algorithm
#-----------------------------------------------------------------------------

# 1. solve the problem with homogeneous Dirichlet BCs

# create the multigrid object
a = multigrid.ccMG3d(nx, ny, nz,
                     xmin=xmin, ymin=ymin, zmin=zmin,
                     xmax=xmax, ymax=ymax, zmax=zmax,
                     xlBCtype="dirichlet-ho", xrBCtype="dirichlet-ho",
                     ylBCtype="dirichlet-ho", yrBCtype="dirichlet-ho",
                     zlBCtype="dirichlet-ho", zrBCtype="dirichlet-ho",
                     smoother="GS",
                     verbose=0)

# initialize the solution to 0
a.initZeros()

# initialize the RHS using the function f
rhs = f(a.solnGrid)
a.initRHS(rhs)

mass = np.sum(rhs) / (4.0 * math.pi * G) * a.dx * a.dy * a.dz

# solve to a relative tolerance of 1.e-11
a.solve(rtol=1.e-11)
#a.smooth(a.nlevels-1, 10000)

# get the solution -- this is the homogeneous potential
phi_h = a.getSolution()


# 2. compute surface masses

const = 1.0 / (4.0 * math.pi * G)

# sigma = grad phi_h . n / (4pi G)
# mass  = sigma . dA

# Make sure that we only sum over masses adjacent to the domain
# faces. Edges and corners don't count, so we leave those entries
# zeroed out.

# we compute grad phi_h . n by constructing a quadratic interpolant
# whose average reproduces the cell values of the first two interior
# zones and is equal to 0 on the boundary.  This gives:
#
# phi'_{-1/2} = (7 phi_0 - phi_1)/(2 dx) + O(dx**2)

dA = a.dy * a.dz

mass_xl = np.zeros(phi_h[a.ilo,:,:].shape)
mass_xr = np.zeros(phi_h[a.ihi,:,:].shape)

mass_xl[a.jlo:a.jhi+1,a.klo:a.khi+1] = -0.5 * const * dA * \
        (7.0 * phi_h[a.ilo,  a.jlo:a.jhi+1,a.klo:a.khi+1] - 
               phi_h[a.ilo+1,a.jlo:a.jhi+1,a.klo:a.khi+1]) / a.dx 
mass_xr[a.jlo:a.jhi+1,a.klo:a.khi+1] = -0.5 * const * dA * \
        (7.0 * phi_h[a.ihi,  a.jlo:a.jhi+1,a.klo:a.khi+1] - 
               phi_h[a.ihi-1,a.jlo:a.jhi+1,a.klo:a.khi+1]) / a.dx 

dA = a.dx * a.dz

mass_yl = np.zeros(phi_h[:,a.jlo,:].shape)
mass_yr = np.zeros(phi_h[:,a.jhi,:].shape)

mass_yl[a.ilo:a.ihi+1,a.klo:a.khi+1] = -0.5 * const * dA * \
        (7.0 * phi_h[a.ilo:a.ihi+1,a.jlo,  a.klo:a.khi+1] - 
               phi_h[a.ilo:a.ihi+1,a.jlo+1,a.klo:a.khi+1]) / a.dy 
mass_yr[a.ilo:a.ihi+1,a.klo:a.khi+1] = -0.5 * const * dA * \
        (7.0 * phi_h[a.ilo:a.ihi+1,a.jhi,  a.klo:a.khi+1] - 
               phi_h[a.ilo:a.ihi+1,a.jhi-1,a.klo:a.khi+1]) / a.dy 

dA = a.dx * a.dy

mass_zl = np.zeros(phi_h[:,:,a.klo].shape)
mass_zr = np.zeros(phi_h[:,:,a.khi].shape)

mass_zl[a.ilo:a.ihi+1,a.jlo:a.jhi+1] = -0.5 * const * dA * \
        (7.0 * phi_h[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo  ] - 
               phi_h[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo+1]) / a.dz 
mass_zr[a.ilo:a.ihi+1,a.jlo:a.jhi+1] = -0.5 * const * dA * \
        (7.0 * phi_h[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.khi  ] - 
               phi_h[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.khi-1]) / a.dz 

mass_list = (mass_xl, mass_xr, mass_yl, mass_yr, mass_zl, mass_zr)

totalMass = np.sum(mass_list)
trueMass = 1.0
actualMass = mass

print "Total surface mass = ", totalMass
print "Total mass in the domain =", actualMass
print "True mass in cube =", trueMass
print "Relative error in surface mass to actual mass = ", abs(totalMass - actualMass) / actualMass
print "Relative error in surface mass to true mass =", abs(totalMass - trueMass) / trueMass

# 3. compute the boundary conditions on phi

# -x face
Phi_xl = np.zeros(mass_xl.shape)

for kk in range(a.klo, a.khi+1):
    for jj in range(a.jlo, a.jhi+1):

        # we are in the x = xmin plane

        x = a.xmin
        y = a.y[jj]
        z = a.z[kk]

        r_list = create_r(x,y,z,a)

        Phi_xl[jj,kk] = convolve(mass_list, r_list)

#print np.min(Phi_xl[a.jlo:a.jhi+1,a.klo:a.khi+1]), np.max(Phi_xl[a.jlo:a.jhi+1,a.klo:a.khi+1])

# +x face
Phi_xr = np.zeros(mass_xr.shape)

for kk in range(a.klo, a.khi+1):
    for jj in range(a.jlo, a.jhi+1):

        # we are in the x = xmax plane

        x = a.xmax
        y = a.y[jj]
        z = a.z[kk]

        r_list = create_r(x,y,z,a)

        Phi_xr[jj,kk] = convolve(mass_list, r_list)
#        true_xr[jj,kk] = true(masses,x,y,z)

#print np.min(Phi_xr[a.jlo:a.jhi+1,a.klo:a.khi+1]), np.max(Phi_xr[a.jlo:a.jhi+1,a.klo:a.khi+1])

# -y face
Phi_yl = np.zeros(mass_yl.shape)

for kk in range(a.klo, a.khi+1):
    for ii in range(a.ilo, a.ihi+1):

        # we are in the y = ymin plane

        x = a.x[ii]
        y = a.ymin
        z = a.z[kk]

        r_list = create_r(x,y,z,a)

        Phi_yl[ii,kk] = convolve(mass_list, r_list)

#print np.min(Phi_yl[a.ilo:a.ihi+1,a.klo:a.khi+1]), np.max(Phi_yl[a.ilo:a.ihi+1,a.klo:a.khi+1])

# +y face
Phi_yr = np.zeros(mass_yr.shape)

for kk in range(a.klo, a.khi+1):
    for ii in range(a.ilo, a.ihi+1):

        # we are in the y = ymax plane

        x = a.x[ii]
        y = a.ymax
        z = a.z[kk]

        r_list = create_r(x,y,z,a)

        Phi_yr[ii,kk] = convolve(mass_list, r_list)

#print np.min(Phi_yr[a.ilo:a.ihi+1,a.klo:a.khi+1]), np.max(Phi_yr[a.ilo:a.ihi+1,a.klo:a.khi+1])

# -z face
Phi_zl = np.zeros(mass_zl.shape)

for jj in range(a.jlo, a.jhi+1):
    for ii in range(a.ilo, a.ihi+1):

        # we are in the z = zmin plane

        x = a.x[ii]
        y = a.y[jj]
        z = a.zmin

        r_list = create_r(x,y,z,a)

        Phi_zl[ii,jj] = convolve(mass_list, r_list)

#print np.min(Phi_zl[a.ilo:a.ihi+1,a.jlo:a.jhi+1]), np.max(Phi_zl[a.ilo:a.ihi+1,a.ilo:a.ihi+1])

# +z face
Phi_zr = np.zeros(mass_zr.shape)

for jj in range(a.jlo, a.jhi+1):
    for ii in range(a.ilo, a.ihi+1):

        # we are in the z = zmax plane

        x = a.x[ii]
        y = a.y[jj]
        z = a.zmax

        r_list = create_r(x,y,z,a)

        Phi_zr[ii,jj] = convolve(mass_list, r_list)
                
#print np.min(Phi_zr[a.ilo:a.ihi+1,a.ilo:a.ihi+1]), np.max(Phi_zr[a.ilo:a.ihi+1,a.ilo:a.ihi+1])


# 4. solve for the isolated potential

# we do inhomogeneous BCs for this, and solve Laplace's equation.  The
# BCs are simply the Phi's on the surfaces constructed above.
b = multigrid.ccMG3d(nx, ny, nz,
                     xmin=xmin, ymin=ymin, zmin=zmin,
                     xmax=xmax, ymax=ymax, zmax=zmax,
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

print "getting true"

t = true(a.solnGrid, a.x3d, a.y3d, a.z3d)

e = phi - t

print " L2 error from true solution = %g\n rel. err from previous cycle = %g\n num. cycles = %d" % \
      (error(a.solnGrid, e) / error(a.solnGrid,t), a.relativeError, a.numCycles)

print "Min (phi) = ", np.min(phi[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo:a.khi+1]), "Max (phi) = ", np.max(phi[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo:a.khi+1])
print "Min (true) = ", np.min(t[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo:a.khi+1]), "Max (true) = ", np.max(t[a.ilo:a.ihi+1,a.jlo:a.jhi+1,a.klo:a.khi+1])

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

pylab.savefig("plot.png")

#pylab.show()
