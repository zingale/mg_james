mg_james
========

A test 3-d multigrid solver implementing the James algorithm for
isolated boundaries.

This is experimental.  The 3-d MG solver is based on the solver from
pyro.  The inhomogeneous Dirichlet BCs appear to work correctly,
converging as second-order when run via:

mg_test_inhomogeneous.py

That test problem was derived by picking the solution first and then
taking the Laplacian (via sympy) to get the RHS and evaluating the
boundary conditions as well.  This is shown in the notebook:

inhomogeneous-bcs.ipynb

