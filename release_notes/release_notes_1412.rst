====================
What's new in 1.4.12
====================

tofu 1.4.12 is a minor upgrade from 1.4.11


Main changes:
=============

- Mesh2DRect class and bsplines implemented #545
- Geometry matrix computation implemented #549
- Operators for integration implemented #551
- CrystalBragg (spherical), splittable, implemented #532 #538
- Non-parallelism implemented for Crystals #531, #542
- Possibility of getting local coordinates of an arbitrary detector implemented #541
- 1d spectrum fitting implemented #532
- Solid angle integrated toroidally for spherical particle implemented #535
- IMAS interface improved (more robust) #530


Detailed changes:
=================

Mesh and basis functions:
~~~~~~~~~~~~~~~~~~~~~~~~~
- A generic Mesh2DRect class is created, can create an arbitrary non-uniform rectangular mesh from a tokamak geometry, provides cropping and plotting #545
- Bivariate b-splines of degree 0, 1, 2, 3 can be used on that mesh #545
- Operators D0, D0N2, D1N2, D2N2 can be computed for any bsplines of deg 0, 1, 2, in toroidal and linear geometry #551

Geometry matrix:
~~~~~~~~~~~~~~~~
- Mesh2DRect.compute_geometry_matrix() implemented for LOS-approximation, reasonably fast, with plotting routines and cropping #549

Imaging X-Ray spectrometer:
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- A CrystalBragg class is created to model a 2D spherically curved X-ray spectrometer, to be used with a detector, with ray-tracing and plots #532
- The non-parallelism between optical surface and crystal lattice can be set and taken into account in ray-tracing, to account for defects on WEST spectrometer (@adriendaros) #542
- Crystal splitting in halves implemented, as non-parallelism to can set for each half independently (@adriendaros) #538
- The possibility of getting the local coordinates of an arbitrary detector in the Crystal's frame is implemented (@adriendaros) #541

Spectral fitting:
~~~~~~~~~~~~~~~~~
- 1d spectrum fitting routine implemented, with lot of flexibility and re-use of solution for next time step #532
- 2d fitting pre-implemented, to be finished #532

Solid angles:
~~~~~~~~~~~~~
- The toroidal integral of the solid angles subtended by a spherical particle can be computed (@lasofivec) #535

IMAS interfacing:
~~~~~~~~~~~~~~~~~
- IMAS interface more robust when writing data to bolometers IDS #530

Portability / CI:
~~~~~~~~~~~~~~~~~
- Github Actions and porting of travis-ci.org to travis-ci.com fixed (@lasofivec)

Miscellaneous:
~~~~~~~~~~~~~~
- More detailed error messages
- More unit tests

Contributors:
=============
Many thanks to all developpers and contributors:
- Didier Vezinet (@Didou09)
- Laura Mendoza (@lasofivec)
- Adrien Da Ros (@adriendaros)

What's next (indicative):
=========================
- Solid angles for Volume-Of-Sight and radiative heat loads computation (ongoing for @lasofivec : Issues #74, #75, #76, #77, #78)
- Basic tools for inversions
- Generic data class to incorporate plateau-finding, data analysis and 1d Bayesian fitting routines and classes (ongoing for @Didou09 and @jmoralesFusion and @MohammadKozeiha: issues #208, #260 and #262)
- More general magnetic field line tracing workflow
- Better unit tests coverage
- More complete documentation


List of PR merged into this release:
====================================
- PR: #530, #531, #532, #535, #538, #541, #542, #545, #549, #551
