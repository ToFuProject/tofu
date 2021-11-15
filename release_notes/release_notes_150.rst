====================
What's new in 1.4.12
====================

tofu 1.5.0 is a major upgrade from 1.4.15
It is the first release with basic tools for inversions


Main changes:
=============

- Better CI, packaging and doc #554, #556, #558, #560
- Implemented basics for tomographic inversions on rectangular meshes #561, #562, #570
- Implemented basics for triangular meshes #571
- More robust Config object #562
- Basic standardized speed and memory benchmarks implemented using asv #573
- New tokamak geometry #577
- Optimized solid angle toroidal integral for spherical particle #578


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
- Kevin Obrejan (@obrejank)
- Florian Le Bourdais (@flothesof)
- Adrien Da Ros (@adriendaros)

What's next (indicative):
=========================
- Solid angles for Volume-Of-Sight and radiative heat loads computation (ongoing: Issues #74, #75, #76, #77, #78)
- Basic tools for inversions on triangular meshes
- Basic tools for inversions on global basis functions
- Generic data class to incorporate plateau-finding, data analysis and 1d Bayesian fitting routines and classes (ongoing for @Didou09 and @jmoralesFusion and @MohammadKozeiha: issues #208, #260 and #262)
- More general magnetic field line tracing workflow
- Better unit tests coverage
- Better benchmarks
- More complete documentation
- Reference article

List of PR merged into this release:
====================================
- PR: #554, #556, #558, #560, #561, #562, #564, #570, #571, #573, #575, #577, #578
