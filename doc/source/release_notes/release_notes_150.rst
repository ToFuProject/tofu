====================
What's new in 1.5.0
====================

tofu 1.5.0 is a major upgrade from 1.4.15
It is the first release with basic tools for inversions


Main changes:
=============

- Better CI, packaging and doc #554, #556, #558, #560
- Implemented basics for tomographic inversions on rectangular meshes #561, #564, #570
- Implemented basics for triangular meshes #571
- More robust Config object #562
- Basic standardized speed and memory benchmarks implemented using asv #573
- New tokamak geometry #577
- Optimized solid angle toroidal integral for spherical particle #578


Detailed changes:
=================

Mesh and basis functions:
~~~~~~~~~~~~~~~~~~~~~~~~~
- Mesh2DRect renamed into Mesh2D to handle both rectangular and triangular meshes
  Triangular meshes handle bsplines of degree 0 and 1
  Interpolation, geometry matrix computation and inversion operators remain to be done for triangular meshes #571
- Mesh2D.add_inversion() implemented for rectangular meshes, for linear operators up to degree 2 #561
- Mesh2D.add_inversion() implemented for discrete gradient operator for deg 0, in non-centered version #564, #570

Config object:
~~~~~~~~~~~~~~
-Config object now more robust against deleting of all structures #562

Tokamak geometries:
~~~~~~~~~~~~~~~~~~~
-Added COMPASS simplified 2D geometry in 2 versions #577

CI, packaging and doc:
~~~~~~~~~~~~~~~~~~~~~~
- Added release notes of versions 1.4.10 and 1.4.12 to doc #564
- Simplified Travis builds (now runs on tagged, master, devel and deploy-test only) #556
- Added missing package data (for test data) #558
- Fixed a selector issue for the sphinx gallery in the doc (for Windows) #560

Benchmarks:
~~~~~~~~~~~
- Created a standardized benchmark suite using the asv library #573
  To be used to follow in time the evolution of the speed and peak memory usage of key methods
  To be used to make sure the optimization of a routine does not slow down other use cases

Optimization / parallelization:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- The routine conf.calc_solidangle_particle_integrated() has been optimized further
  A double loop with unbalanced load on CPUs has been replaced by a single pre-computed and more balanced loop
  The parallelization schedule has been changed (to guided) in some cases
  Two more routines were parallelized
  Results in a very good scaling and 80 % speed gain in the standardized benchmark #578


Contributors:
=============
Many thanks to all developpers and contributors
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
