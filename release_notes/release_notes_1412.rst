====================
What's new in 1.4.10
====================

tofu 1.4.12 is a minor upgrade from 1.4.11


Main changes:
=============

- Mesh2DRect class and bsplines implemented #545
- Geometry matrix computation implemented #549
- Operators for integration implemented, in linear and toroidal versions #551
- IMAS interface improved (more robust) #530
- Non-parallelism implemented for Crystals #531, #542
- Crystal spherical implemented, with plots and ray-tracing #532
- 1d spectrum fitting implemented #532
- Crystal splitting implemented #538
- Possibility of getting local coordinates of an arbitrary detector implemented #541
- Solid angle integrated toroidally for spherical particle implemented #535


Detailed changes:
=================

Geometry:
~~~~~~~~~

Generic classes:
~~~~~~~~~~~~~~~~

Visualization:
~~~~~~~~~~~~~~

openadas interfacing:
~~~~~~~~~~~~~~~~~~~~~~

nist interfacing:
~~~~~~~~~~~~~~~~~

IMAS interfacing:
~~~~~~~~~~~~~~~~~

Portability / CI:
~~~~~~~~~~~~~~~~~

Documentation:
~~~~~~~~~~~~~~

Miscellaneous:
~~~~~~~~~~~~~~


Contributors:
=============
Many thanks to all developpers and contributors:
- Didier Vezinet (@Didou09)
- Laura Mendoza (@lasofivec)


What's next (indicative):
=========================
- Solid angles for Volume-Of-Sight and radiative heat loads computation (ongoing for @lasofivec : Issues #74, #75, #76, #77, #78)
- Inversions
- Generic data class to incorporate plateau-finding, data analysis and 1d Bayesian fitting routines and classes (ongoing for @Didou09 and @jmoralesFusion and @MohammadKozeiha: issues #208, #260 and #262)
- More general magnetic field line tracing workflow
- Better unit tests coverage
- More complete documentation


List of PR merged into this release:
====================================
- PR: #530, #531, #532, #535, #538, #541, #542, #545, #549, #551
