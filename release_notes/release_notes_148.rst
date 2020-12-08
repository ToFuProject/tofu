====================
What's new in 1.4.8
====================

tofu 1.4.8 is a minor upgrade from 1.4.7
It contains bug fixes, optimized volume sampling routines and new tokamak geometries


Main changes:
=============

- Optimized volume sampling #334
- More flexible bash commands #417
- New tokamak geometries #419
- Compatibility with tools_dc/5 on IRFM intra  #423
- Debugging for IMAS compatibility #424, #427
- Updated README for PlasmaPy and conda-forge #426
- Better PEP8 compliance


Detailed changes:
=================

Installation / portability:
~~~~~~~~~~~~~~~~~~~~~~~~~~~
-

scripts:
~~~~~~~~
- Cleaner and more flexible #417

Geometry:
~~~~~~~~~
- New **TCV configuration** available thanks to Oulfa Chellai! #419

IMAS interfacing:
~~~~~~~~~~~~~~~~~
- Debugging for compatibility with tools_dc/5 #396


Geometry:
~~~~~~~~~
- Faster (optimized and parallelized) volume sampling from sub-domain and indices (in toroidql geometry only) #334


Documentation:
~~~~~~~~~~~~~~
-

Miscellaneous:
~~~~~~~~~~~~~~
- Better PEP8 compliance


Contributors:
=============
Many thanks to all developpers and contributors:
- Didier Vezinet (@Didou09)
- Laura Mendoza (@lasofivec)
- Florian Le Bourdais  (@flothesof)
- Oulfa Chellai for TCV geometry


What's next (indicative):
=========================
- Migrating from nosetests (ongoing for @lasofivec : issues #95 and #232)
- Easier binary and source installs from pip and conda for all platforms, including unit tests on all platforms (ongoing for @lasofivec and @flothesof : issue #259)
- Solid angles for Volume-Of-Sight and radiative heat loads computation (ongoing for @lasofivec : Issues #71, #72, #73, #74, #75, #76, #77, #78)
- Tools and classes to handle 2D Bragg X-Ray crystal spectrometer (ongoing for @Didou09 : Issues #202 and #263)
- Generic data class to incorporate plateau-finding, data analysis and 1d Bayesian fitting routines and classes (ongoing for @Didou09 and @jmoralesFusion and @MohammadKozeiha: issues #208, #260 and #262)
- More general magnetic field line tracing workflow
- Better unit tests coverage
- More complete documentation


List of PR merged into this release:
====================================
- PR: #334, #417, #418, #419, #423, #425, #426, #427
