====================
What's new in 1.4.4
====================

tofu 1.4.4 is a minor upgrade from 1.4.3

Main changes:
=============

- Bug fix in deployment workflow #371
- Bug fix in installation with pip #370
- More robust bash commands for IMAS interfacing #374
- Introduced movements for geometry objects #372
- More informative error messages
- Better PEP8 compliance


Detailed changes:
=================

Installation / portability:
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- conditions for deploying of each platform are more strict (one-liner) #371
- Includes _updateversion.py and .pxd files in sdist for pip installs #370

scripts:
~~~~~~~~
- bash command tofucalc now accepts up to 3 sets of IMAS idd identifiers
  (tokamak, user, shot, run) offering more flexibility to choose the idd for
  geometry, equilibrium and profiles #374


Geom classes (Struct, Config, CamLOS1D, CamLOS2D...):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Parent class now provides basic movement methods: rotations and translations
  All Struct, Rays and CrystalBragg subclasses can use them
  They can have an instance-grade predefined custom default movement #372


Miscellaneous:
~~~~~~~~~~~~~~
- Better PEP8 compliance


Contributors:
=============
Many thanks to all developpers:
- Didier Vezinet (@Didou09)
- Laura Mendoza (@lasofivec)


What's next (indicative):
=========================
- Migrating from nosetests (ongoing for @lasofivec : issues #95 and #232 )
- Easier binary and source installs from pip and conda for all platforms, including unit tests on alla platforms (ongoing for @lasofivec and @flothesof : issue #92 and #259 )
- Solid angles for Volume-Of-Sight and radiative heat loads computation (ongoing for @lasofivec : Issues #71, #72, #73, #74, #75, #76, #77, #78 )
- Tools and classes to handle 2D Bragg X-Ray crystal spectrometer (ongoing for @Didou09 : Issues #202 and #263)
- Generic data class to incorporate plateau-finding, data analysis and 1d Bayesian fitting routines and classes (ongoing for @Didou09 and @jmoralesFusion and @MohammadKozeiha: issues #208, #260 and #262)
- More general magnetic field line tracing workflow
- Better unit tests coverage
- More complete documentation


List of PR merged into this release:
====================================
- PR: #370, #371, #372, #374
