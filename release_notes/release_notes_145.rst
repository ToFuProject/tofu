====================
What's new in 1.4.4
====================

tofu 1.4.5 is a minor upgrade from 1.4.4

Main changes:
=============

- Better portability #378
- Preparing for conda-forge #388
- Better documentation #384
- New bash command to get tofu version # 390
- More informative error messages
- Better PEP8 compliance


Detailed changes:
=================

Installation / portability:
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- default config txt files are now included in sdist #378
- License is now included in sdist (preparing for conda-forge) #388

scripts:
~~~~~~~~
- bash command tofu-version return version of currently installed tofu without
  loading tofu itself (read tofu/version.py, faster) #390
- More robust tofucalc and tofuplot, with less warnings #393

Geometry:
~~~~~~~~~
- New **DEMO 2019 configuration** available! #391
- New **TOMAS configuration** available! #391

IMAS interfacing:
~~~~~~~~~~~~~~~~~
- imas2tofu now provides tools for inspecting units from the dd_units tool
  Consequently it checks units when multi.calc_signal() is called to determine
  if a 4pi factor is necessary.
  Also, a user-provided corrective coefficient can be used.
  Also, time limts can be passed and events is usable as ids in tofuplot
  Also, events can be passed as time limits
  Also, a function provides a comparison between tofu and IMAS units #380

Documentation:
~~~~~~~~~~~~~~
- More accurate name, version number and copyright in doc #384

Miscellaneous:
~~~~~~~~~~~~~~
- Better PEP8 compliance


Contributors:
=============
Many thanks to all developpers and contributors:
- Didier Vezinet (@Didou09)
- Laura Mendoza (@lasofivec)
- Florian Le Bourdais  (@flothesof)
- Riccardo Ragona (@rragona)


What's next (indicative):
=========================
- Migrating from nosetests (ongoing for @lasofivec : issues #95 and #232)
- Availability via conda-forge (ongoing for @flothesof: issues )
- Easier binary and source installs from pip and conda for all platforms, including unit tests on all platforms (ongoing for @lasofivec and @flothesof : issue #259)
- Solid angles for Volume-Of-Sight and radiative heat loads computation (ongoing for @lasofivec : Issues #71, #72, #73, #74, #75, #76, #77, #78)
- Tools and classes to handle 2D Bragg X-Ray crystal spectrometer (ongoing for @Didou09 : Issues #202 and #263)
- Generic data class to incorporate plateau-finding, data analysis and 1d Bayesian fitting routines and classes (ongoing for @Didou09 and @jmoralesFusion and @MohammadKozeiha: issues #208, #260 and #262)
- More general magnetic field line tracing workflow
- Better unit tests coverage
- More complete documentation


List of PR merged into this release:
====================================
- PR: #378, #380, #384, #388, #390, #391, #393
