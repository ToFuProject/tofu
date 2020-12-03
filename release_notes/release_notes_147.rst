====================
What's new in 1.4.7
====================

tofu 1.4.7 is a minor upgrade from 1.4.6
It is the first attempt at porting tofu to conda-forge

Main changes:
=============

- Better portability #398
- First attempt at deployment to conda-forge #410
- Better imas units #396
- Better documentation #399
- Better bash command #402, #404, #405
- New config: COMPASS #406
- Compatibility with SOLEDGE3X wall geometry #411
- Better PEP8 compliance


Detailed changes:
=================

Installation / portability:
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Added missing files in MANIFEST.in #398
- tests data is now included in sdist for conda-forge #410

scripts:
~~~~~~~~
- arg bck can now be used from tofucalc #402
- 'tofu XXX' can now be used instead of 'tofuXXX' #404
- Minor debugging #405

Geometry:
~~~~~~~~~
- New **COMPASS configuration** available! #406

IMAS interfacing:
~~~~~~~~~~~~~~~~~
- tofu now provides units harmonized with IMAS #396

SOLEDGE3X interfacing:
~~~~~~~~~~~~~~~~~~~~~~
- tofu can now read/write tokamak wall geometry from/to SOLEGDE3X format: #411
  - Config.to_SOLEGDE3X() to save a mat file
  - Config.from_SOLEDGE3X() to read from a mat file

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
- PR: #396, #398, #399, #402, #404, #405, #406, #410, #412
