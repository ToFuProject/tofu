====================
What's new in 1.4.9
====================

tofu 1.4.9 is a minor upgrade from 1.4.8
It fixes a bug affecting Windows conda packages only.


Main changes:
=============

- Fixed bug on Windows distributions #432
- Group all imas2tofu default parameters in imas2tofu/_def.py #434

Detailed changes:
=================

Community:
~~~~~~~~~~
-

scripts:
~~~~~~~~
-

Geometry:
~~~~~~~~~
-

IMAS interfacing:
~~~~~~~~~~~~~~~~~
-


Geometry:
~~~~~~~~~
-


Documentation:
~~~~~~~~~~~~~~
-

Miscellaneous:
~~~~~~~~~~~~~~
-


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
- PR: #432, #434
