====================
What's new in 1.4.11
====================

tofu 1.4.11 is a minor upgrade from 1.4.10


Main changes:
=============

- New Config.from_svg() methot to load tokamak geometry from Inkscape drawing!
- More fexible imas2tofu, openadas2tofu and nist2tofu interfaces!
- An efficient routine for computing the toroidally-integrated solid angle subtended by a spherical particle!
- New class for handling Spectral Lines! and loading routines from free online databases openadas and nist!
- Better compatibility with ITER linux clusters environment!
- Online documentation now has some video tutorials!

Detailed changes:
=================

Geometry:
~~~~~~~~~
- Added an optimised and parallelized routine for computing the toroidally integrated solid angle subtened by a spherical particle at multiple positions #491
- Added a Config.from_svg() method to load a tokamak geometry from an Inkscape drwaing saved as svg, including auto-scaling #511, #512, #515, #518, #524
- Config.from_txt() now more robust vs delimiters #516
- Already closed polygons do not raise a warning anymore #522

Generic classes:
~~~~~~~~~~~~~~~~
- A generic class for data handling is being implenmented, a first operational version is released and inherited for SpectralLines #499

Visualization:
~~~~~~~~~~~~~~
- Debugged figure interactivity #493

openadas interfacing:
~~~~~~~~~~~~~~~~~~~~~~
- Routine names are now more user-friendly, more documented and unit tests have been added #497
- Interface more flexible and robust, better unit tests, better documented #504
- The openadas interface was not working on ITER linux clusters due to wrong bundle of certificates, fixed #508

nist interfacing:
~~~~~~~~~~~~~~~~~
- Similar to openadas interface, with similar functionalities and a cache system #506
- The nist interface was not working on ITER linux clusters due to wrong bundle of certificates, fixed #510

IMAS interfacing:
~~~~~~~~~~~~~~~~~
- tofu now handles names with spaces and underscores (automaticlly removes them) #486
- keyword tokamak now replaced by databse for consistency with IMAS updated #507

Portability / CI:
~~~~~~~~~~~~~~~~~
- requests library dependency now properly managed #482
- Better handling of Github Actions for deployment #492

Documentation:
~~~~~~~~~~~~~~
- Minor updates and added 'Known bugs' section #484 #502
- Tutorial added for solid angle computation #501
- Tutorial (with video!) for creating a config from Inkscape #517

Miscellaneous:
~~~~~~~~~~~~~~
- Code refactoring #498


Contributors:
=============
Many thanks to all developpers and contributors:
- Didier Vezinet (@Didou09)
- Laura Mendoza (@lasofivec)
- Florian Le Bourdais  (@flothesof)

What's next (indicative):
=========================
- Solid angles for Volume-Of-Sight and radiative heat loads computation (ongoing for @lasofivec : Issues #74, #75, #76, #77, #78)
- Tools and classes to handle 2D Bragg X-Ray crystal spectrometer (ongoing for @Didou09 : Issues #202 and #263)
- Generic data class to incorporate plateau-finding, data analysis and 1d Bayesian fitting routines and classes (ongoing for @Didou09 and @jmoralesFusion and @MohammadKozeiha: issues #208, #260 and #262)
- More general magnetic field line tracing workflow
- Better unit tests coverage
- More complete documentation


List of PR merged into this release:
====================================
- PR: #482, #484, #486, #491, #492, #493, #497, #498, #499, #501, #502, #504, #506, #507, #508, #510, #511, #512, #515, #516, #517, #520, #523, #524
