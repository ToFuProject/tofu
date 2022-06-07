====================
What's new in 1.5.2
====================

tofu 1.5.2 is a minor upgrade from 1.5.1


Main changes:
=============

- compatibility with tomotok #636, #637
- first versions of thermal expansion for CrystalBragg #639
- HDF5 IMAS backend compatibility #642
- Better crystal bragg data treatement #644
- Minor refactoring #645
- New tokamaks geometries #647, #649
- Minor debugging #651


Detailed changes:
=================


Tokamak geometries:
~~~~~~~~~~~~~~~~~~~
- MAST: added a simple one-polygon version #647
- KSTAR: added a simple one-polygon version #649

New compatibility:
~~~~~~~~~~~~~~~~~~
- Users can now call inversion methods from tomotok (optional dependency) #636

CrystalBragg class:
~~~~~~~~~~~~~~~~~~~
- Cryst.compute_rockingcurve(therm_exp=True) enabled #639

Fit2d:
~~~~~~
- Extraction of raw_data enabled by fit2d_dinput(return_raw_data=...) #644

IMAS interfacing:
~~~~~~~~~~~~~~~~~
- Added ids spectrometer_x_ray_crystal #644
- Added compatibility with the new IMAS HDF5 backend #642

Unit tests:
~~~~~~~~~~~
- Added unit tests for tomotok #637

Miscellaneous:
~~~~~~~~~~~~~~
- Minor refactoring #645
- Minor debugging #651

Contributors:
=============
Many thanks to all developpers and contributors
- Didier Vezinet (@Didou09)
- Adrien Da Ros (@adriendaros)
- Kevin Obrejan (@obrejank)
- Florian Le Bourdais (@flothesof)

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
- PR: #636, #637, #639, #642, #644, #645, #647, #649, #651
