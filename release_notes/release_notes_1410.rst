====================
What's new in 1.4.10
====================

tofu 1.4.10 is a minor upgrade from 1.4.9


Main changes:
=============

- More detailed tokamak geometries (ITER, WEST, TCV)
- More user-friendly API for some functions
- 3 minor bugs fixed
- Updated CI (nosetests -> pytest and Github Actions)
- Better code, documentation

Detailed changes:
=================

scripts:
~~~~~~~~
- Keyword arg t0 is now properly handled by tofuplot and tofucalc #451

Geometry:
~~~~~~~~~
- ITER: more detailed geometry, with inner and outer vessel, poloidal field coils, and cryostat #460
- WEST: more detailed geometry, with inner and outer vessel, HFS and LFS thermal shield, all parts of the divertor coils casing #462
- General: the name of the tokamak followed by '-coils' now also returns poloidal field coils (when available) #464
- General: new, more accessible, function tf.load_config() (alias for tf.geom.utils.create_Config()) #464
- TCV: inner and outer vessel are now separated, avoiding a bug in collision-detection #468

Computation:
~~~~~~~~~~~~
- Debugged a cython / numpy issue #461
- Fixed wrong numbering of structures in Rays._prepare_inputs_kInOut() for Rays_plot_touch() #466
- Implemented: computation of solid angles subtended by a spherical particle from point in the plasma #470
- Fixed: wrong definition of input k (renamed dist) in _GG.LOS_areVis_PtsFromPts_VesStruct() #472

Documentation:
~~~~~~~~~~~~~~
- Minor updates #436, #469

Unit tests and CI:
~~~~~~~~~~~~~~~~~~
- Switched from nosetests to pytest #445
- Travis: lighter tests (only 2 jobs per run except for devel and master branches) #455
- Github Actions implemented: light tests on all branches except devel and master #455
- Removed a few warnings #476

Miscellaneous:
~~~~~~~~~~~~~~
- tofu does not force matplotlib's backend to agg anymore #448
- Removed notebooks that were counted as code by github #449
- Code review (lintering) by flake8 to clean-up messy parts of code (orphan or undefined variables...) #456
- Data.plot() now properly takes into account arguments vmin and vmax #457
- More explicit input arguments for tofu.geom.utils.create_camLOS1D() and tofu.geom.utils.create_camLOS2D() #474

Contributors:
=============
Many thanks to all developpers and contributors:
- Didier Vezinet (@Didou09)
- Laura Mendoza (@lasofivec)
- Florian Le Bourdais  (@flothesof)

What's next (indicative):
=========================
- Solid angles for Volume-Of-Sight and radiative heat loads computation (ongoing for @lasofivec : Issues #73, #74, #75, #76, #77, #78)
- Tools and classes to handle 2D Bragg X-Ray crystal spectrometer (ongoing for @Didou09 : Issues #202 and #263)
- Generic data class to incorporate plateau-finding, data analysis and 1d Bayesian fitting routines and classes (ongoing for @Didou09 and @jmoralesFusion and @MohammadKozeiha: issues #208, #260 and #262)
- More general magnetic field line tracing workflow
- Better unit tests coverage
- More complete documentation


List of PR merged into this release:
====================================
- PR: #436, #445, #448, #449, #451, #455, #456, #457, #460, #460, #462, #464, #466, #468, #469, #470, #472, #474, #476
