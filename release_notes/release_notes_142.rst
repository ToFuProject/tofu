Main changes:
==========

- Python 2.7 is not supported anymore
- Python 3.6 and 3.7 are supported
- Several changes to try and make installation easier (on clusters, windows, mac....) and less verbose for users
- More explicit names for default saved configurations
- Major bug fix in one of the methods for computing synthetic signal
- Minor bug fixes in interactive figures
- Minor bug fixes in Plasma2D interpolation
- New configuration (ITER) available
- First version of a class handling 2D XRay bragg spectrometers
- First tools for magnetic field line tracing available on WEST
- Better documentation, more ressources
- More informative error messages
- extra tools for computing LOS length, closest point to magnetic axis...
- Better PEP8 compliance


Detailed changes:
============

Installation / portability:
---------------------------
- Bug fixes for installation on ITER and Gateway clusters #250
- Easier installation on Mac #183 (`requirements.txt`)
- Removed explicit compiler specification in `setup.py` for more flexibility
- When sub-packages imas2tofu or mag are missing, the warning is much more concise, but the error traceback is still accessible in a hidden dictionnary for developpers #243
- `python setup.py clean` now doesn't cythonize
- Dropped python 2 support: #228
  - merged `_GG02` and `_GG03` into the `_GG` file
  - no longer being tested in travis
  - no longer packaged in conda
  - updated README file accordingly
  - dropped `funcsigs` dependency
  - made necessary changes in `setup.py`
  - `benchmarks/calc_signal_benchmark.py`: now working with python 3

Bug fixes:
-----------
- Major bug fix in `LOS_calc_signal()` for computing the synthetic signal of a LOS camera using a particular algortihm : method='sum', minimize='hybrid', ani=True,  #247 #255
- Major bug fix in `LOS_get_sample()` when `minimize='hybrid'` and `minimize='memory'` the limits were not set correctly
  in some cases the formula for sampling a LOS was wrong (`los_get_sample_core_var_res`).
- Minor bug fixes in interactive figures when `t=None` was used (the interactivity was lost due to wrong formatting of the time array)
- Minor bug fixed in Plasma 2D interpolation (`interp_t` was not being set), imporved error messages #209
- Removed unused variable in `_Ves_get_sampleS` (_GG), in `_core.py` #190
- Gave more explicit names to some variables in `_core.py` to avoid bugs/typos (eg. `I` to `current`) #186
- Removed a secondary `init` function for the class `tf.geom.CoilsPF` #187
- `_checkformat_inputs_dgeom` now is a function of `Rays` class
- Change default separator in `to_dict()` from '_' to '.', #228
- Rays.calc_kInOut_IsoFlux() is now more flexible (accepts 3d np.arrays as well as lists of np.ndarrays), #188
- __repr__() is now overloaded in a more robust way so it falls back to printing the class if get_summary() fails to avoid crashing in debug mode #242

Documentation:
--------------
- Updated information about support of python version
- Added slides of talk given at PyConFR 2019 conference
- Added a `gallery` in our documentation with 3 different tutorials:
  - 5 minutes tutorial to show to create a geometry and 1D/2D cameras
  - Guide on how to create your own Geometry from scratch (vacuum vessel, structures, etc.)
  - How to compute the signal received by a camera using a synthetic signal.
  For all of these tutorial, you can see directly the codes and the
  resulting images, and you can get the source code or download it as a
  Jupyter notebook!
- Minor changes to the web doc: updated install instructions to be "cleaner"
  now in rST and not HTML), small changes in navigation bar.
- Guide on how to contribute to ToFu.

New features:
---------------
- First version of `magnetic field line tracing` (for WEST only so far, to be improved) #213 #235
- First version of `2D XRay Bragg spectrometers` for synthetic diagnostics, modelling the diagnotics geometry and providing 2d spectral fitting routines (to be improved) #267
- When computing a signal `LOS_calc_signal` emissivity function can now return
  a 1D array if `t=None` #217 #252
- Three functions added to `tf.geom.Rays`: #211
	- `calc_length_in_isoflux()`: compute the length inside a set of isoflux surfaces of each LOS
	- `calc_min_geom_radius()`: compute the minimal geometrical radius (impact parameter) of each LOS
	- `calc_min_rho_from_Plasma2D()`: compute the minimum normalized radius (or any field with a minimum on the axis) for each LOS
- New `ITER configuration` available! #220 #227
- More explicit names for default configurations #264

Contributors:
=============

Many thanks to all developpers:

- Didier Vezinet (@Didou09)
- Laura Mendoza (@lasofivec)
- Florian Le Bourdais (@flothesof)
- Jorge Morales (@jmoralesFusion)
- Koyo Munechika (@munechika-koyo)
- Louwrens Van Dellen (@Louwrensth)


What's next (indicative):
=========================
- Migrating from nosetests (ongoing for @lasofivec : issues #95 and #232 )
- Easier binary and source installs from pip and conda for all platforms, including unit tests on alla platforms (ongoing for @lasofivec and @flothesof : issue #92 and #259 )
- Solid angles for Volume-Of-Sight and radiative heat loads computation (ongoing for @lasofivec : Issues #71, #72, #73, #74, #75, #76, #77, #78 )
- Tools and classes to handle 2D Bragg X-Ray crystal spectrometer (ongoing for @Didou09 : Issues #202 and #263)
- Generic data classe to incorporate plateau-finding, data analysis and 1d Bayesian fitting routines and classes (ongoing for @Didou09 and @jmoralesFusion and @MohammadKozeiha: issues #208, #260 and #262)
- More general magnetic field line tracing workflow
- Better unit tests coverage
- More complete documentation


List of issues and PR closed:
=============================
- Issues: #30, #180, #183, #185, #186, #187, #188, #189, #190, #201, #209, #211, #213, #217, #220, #224, #227, #228, #230, #235, #243, #247, #248, #250, #252, #255, #264
- PR: #173, #175, #179, #181, #182, #184, #191, #192, #193, #194, #195, #196, #197, #199, #206, #207, #210, #212, #222, #223, #225, #226, #229, #231, #233, #234, #236, #237, #238, #240, #242, #244, #245, #246, #249, #251, #253, #254, #256, #257, #258,
  #261, #265, #267
