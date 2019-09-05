What's new in 1.4.1:

# Summary

## General
- All tofu object have overloaded __repr__ method
- units tests improved, Continuous Integration more complete, several bugs fixed
- More robust wndows pip installation
- More robust saving to .mat
- tofuplot and tofucalc now provide simple bash interfaces

## The geometry module is now more complete
- A Config class handles the tokamak geometry
- A CamLOS1D class handles arbitrary sets of LOS
- A CamLOS2D class handles 2D cameras approximated as a set of LOS
- Basic reflexions are taken into account
- Ray-tracing, LOS-campling and LOS-integration algorithls have been accelerated and parallelized
- Several utility function implemented for easy instance creation and useful geometrical computations
- The basics of a magnetic field-line tracing algorithm and plotting routine have been implemented

## IMAS-interfacing
- More robust MultiIDSLoader class for IMAS interfacing
- Can import from IMAS and export as tofu classes
- Very flexible and customizable
- Some classes have a save_to_imas() method


## Next to come
- Solid angles, solid angles, solid angles



# Detailed Changelog

## Enhancements

### Geometry


#### Algorithms to compute distance between LOS and a flux surfaces

- Compute distance between LOS and a circle (_+ vectorial version_)
- Test if a LOS and a circle are close to a epsilon (_+ vectorial version_)
- Compute distance between LOS and VPoly (_+ vectorial version_)
- Test if a LOS and a VPoly are close to a epsilon (_+ vectorial version_)
- Compute which LOS is closer to a VPoly
- Compute which VPoly is closer to a LOS

#### Algorithms for discretization and sampling
- Optimization of discretization of 1D lines (simple + complex version), 2D lines, 2D polygons
- Optimization of LOS sampling and all the different rules: parallelized and using less memory.

#### Calc signal (function integration over LOS):
- ``LOS_calc_signal`` now cythonized for isotropic and anisotropic emissivity and several sampling / integration strategies

#### Algorithms to check if visible
- Optimization `LOS_isVis_PtFromPts_VesStruct` checks if a point is visible with respect to a list of points and taking into account a vessel (with structures, limits, etc.) : function cythonized properly, speed up ~50%
- `LOS_areVis_PtsFromPts_VesStruct` equivalent to last function but input is a list of points
- `Dust_calc_SolidAngle` is now using new versions of above mentioned functions

#### Vignetting

- Added functions to compute the 3d bounding box of a polygon in 3d space
- Added functions for the triangulation of a polygon by earclipping
- Added function that tests if intersection between ray and triangle in space
- Added functions for vignetting : for a set of rays and a sets of polygons, test if the rays go through the polygons or not.

#### Structures and Configuration
- `tf.geom.Struct._get_phithetaproj()`: to get the (phi, theta) limits in a project of each toroidal occurence of Struct, with respect to a refpt in (R,Z)
- `tf.geom.Struct._get_phithetaproj_dist()`: + get the distance
- `tf.geom.dist._get_phithetaproj_dist()`: generalize to all Struct of a Config
- Config has set_colors_random()

#### Solid angles
- Added `_GG.Dust_calc_SolidAngle` for computation of a solid angle on dust particle (multi-pos, multi-pts, with and without vect, with and without approx, with and without block)

#### Basic Reflexions
* Rays subclasses can now have added reflections (self.add_reflections())
* Reflections can alos be extracted (instead of stored) as a list of separated cameras using Rays.get_reflections_as_cam()
* Reflections can be of type 'specular', 'diffusive' or 'ccube'
* The reflection type can be forced by the user (identical for all LOS of a camera) or automatically set by the Structural elements each LOS touches
* There can be any number of reflections
* Rays.plot() and Rays.plot_touch() have been updated to properly represent reflections
* Rays.calc_signal() and Rays.calc_signal_from_Plasma2D() have been updated to account for reflections with a user-provided coefficient

#### Cythonization
- Re wrote some numpy function to avoid over heading: hypothenus and composition of minimum of hypotenus, minimum of vector, tile,

#### Extra
- Erased *old* functions:
    - `SLOW_LOS_Calc_PInOut_VesStruct`
    - `Calc_LOS_PInOut_Lin`
    -  `Calc_LOS_PInOut_Tor`
- extract_svd() now compatible with Py27 and Py36
- Rays.dgeom['indout'][0,:] now refers to the global index of each structure (as in config.lStruct)
- Operator overloading of __\__repr__\__ of all tofu objects with attribute get_summary()

### Cameras and visualization

- overloaded __add__ operator to allow for easy CamLOS1D concatenation
- `tf.geom.dist.plot_phithetaproj_dist()`: plot a static figure with a place holder for time traces, the (phi,theta) projection (with aspect ratio), the cross and hor Config projections and legend
- `Rays.plot_touch()` now handles: angles (wrote docstring)

### General
- get_summary() is now a method of the AbstractToFuObject parent class, implemented for Config, CamLOS1D, Plasma2D
- saving to .mat file is now more robust (operational)
- Added benchmarking for ``calc_signal``

## IMAS interfacing

1. libraries that need imas2tofu are only imported if the **IMAS** module is found (Warning not Error)

2. **MultiIDSLoader** class now handles all imas interfacing:
- allows for loading several ids from different idd
- dynamic printing of ids loading
- get_summary() methods giving an overview of content
- provides a dictionary of shortcuts for fast typing of pre-recorded imas signals
- provides dictionary of preset ids and signals
- Flexible instanciation, from idd of idd args, ids or ids args, open or not, get or not
- provides add_idd(), remove_idd(), add_ids(), remove_ids()
- Include to_Config(), to_Plasma2D(), to_Diag() to export to tofu objects

3. **All stored in a single _core.py**:
- MultiIDSLoader
- load_Config(), load_Plasma2D(), load_Diag()

4. **save_to_imas()**
- Actual routines implemented in tf.imas2tofu._core
- methods implemented for Struct, Config, CamLOS1D

5. Other
* MultiIDSLoader now has calc_signal()
* ids = 'magfieldlines' returns mag field line tracking from IMAS equilibrium
* More robust / detailed sanity checks of triangular meshes from IMAS
* Synthetic data and experimental data are now time-synchronized
* Synthetic diag operational on WEST for interferometer, polarimeter, bremsstrahlung and bolometers

## bash interface:
* tofuplot debugged
* tofucalc created

## Code structure and format

### Restructuring of the geometry modules
Created Cython include files `*.pxd` and `*.pyx` for the following group of functions:
- `_basic_geom_tools.*`: global variables definition (`_VSMALL` and `_SMALL`), and basic geometric tools (vector calculus, path distance point-point, point-vector,...)
-  `_raytracing_tools.*`: for intersection of LOS and different objects (bounding boxes, poly, surfaces, ...)
- `_distance_tools.*`: distance between LOS and circle, VPoly and so on
- `_sampling_tools.*`: line, LOS, poly, volume, and surface sampling


## Installation and portability

- Had to use some cpp tools, so had to update the `setup.py`
- Updated the `setup.py` to be able to "clean" an installation (erase compilation files)
- Took out the possibility of not using cython, as this is now impossible
- git dependency is now optional (issue #67 )
- changes in `setup.py` for **Windows** portability
- Now only supporting ``Cython`` versions ``>=0.26``
- Removed all **pandas** dependencies
- Using ``-O3`` flag instead of ``-O0``  for faster execution time even if compilation is slower
- Removed all **Polygon** dependencies

## Unit test

- Added unit tests for triangulations, and vignetting
- Added unit test for computation of kmin, kmax ``LOS_Calc_kMinkMax_VesStruct``
- Update of `in _vessel` tests
- Added tests for `is_visble` (vectorized and point wise)
- Now by default ``python setup.py nosetests`` with run with: verbose, detailed erorrs, coverage (nose) and other utilities for debugging. But most importantly **only tests in ``tofu/tofu/tests/`` will be run**.
- Testing all get_sample options in a short new unit test
- Testing all options for LOS_calc_signal (method of discretization, of integration, steps relative, absolute, unique or changing for each los, etc.).

## Update of documentation

- Added documentation on how to install on Windows
- Updated, restructured and adde figures to README. Change of format `*.rst` to `*.md`
- Updated Wiki pages on GitHub

## Bugs

- Found several small bugs in the function that computes for a list of flux surfaces and a list of rays the kmin, kmax
- There was a bug in some special cases of 1D camera definition, when all Ds are on a similar plane sharing the same D for the 2 first LOS, solved
- There was a bug in tf.data._plot_combine() (wrong graph), solved (issue #65 )
- There was another bug with plot_combine() when several equilibria were provided, the reference time was not properly defined (short-term fix, on the long term issue #79 should fix it)
