====================
What's new in 1.4.3
====================

Main changes:
=============

- Python 2 not supported anymore
- Several changes to try and make installation cleaner / easier
- More robust IMAS interface #280, #317, #319, #336, #343, #350
- Better documentation, more ressources
- More informative error messages
- Better PEP8 compliance

New features:
~~~~~~~~~~~~~
- New bash commands tofuplot, tofucalc and tofu-custom #345, #352, #353, #360
- New **AUG configuration** available! #333
- More complete CrystalBragg class #320


Detailed changes:
=================

Installation / portability:
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- travis test matrix #328
- Skipped deployment if package already detected, deploy bdist for Mac #293
- Bug corrected Cython vs cython #274
- pip install working, cleaned up setup.py, requirements.txt included #283 #299 #302
- More general path delimiters, corrected path in update_version #296 #315
- Bug fixed on Mac (due to wrong modulo operation) #323
- Debugged deployment and updating of version number #348


scripts:
~~~~~~~~
- Introduction of bash command line tools: tofuplot (for plotting IMAS data)
  tofucalc (for calculating synthetic signal) and tofu-custom for user-specific
  customization of tofu default parameters #345
- tofucalc also handles input_file and output_file in .mat format #352
- tofu-custom allows for cutomization of IMAS shortcuts #353
- tofu-custom allows for cutomization of script parameters #360


Exception / warnings:
~~~~~~~~~~~~~~~~~~~~~
- More details exception msg when loading wrong Config #327
- Better warning when a sub-package is not available #269 #270
- Warning added when gcc compiler < 8 used on Mac #268
- Better (more concise) warnings when unable to load data from IMAS #280
- Now warns if IMAS version loaded is not the latest available on the system #289
- Exception raised when poly ill-defined #332


Default configurations:
~~~~~~~~~~~~~~~~~~~~~~~
- Asdex Upgrade added (AUG) #333


Geom classes (Struct, Config, CamLOS1D, CamLOS2D...):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Reformatted _GG.Poly_Order by _GG.format_poly #338
- get_summary() routine implemented for Struct and CamLOS2D #305
- Faster synthetic signal cmputation (calc_signal()) using method='sum' #308 #322


Plasma2D:
~~~~~~~~~
- Bug fixed in interpolation routine #279
- Now handles rectangular meshes #280
- Can now save/load Config to/from IMAS and indch_auto propagated #325


2D XRay spectrometer:
~~~~~~~~~~~~~~~~~~~~~
- CrystalBragg ow has several geometry and data plotting methods useful for geometrical adjustments and spectrum visualization #320


IMAS interfacing:
~~~~~~~~~~~~~~~~~
- More robust MultiIDSLoader vs one-time-step signal #280
- Now loads rectangular meshes #280
- add_ids() now also works with idd handle #317
- It is now possible to add the ids corresponding to the synthetic diagnostics of an ids, and optionally to add them from a diffrent idd #319
- description_2d input argument propagated #336
- Automated selection of valiud channels (indch_auto=True) now more robust #343
- Updated signal for relectometer_profile #350


Documentation:
~~~~~~~~~~~~~~
- New tutorials and link to release notes #273 #297 #300 #330
- Better badges on Github #278
- Better release notes #284
- Documentation page on bash command lines


Miscellaneous:
~~~~~~~~~~~~~~
- Better PEP8 compliance #285
- Removed useless files #290 #324
- Minutes from meeting on 27.11.2019 added #291
- allow_pickle available to users #358


Contributors:
=============
Many thanks to all developpers:
- Didier Vezinet (@Didou09)
- Laura Mendoza (@lasofivec)
- Florian Le Bourdais (@flothesof)
- Jack Hare (@harej)


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
- PR: #268, #269, #270, #273, #274, #278, #279, #280, #282, #283, #284, #285,
  #289, #290, #291, #293, #296, #297, #299, #300, #302, #305, #308, #309, #315,
  #317, #319, #320, #322, #323, #324, #325, #327, #328, #330, #332, #333, #336,
  #338, #343, #345, #348, #352, #353, #358, #360
