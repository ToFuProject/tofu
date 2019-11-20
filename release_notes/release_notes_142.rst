in changes:
==========

- Python 2.7 is not supported anymore
- Python 3.6 and 3.7 are supported
- Several changes to try and make installation easier (on clusters, windows, mac....) and less verbose for users
- More explicit names for default saved configurations
- Major bug fix in one of the methods for computing synthetic signal
- Minor bug fixes in interactive figures
- Minor bug fixes in Plasma2D interpolation
- New configuration (ITER) available
- First tools for magnetic field line tracing available on WEST
- Better documentation, more ressources
- More informative error messages
- extra tools for computing LOS length, closest point to magnetic axis...
- Better PEP8 compliance


Detailed changes:
============

Installation / portability:
---------------------------
- Bug fixes for installation on ITER and Gateway clusters ()
- Easier installation on Mac
- Removed explicit compiler specification in setup.py for more flexibility
- When sub-packages imas2tofu or mag are missing, the warning is much more concise, but the error traceback is still accessible in a hidden dictionnary for developpers
-

Bug fixes:
-----------
- Major bug fixed in LOS_calc_signal() for computing the synthetic signal of a LOS camera using a particular algortihm : method='sum', minimize='hybrid', ani=True,
- Minor bug fixes in interactive figures when t=None was used (the interactivity was lost due to wrong formatting of the time array)
-

Documentation:
--------------
-

New features:
---------------
- First version of magnetic field line tracing (for WEST only so far, to be improved),
- First version of Spectro X 2D crystal for synthetic diagnostics (to be improved),
-


What's next (indicative):
=========================
- Migrating from nosetests (ongoing for @lasofivec : issues #95 and #232 )
- Easier binary and source installs from pip and conda for all platforms, including unit tests on alla platforms (ongoing for @lasofivec and @flothesof : issue #92 and #259 )
- Solid angles for Volume-Of-Sight and radiative heat loads computation (ongoing for @lasofivec : Issues #71, #72, #73, #74, #75, #76, #77, #78 )
- Tools and classes to handle 2D Bragg X-Ray crystal spectrometer (ongoing for @Didou09 : Issues #202 and #263)
- Generic data classe to incorporate plateau-finding, data analysis and 1d Bayesian fitting routines and classes (ongoing for @Didou09 and @jmoralesFusion and : issues #208, #260 and #262)
- More general magnetic field line tracing workflow
- Better unit tests coverage
- More complete documentation


List of issues and PR closed:
=============================
- Issues: #30, #180, #183, #185, #186, #187, #188, #189, #190, #201, #209, #211, #213, #217, #220, #224, #227, #228, #230, #235, #243, #247, #248, #250, #252, #255
- PR: #173, #175, #179, #181, #182, #184, #191, #192, #193, #194, #195, #196, #197, #199, #206, #207, #210, #212, #222, #223, #225, #226, #229, #231, #233, #234, #236, #237, #238, #240, #242, #244, #245, #246, #249, #251, #253, #254, #256, #257, #258
