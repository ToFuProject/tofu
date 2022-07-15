====================
What's new in 1.6.0
====================

tofu 1.6.0 is a major upgrade from 1.5.2
It introduces a strong datastock dependency, used for Plasma2D, also handling geometry matrices and inversions


Main changes:
=============

- datastock (>= 0.0.17) is used for Plasma2D and SpectralLines classes inheritance #632
- CrystaBragg now handles ray-tracing from rocking curves #655

Detailed changes:
=================

CrystalBragg class:
~~~~~~~~~~~~~~~~~~~
- Cryst.compute_rockingcurve(therm_exp=True) enabled #639

Plasma2D class:
~~~~~~~~~~~~~~~
- Plasma2D is now inherited from datastock.DataStock class #655
  It comes with:
      - Built-in specialized interactive plotting of all data
      - Sub-class mesh, that can be of type 'rect', 'tri' or 'polar'
      - Sub-class bsplines
      - Interactive plotting of bsplines and profile2d (and interpolations)
      - add_geometry_matrix() method (with user-provided camera, and plotting)
      - add_inversion() method, with many algorithms and interactive plotting, for 'rect' and 'polar' meshes

SpectralLines class:
~~~~~~~~~~~~~~~~~~~~
- SpectralLines is now inherited from datastock.DataStock class #655
  It comes with:
      - calc_pec() method 
      - calc_intensity() method
      - plot_spectral_lines() method
      - plot_pec_single() method

Contributors:
=============
Many thanks to all developpers and contributors
- Didier Vezinet (@Didou09)
- Adrien Da Ros (@adriendaros)

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
- PR: #632, #655
