====================
What's new in 1.6.5
====================

tofu 1.6.5 is a significant upgrade from 1.6.4


Main changes:
=============

- Rays have more tools #712
- Cameras are fully integrated into Collections, inc. for inversions #714


Detailed changes:
=================

Rays:
--------------
- sample_rays(return_coords='ang_vs_phi') implemented # 712
- plot_diagnostic_interapolated_along_los(key_data_y='ang_vs_ephi', vmin=float, vmax=float) implemented #712

Cameras:
---------
- Cameras in Collections are now fully integrated up to inversions, including:  #714
    - compute_diagnostic_signal()
    - add_geometry_matrix()
    - add_inversion()
    - plot_inversion()

Dependencies:
-------------
- now uses datastock >= 0.0.22 #714


Contributors:
=============
Many thanks to all developpers and contributors
- Didier Vezinet (@Didou09)

What's next (indicative):
=========================
- Basic tools for inversions on triangular meshes
- Generic data class to incorporate plateau-finding, data analysis and 1d Bayesian fitting routines and classes (ongoing for @Didou09 and @jmoralesFusion and @MohammadKozeiha: issues #208, #260 and #262)
- More general magnetic field line tracing workflow
- Better unit tests coverage
- Better benchmarks
- More complete documentation
- Reference article

List of PR merged into this release:
====================================
- PR: #712, #714
