====================
What's new in 1.7.0
====================

tofu 1.7.0 is a significant upgrade from 1.6.5


Main changes:
=============

- New dependency: bsplines2d, which concentrates all bsplines-related #723
- camera now has 'mode' (current vs PHA) #723
- Fixed a bug from Polygon3 that was causing noisy etendues #723
- plot_as_profile2d_compare() impemented #723
- get_diagnostic_lamb() implemented #723
- Inversion with regularization now available for 1d bsplines too #723
- Rocking curve setting and plotting implemented, and used for etendue of spectrometers #734, #735
- Debugged missing numpy in pyproject.toml #736
- compute_diagnostic_signal() for spectrometers #738
- VOS cross-section for non-spectrometers #742
- compute_diagnostic_signal(visibility=bool, timing=bool) implemented #743

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
- PR: #718, #723, #735, #736, #738, #742, #743
