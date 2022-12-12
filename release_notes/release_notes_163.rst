====================
What's new in 1.6.3
====================

tofu 1.6.3 is a significant upgrade from 1.6.2


Main changes:
=============

- Full revamp of triangulation, solid angle and etendue computation #661
- Full revamp of apertures, cameras, diagnostics now all in the same Datastock-inherited class # 664
- Interactive diagnostic plotting #667
- Better rocking curve units and dict #669
- Crystal spectrometers now handled #670
- Minor renaming, debugging, minor features #674 #675 #683
- Re-implementation of rays #676
- Etendue for spectrometers + many spectrometer features #681
- Now using contourpy #686


Detailed changes:
=================

Contributors:
=============
Many thanks to all developpers and contributors
- Didier Vezinet (@Didou09)
- Adrien Da Ros (@adriendaros)

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
- PR: #661, #664, #667, #669, #670, #674, #675, #676, #681, #686, #683
