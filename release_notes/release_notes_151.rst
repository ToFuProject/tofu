====================
What's new in 1.5.0
====================

tofu 1.5.1 is a minor upgrade from 1.5.0


Main changes:
=============

- Better packaging and CI #583, #584, #585, #587, #590
- Better benchmarks #589
- Improved and new features for CrystalBragg class #591, #601, #611, #616
- Figures interactivity recovered #594
- New/improved tokamaks geometries: SPARC, AUG and NSTX #598, #606, #615
- Better doc #609

N.B: issue #603 was opened and designed as a tutorial for inversions


Detailed changes:
=================


Tokamak geometries:
~~~~~~~~~~~~~~~~~~~
- AUG: added a simple one-polygon version #606
- SPARC: added 2 versions #598
- NSTX: added a very detailed version #615

CrystalBragg class:
~~~~~~~~~~~~~~~~~~~
- Cryst.get_plasmadomain_at_lamb() which plots the plasma domain from which a wavelength is accessible to a detector #591
- Cryst.calc_signal_from_emissivity() which plots the synthetic signal from a user-provided plasma emissivity map and spectrum #601
- tofu.spectro has new routines for 2d spectra fitting and plotting (fit2d(), plot_dinput2d(), plot_fit2d()) #611
- Cryst.plot() can now take any pts for ray-tracing, not only from the plasma (also from the detector) #616

CI, packaging and doc:
~~~~~~~~~~~~~~~~~~~~~~
- svg.path is a new dependency #583
- Support for python 3.9 (python 3.6 removed) #584
- scikit-sparse and scikit-umfpack are optional dependencies #585
- Removed temporary latex files from notes #587
- C++ dependency removed to avoid a bug (and to avoid a dependency) #590

Benchmarks:
~~~~~~~~~~~
- Benchmark suite has been completed with structure sampling and 1d spectral fitting #589

Miscellaneous:
~~~~~~~~~~~~~~
- Figures interactivity had been lost, it is recovered #594
- Documentation is updated #609

Contributors:
=============
Many thanks to all developpers and contributors
- Didier Vezinet (@Didou09)
- Laura Mendoza (@lasofivec)
- Oulfa Chellai ()
- Florian Le Bourdais (@flothesof)
- Adrien Da Ros (@adriendaros)
- Kevin Obrejan (@obrejank)

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
- PR: #583, #584, #585, #587, #589, #590, #591, #594, #598, #601, #606, #609, #611, #615, #616
