====================
What's new in 1.6.4
====================

tofu 1.6.4 is a significant upgrade from 1.6.3


Main changes:
=============

- Convex crystals are now handled + several minor bug fixes #691
- Rays: better sampling + segments handling + tangency radius #691, #694
- More versatile diagnostic subclass, handles and plots multiple 1d cameras #694
- diagnostics cameras can now be moved / set as pinhole cameras #701
- datastock >= 0.0.21 is now needed to handle astropy units #701
- solid angle can be computed for any points for any diagnostic (non-spectro) #705


Detailed changes:
=================

Spectral fits:
--------------
- minor bug fix on the line-doubling parameter scale initialization (dshift)    #699

Crystals:
---------
- Several minor bug fixes       #691
- Convex crystals implemented using negative curvature for cylindrical and spherical cases      #691

Rays:
-----
- Rays can be sampled in 'rel' or 'abs' mode    #691
- Much more control on rays sampling (by segments, by major radius)     #694
- Rays sampling can now return multiple coordinates     #694
- can now be removed    #701

Cameras:
--------
- can now create 1d and 2d pinhole cameras      #701

Diagnostics:
------------
- Now handle multiple cameras using doptics (dict) instead of optics (list)     #694
- Can now plot multiple 1d cameras (not 2d yet) #694
- Can now plot tangency radius as data  #694
- Minor bug fixed in plots, inherited from datastock <= 0.0.20  #701
- can plot any interpolated quantity along its lines of sight   #694
- can now compute the solid angle subtended by each pixel of each camera through its apertures for any points   #705
- can now move selected cameras #701
- can now be removed            #701

Dependencies:
-------------
- now uses datastock >= 0.0.21 #701


Contributors:
=============
Many thanks to all developpers and contributors
- Didier Vezinet (@Didou09)
- Adrien Da Ros (@adriendaros)
- Tullio Barbui (@tbarbui)
- Conor Perks (@cjperks7)

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
- PR: #691, #694, #699, #701, #706
