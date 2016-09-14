.. role:: envvar(literal)
.. role:: command(literal)
.. role:: file(literal)
.. role:: ref(title-reference)
.. _overview:

**Overview**
============

(This project is not finalised yet, work in progress...)


**ToFu**, which stands for "TOmography for FUsion" is a python package (with parts in C/C++) providing all necessary tools for tomography diagnostics for the Fusion community, it is particularly relevant for X-ray and bolometer diagnostics on Tokamaks. On of the objectoves is to provide a common tool for tomographic inversions, with both accurate methods and enough flexibility to be easily adapted to any Tokamak and to the specific requirements of each user. The main language (Python) has been chosen for its open-source philosophy, for its object-oriented capacities, and for the good performance / flexibility ratio that it offers. The architecture of the **ToFu** package is intended to be modular to allow again for maximum flexibility and to facilitate customisation and evolutivity from the users.

**ToFu**: provides in particular, but not only, the main following functionnalities :
  - Using the 3D geometry of the diagnostic (positions of detectors and apertures are provided as inputs) to compute quantities of interest (e.g.: the optimal line of sight, the exact etendue..). This is done by the module ToFu_Geom.
  - Building of a variable grid size mesh for spatial discretisation of the solution (i.e. emissivity field) on which B-splines of any degree can be added to serve as Local Basis Functions. This is done by the module ToFu_Mesh.
  - Computing of the geometry matrix associated to a set of detectors and a set of basis functions, both with a full 3D approach or with a Line Of Sight (LOS) approximation. This is done by the module ToFu_MatComp, which uses both ToFu_Geom and ToFu_Mesh.
  - Computing tomographic inversions based on the constructed geometry matrix and Phillips-Tikhonov inversion with a choice of objective functionals (among which first order and second order derivatives or Fisher information, and more to come). This is done by the module ToFu_Inv, which uses the matrix computed by ToFu_MatComp.
  - Visualizing, exploring and interpreting the resulting inversions using a built-in Graphic User Interface.

The joint use of a full 3D approach and of regular basis functions (B-splines) allows for advanced functionalities and flexibility, like in particular :
  - Accurate computation of etendue and geometry matrix.
  - Exact differential operators (provided sufficient degree of the basis function) instead of discretised operators (this feature and the previous one aim at improving the accuracy of tomographic inversions).
  - Accurate description of toroidal-viewing detectors with potentially large viewing cones and for which the LOS approximation cannot be used.
  - Making possible 3D inversions (provided the geometrical coverage of the plasma volume is sufficient, for example thanks to toroidal-viewing detectors).
  - Enabling proper taking into acccount of anisotropic radiation (for example due to fast electrons due to disruptions).

The **ToFu** package has built-in mesh and B-spline definitions, however, if used alone, it can only create and handle rectangular mesh (with variable grid size though). In order to allow for more optimised mesh and basis functions, the **ToFu** package is fully compatible with **Pigasus** (and **CAID**), which is a another Python package (with a Fortran core), which uses cutting-edge technologies from Computer-Assisted Design (CAD) to create optimised mesh (using Non-Unifrom
  Rational B-Splines, or NURBS, curves) on which it can also add several different types of regular basis functions. It is a next-gen solution for optimisation of plasma-physics simulation codes. Hence, the final idea is that the same mesh and tools can be used for running CPU-expensive plasma physics simulations and, from their output, to compute the associated simulated measurements on any radiation diagnostics. This synthetic diagnostic approach is aimed at facilitating direct
  comparisons between simulations and experimental measurements and at providing the community with flexible and cross-compatible tools to fit their needs. Plasma physics codes that are planning on using **Pigasus** in a near future include in particuler **JOREK** (in its **Django** version) and **GYSELA** (**SELALIB** in its next version). More information about **Pigasus** (lien), **JOREK** (lien) and **GYSELA** can be found on their respective pages.  

In order to avoid too much dependency issues, the **ToFu** package resorts to widely used Python libraries like scipy, numpy and matplotlib. Whenever it was possible, the idea was either to use a very common and accessible library or to have built-in methods doing the job. It can be run as a stand-alone on an offline computer (i.e.: on a laptop while travelling), in an online mode (using a central database on the internet) and with or without **Pigasus** (keeping in mind that only rectangular mesh can be created without it).

For faster computation, some modules and/or methods are coded with Cython or Boost.Pyton. It is also intended to be MPI and OpenMP parallelized.

The general architecture is briefly represented in the following figure:

.. figure:: /figures_doc/Fig_Tutor_BigPicture_General.png
   :height: 700px
   :width: 1000px
   :scale: 100 %
   :alt: ToFu big picture
   :align: center

   Modular architecture of ToFu, with its main modules.

This general overview shows all the **ToFu** modules and their main functionnalities and dependancies. Particularly important are the modules **ToFu_Geom**, **ToFu_Mesh** and **ToFu_MatComp** which provide all necessary tools to pre-calculate the geometry matrix which is a key feature of the two main uses of **ToFu**. 

On the one hand, **ToFu** can be used as a synthetic diagnostic since from a simulated emissivity field it can compute the corresponding synthetic measurements for comparison with experimental measurements. This, as illustrated below, can be done in different ways depending on whether the simualted is used directly as a function, projected on a predefined mesh of the plasma volume, or if the simulated emissivity itself was computed on a mesh using the **Pigasus/CAID** code suite which is directly compatible with **ToFu**. These three possibilities are illustrated in the following figure:

.. figure:: /figures_doc/Fig_Tutor_BigPicture_SynthDiag.png
   :height: 700px
   :width: 1000px
   :scale: 100 %
   :alt: ToFu big picture for synthetic diagnostics
   :align: center

   Modular architecture of ToFu, with its main modules for synthetic diagnostics.

On the other hand, **ToFu** can be used the other way around : use the experimental measurements to compute a reconstructed experimental emissivity field via a tomographic inversion, for comparisopn with a simulated emissivity field or simply for getting an idea of what the emissivity field looks like, which is illustrated in the following figure:

.. figure:: /figures_doc/Fig_Tutor_BigPicture_Tomo.png
   :height: 700px
   :width: 1000px
   :scale: 100 %
   :alt: ToFu big picture for tomography
   :align: center

   Modular architecture of ToFu, with its main modules for tomography.

The following will go into further details regarding each module.


ToDo list:
  - Rest of documentation, with relevant references (like :cite:Ingesson08FST) and figures
  - Tutorial
  - ToFu_Inv
  - GUI (one for each module)
  - Accelerate existing modules with Cython, Boost.Python + Parallelization
  - Use it to do some physics at last !!!


.. Local Variables:
.. mode: rst
.. End:
