.. role:: envvar(literal)
.. role:: command(literal)
.. role:: file(literal)
.. role:: ref(title-reference)


ToFu_Inv
========

(This project is not finalised yet, work in progress...)

**ToFu_Mesh**, is a ToFu module aimed at handling spatial discretisation of a 3D scalar field in a vacuum chamber (typically the isotropic emissivity of a plasma). Such discretisation is done using B-splines of any order relying on a user-defined rectangular mesh (possibily with variable grid size). It is particularly useful for tomographic inversions and fast synthetic diagnostics.

It is designed to be used jointly with the other **ToFu** modules, in particular with **ToFu_Geom** and **ToFu_MatComp**. It is a ToFu-specific discretisation library which remains quite simple and straightforward. However, its capacities are limited to rectangular mesh and it may ultimately be percieved as a much less powerful version of **PIGASUS/CAID**. Users who wish to use **ToFu** only for tomographic inversions may find **ToFu_Mesh** sufficient for thir needs, others, who wish to use a synthetic diagnostic approach, and/or to use **ToFu_Mesh** jointly with plasma physics codes (MHD...) may prefer using **PIGASUS\CAID** for spatial discreatisation.

Hence, **ToFu_Mesh** mainly provides two object classes : one representing the mesh, and the other one (which uses the latter) representing the basis functions used for discretisation: 

.. list-table:: The object classes in **ToFu_Geom**
   :widths: 10 30 20
   :header-rows: 1

   * - Name
     - Description
     - Inputs needed
   * - ID
     - An identity object that is used by all **ToFu** objects to store specific identity information (name, file name if the object is to be saved, names of other objects necessary for the object creation, date of creation, signal name, signal group, version...)
     - By default only a name (a character string) is necessary, A default file name is constructed (including the object class and date of creation), but every attribute can be modified and extra attribute can be added to suit the specific need of the the data acquisition system of each fusion experiment or the naming conventions of each laboratory.
   * - Mesh1D, Mesh2D, Mesh3D
     - 1D, 2D and 3D mesh objects, storing the knots and centers, as well as the correspondence between knots and centers in both ways. The higher dimension mesh objects are defined using lower dimension mesh objects. The Mesh 2D object includes an enveloppe polygon. They all include plotting methods and methods to select a subset of the total mesh. The Mesh 3D object is not finished.
     - A numpy array of knots, which can be defined using some of the functions detailed below (for easy creation of linearly spaced knots with chosen resolution). 
   * - BaseFun1D, BaseFunc2D, BaseFunc3D
     - 1D, 2D and 3D families of B-splines, relying on Mesh1D, Mesh2D, Mesh3D objects, with chosen degree and multiplicity for each dimension. Includes methods for plotting, for determining the support and knots and centers associated to each basis function, as well as for computing 1st, 2nd or 3rd order derivatives (as functions), and local value (summation of all basis functions or their derivatives at a given point and for given weights). Includes methods for computing integrals of derivative operators...
     - A Mesh object of the adapted dimension, and a degree value.
   

The following will give a more detailed description of each object and its attributes and methods through a tutorial at the end of which you should be able to create your own mesh and basis functions and access its main characteristics.

Getting started with **ToFu_Mesh**
----------------------------------
Once you have downloaded the whole **ToFu** package (and made sur you also have scipy, numpy and matplotlib, as well as a free polygon-handling library called Polygon which can be downloaded at ), just start a python interpreter and import **ToFu_Geom** (we will always import **ToFu** modules 'as' a short name to keep track of the functionalities of each module). To handle the local path of your computer, we will also import the small module called **ToFu_PathFile**, and **matplotlib** and **numpy** will also be useful:

.. literalinclude:: ../../src/Tutorial_example.py
   :language: python
   :lines: 7-12

The os module is used for exploring directories and the cPickle module for saving and loading objects.

The Tor object class
--------------------

To define the volume of the vacuum chamber, you need to know the (R,Z) coordinates of its reference polygon (in a poloidal cross-section). You should provide it as a (2,N) numpy array where N is the number of points defining the polygon. To give the Tor object its own identity you should at least choose a name (i.e.: a character string). For more elaborate identification, you can define an ID object and give as an input instead of a simple name. You can also provide the position of a "center" of the poloidal cross-section (in 2D (R,Z) coordinates as a (2,1) numpy array) that will be used to compute the coordinates in transformation space any LOS using this Tor object (and the sinogram of any scalar emissivity field using this Tor object). If not provided, the center of mass of the reference polygon is used as a default "center".

In the following, we will use the geometry of ASDEX Upgrade as a example. 
We first have to give a reference polygon ('PolyRef' below) as a (2,N) numpy array in (R,Z) coordinates. 

.. literalinclude:: ../../src/Tutorial_example.py
   :language: python
   :lines: 21-26


Alternatively, you can store PolyRef in a file and save this file locally, or use one of the default tokamak geometry stored on the **ToFu** database where Tor input polygons are stored in 2 lines .txt files (space-separated values of the R coordinates on the first line, and corresponding Z coordinates on the second line). Here, we use the default ASDEX Upgrade reference polygon stored in AUG_Tor.txt.

.. literalinclude:: ../../src/Tutorial_example.py
   :language: python
   :lines: 29-33

We now have created two Tor objects, and **ToFu_Geom** has computed a series of geometrical characteristics that will be useful later (or that simply provide general information). 
TO BE FINISHED !!!!!!!!!!!!!!!


.. math::

   \nabla^2 u = \sin(x)

.. Local Variables:
.. mode: rst
.. End:
