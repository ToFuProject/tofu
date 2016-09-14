.. role:: envvar(literal)
.. role:: command(literal)
.. role:: file(literal)
.. role:: ref(title-reference)


**ToFu_Mesh**
=============

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

Getting started with ToFu_Mesh
------------------------------

Once you have downloaded the whole **ToFu** package (and made sur you also have scipy, numpy and matplotlib, as well as a free polygon-handling library called Polygon which can be downloaded at ), just start a python interpreter and import **ToFu_Geom** and **ToFu_Mesh** (we will always import **ToFu** modules 'as' a short name to keep track of the functionalities of each module). To handle the local path of your computer, we will also import the small module called **ToFu_PathFile**, and **matplotlib** and **numpy** will also be useful:

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 7-14

The os module is used for exploring directories and the cPickle module for saving and loading objects.

The Mesh1D, Mesh2D and Mesh3D object classes
--------------------------------------------

In this section, we describe the Mesh objects starting from the unidimensional to the 3D version.

.. list-table:: The attributes of a Mesh1D object
   :widths: 10 40
   :header-rows: 1

   * - Attribute
     - Description
   * - self.ID
     - The ID class of the object
   * - self.NCents, self.NKnots
     - The number of mesh elements or centers (resp. knots) of the object (typically self.NKnots = self.NCents+1)
   * - self.Cents, self.Knots
     - The coordinates of the centers and knots themselves, as two numpy arrays
   * - self.Lengths, self.Length, self.BaryL, self.BaryP
     - The length of each mesh element, the total length of the mesh and the center of mass of the mesh (i.e.: weight by the respective length of each mesh element), and the barycenter of the self.Cents
   * - self.Cents_Knotsind, self.Knots_Centsind
     - The index arrays used to get the correspondence between each mesh element (resp, each knot) and its associated knots (resp. its associated mesh elements)

.. list-table:: The attributes of a Mesh2D object
   :widths: 10 40
   :header-rows: 1

   * - Attribute
     - Description
   * - self.ID
     - The ID class of the object
   * - self.MeshR, self.MeshZ
     - The two Mesh1D objects used to create this Mesh2D object
   * - self.NCents, self.NKnots
     - The number of mesh elements or centers (resp. knots) of the object (typically self.NKnots = self.NCents+1)
   * - self.Cents, self.Knots
     - The coordinates of the centers and knots themselves, as two numpy arrays
   * - self.Surfs, self.Surf, self.VolAngs, self.VolAng, self.BaryV, self.BaryS, self.BaryL, self.BaryP
     - The surface of each mesh element, the total surface of the mesh, the volume per unit angle of each mesh element, the total volume per unit angle, the volume barycenter of the mesh (i.e. taking into account not only the surface repartition but also the toroidal geometry), the center of mass of the mesh (i.e.: weight by the respective surface of each mesh element), the middle point (the average between the extreme (R,Z) coordinates) and the barycenter of all the self.Cents
   * - self.Cents_Knotsind, self.Knots_Centsind
     - The index arrays used to get the correspondence between each mesh element (resp, each knot) and its associated knots (resp. its associated mesh elements)
   * - self.BoundPoly
     - The boundary polygon of the mesh, useful for fast estimation whether a point lies inside the mesh support or not.

In an experiment-oriented perspective, **ToFu_Mesh** comes with simple functions to help you quickly define an optimal 1D grid, with explicit parametrisation of the spatial resolution on regions of interest.
For example, if you want to define a 1D grid with a 5 cm resolution near the first end, that gradually refines to 1 cm at a given point, stays 1 cm for a given length and is then gradually enlarged to 6 cm at the other end, you just have to feed in the points of interest and their associated resolution to the *LinMesh_List* function, as a two lists of corresponding (start,end) tuples.

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 20-22

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 24-25

You can then feed the resulting knots numpy array to the Mesh1D object class and use this object methods to access all the features of interest of the created mesh:

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 28-30,33

.. figure:: figures_doc/Fig_Tutor_ToFuMesh_M1.png
   :height: 200px
   :width: 500 px
   :scale: 100 %
   :alt: Arbitrary 1D mesh with customized resolution in chosen regions
   :align: center
   
   Arbitrary 1D mesh with customized resolution in chosen regions

.. figure:: figures_doc/Fig_Tutor_ToFuMesh_M1_Res.png
   :height: 250px
   :width: 500 px
   :scale: 100 %
   :alt: Local spatial resolution of the created 1D mesh
   :align: center
   
   Local spatial resolution of the created 1D mesh

It can seen that the algorithm tried to render a mesh with the required resolution, even though it had to decrease it slightly around the first point, where it is lower than the required 6 cm (this is necessary due to the necessity to the number of mesh elements which must be an integer, thus leading to rounding). This is shown also in the *Res* variable which returns the actual resolution.
Like for the **ToFu_Geom** plotting routines, the 'Elt' keyword argument provides you with the possibility of choosing what is going to be plotted (the knots 'K', the centers 'C' and/or the numbers 'N').

The Mesh2D object class relies on the same basics, except that its multi-dimensional nature means that it has extra methods for easy handling of mesh elements. Let us for example create a coarse 2D mesh using 2 different 1D mesh objects:

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 37-42,44

.. figure:: figures_doc/Fig_Tutor_ToFuMesh_M2_Raw.png
   :height: 400px
   :width: 400 px
   :scale: 100 %
   :alt: Arbitrary 2D mesh with customized resolution in chosen regions
   :align: center
   
   Arbitrary 2D mesh with customized resolution in chosen regions


The Mesh2D class comes with a method to automatically create another Mesh2D object that can be seen as a sub-mesh (only the elements lying inside an input polygon are kept, the rest being memorized only as 'Background'). In our example, we can use a specific method of the TFG.Tor object class to create a smooth convex polygon lying inside the Tor enveloppe (see the kwdargs for customization of the smoothing and offset) to concentrate on the region where most SXR radiation comes from:

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 47-51,54

.. figure:: figures_doc/Fig_Tutor_ToFuMesh_M2.png
   :height: 500px
   :width: 400 px
   :scale: 100 %
   :alt: Submesh of the 2D mesh with customized resolution in chosen regions with selected elements only (using an input polygon)
   :align: center
   
   Submesh of the 2D mesh with customized resolution in chosen regions with selected elements only (using an input polygon)

.. figure:: figures_doc/Fig_Tutor_ToFuMesh_M2_Res.png
   :height: 500px
   :width: 500 px
   :scale: 100 %
   :alt: Local spatial resolution of the created 2D mesh (both linear and surface)
   :align: center
   
   Local spatial resolution of the created 2D mesh (both linear and surface)

Here, the 'NLim' kwdarg is used to specifiy how many corners of a mesh element must lie inside the input polygon so that this mesh element can be counted in.

Now, the Mesh2D object class provides tools to easily select and plot chosen elements of the 2D mesh. For example, if you want to get the coordinates of the four knots associated to the mesh element number 50, you can use the attribute 'Centers_Knotsind' to get them, and then plot them:

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 57-59,63

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 60-61

.. figure:: figures_doc/Fig_Tutor_ToFuMesh_M2_Cents.png
   :height: 400px
   :width: 400 px
   :scale: 100 %
   :alt: Selected mesh element and its associated knots
   :align: center
   
   Selected mesh element and its associated Knots

Similarly, you can get and plot all the mesh element centers associated to knots number 160, 655 and 1000:

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 65-68,72

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 69-70

.. figure:: figures_doc/Fig_Tutor_ToFuMesh_M2_Knots.png
   :height: 400px
   :width: 400 px
   :scale: 100 %
   :alt: Selected mesh knots and their associated mesh elements
   :align: center
   
   Selected mesh knots and their associated mesh elements

The Mesh3D object class is currently being built... to be finished.

Now that we have access to a mesh, we can build basis functions on it. The basis functions available in **ToFu_Mesh** are all B-splines, as illustrated below.


The BaseFunc1D, BaseFunc2D and BaseFunc3D object classes
--------------------------------------------------------

The use of B-spline allows for more flexibility and more accuracy than the standard pixels (which are B-splines of degree 0). Indeed, most of the tomographic algorithms using series expansion in physical space assess the regularity of the solution by computing the integral of a norm of one of its derivatives. While the use of pixels forces you to use discrete approximations of the derivative operators, the use of B-splines of sufficient degree allows to use an exact formulation of the derivative operators.

The attributes of a BaseFunc1D objects are the following:

.. list-table:: The attributes of a BaseFunc1D object
   :widths: 10 40
   :header-rows: 1

   * - Attribute
     - Description
   * - self.ID
     - The ID class of the object
   * - self.Mesh
     - The Mesh1D object on which the basis functions are built
   * - self.LFunc, self.NFunc, self.Deg, self.Bound
     - The list of basis functions and the number of basis functions (self.NFunc=len(self.LFunc)), the degree of the basis functions, and the boundary condition (only 0 implemented so far, all points have 1 multiplicity)
   * - self.Func_Centsind, self.Func_Knotsind, self.Func_PMax
     - An array giving the correspondence index between each basis function and all its associated mesh centers (and there are methods to go the other way around), its associated mesh knots, and the position of the maximum of each basis function (either oa mesh center or a knot depending on its degree).

Other quantities, indices or functions of interest are not stored as attributes, but instead accessible through methods, as will be illustrated in the following:

One of the most common issues in SXR tomography on Tokamaks is the boundary constraint that one must enforce at the plasma edge to force the SXR emissivity field to smoothlty decrease to zero in order to avoid artefacts on the tomographic reconstructions. With pixels, this usually has to be done by adding artificial detectors that 'see' the edge pixels only and are associated to a 'measured' value of zero (and the regularisation process does the rest). With B-splines of degree 2 for example, this constraint can be built-in the basis functions and enforced without having to add any artificial constraint, provided the underlying mesh is created accordingly, as illustrated in the following example, where BaseFunc1D object of degree 2 is created and a method is used to fit its coefficients to an input gaussian-like function:

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 76-80,82

.. figure:: figures_doc/Fig_Tutor_ToFuMesh_BF1.png
   :height: 300px
   :width: 500 px
   :scale: 100 %
   :alt: 1D B-splines of a BaseFunc1D object, with arbitrary coefficients to create a gaussian-like profile
   :align: center
   
   1D B-splines of a BaseFunc1D object, with arbitrary coefficients to create a gaussian-like profile

By construction, and because we have only used points with multiplicity equal to one so far, the profile can only decrease smoothly to zero near the edge.

The BaseFunc1D object also comes with methods to compute and plot local values its derivatives, or of some operators of interest that rely on derivatives. In particular, the following example shows the plots of the first derivative, the second derivative and a quantity called the Fisher Information that is the first derivative squared and divided by the function value. As usual, the 'Elt' kwdarg is used to specify whether we want only the total function ('T') or the detail of the list of all the underlying B-splines ('L', which is not possible for non-linear operators):

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 84-86,88

.. figure:: figures_doc/Fig_Tutor_ToFuMesh_BF1_Deriv.png
   :height: 300px
   :width: 500 px
   :scale: 100 %
   :alt: Some quantities of interest, based on derivative operators, for the chosen BaseFunc1D object
   :align: center
   
   Some quantities of interest, based on derivative operators, for the chosen BaseFunc1D object

This was done using the 'Deriv' kwdarg, which can take several values, as shown in the table below:

.. list-table:: The available values of the 'Deriv' keyword argument for a BaseFunc1D object
   :widths: 10 40
   :header-rows: 1

   * - Value
     - Description
   * - 0, 1, 2, 3 or 'D0', 'D1', 'D2', 'D3'
     - Respectively the B-splines themselves (0-th order derivative), the first, second and third order derivative
   * - 'D0N2', 'D1N2', 'D2N2', 'D3N2'
     - The squared norm of the 0th, 1st, 2nd and 3rd order derivatives
   * - 'D1FI'
     - The Fisher Information, which is the squared norm of the 1st order derivative, divided by the function value

Keep in mind that we are only using exact derivatives here, so the current version of **ToFu_Mesh** does not provide discretised operators and you have to make sure that you only compute derivatives for B-splines of sufficiently high degree.

Finally, the BaseFunc1D object also comes with methods to compute the value of the integral of the previous operators on the support of the B-spline. When it is possible, another method also returns the matrix that can be used to compute this integral using a vector of coefficients for the B-splines, along with a flag 'm' that indicates how the matrix should be used:

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 91-94

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 95-96

When m==0, it means that A is a vector (Int=A*Coefs), and when m==1, it means A is matrix and the integral requires a square operation (Int=Coefs*A*Coefs).
The following integrals are implemented:

.. list-table:: The available values of the 'Deriv' keyword argument for integral computation
   :widths: 10 40
   :header-rows: 1

   * - Value
     - Description
   * - 0 or 'D0'
     - The integrals of the B-splines themselves (0-th order derivative, integrals of higher order derivatives are all zero)
   * - 'D0N2', 'D1N2', 'D2N2', 'D3N2'
     - The integrals of the squared norm of the 0th, 1st, 2nd and 3rd order derivatives (only 0-th order derivative implemented so far, for Deg=0,1, but not for Deg=2,3)
   * - 'D1FI'
     - The integrated Fisher Information, not implemented so far

Finally, you can also plot a series of selected basis functions and there associated mesh elements (useful for detailed analysis and for debugging). Note tht you can also provide a 'Coefs' vector if you do not wish to use the default Coefs=1. value for representation.

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 100,102

.. figure:: figures_doc/Fig_Tutor_ToFuMesh_BF1_Select.png
   :height: 300px
   :width: 500 px
   :scale: 100 %
   :alt: Some selected basis functions and their associated mesh centers and knots
   :align: center
   
   Some selected basis functions and their associated mesh centers and knots

All these functionalities are also found in the BaseFunc2D object, which additionally provides specific attributes and methods:

.. list-table:: The attributes of a BaseFunc2D object
   :widths: 10 40
   :header-rows: 1

   * - Attribute
     - Description
   * - self.ID
     - The ID class of the object
   * - self.Mesh
     - The Mesh2D object on which the basis functions are built
   * - self.LFunc, self.NFunc, self.Deg, self.Bound
     - The list of basis functions and the number of basis functions (self.NFunc=len(self.LFunc)), the degree of the basis functions, and the boundary condition (only 0 implemented so far, all points have 1 multiplicity)
   * - self.Func_Centsind, self.Func_Knotsind, self.Func_PMax
     - An array giving the correspondence index between each basis function and all its associated mesh centers (and there are methods to go the other way around), its associated mesh knots, and the position of the maximum of each basis function (either oa mesh center or a knot depending on its degree).
   * - self.FuncInterFunc
     - An array containing indices of all neighbouring basis functions of each basis function (neighbouring in the sense that the intersection of their respective supports is non-zero)

Due to its 2D nature, the BaseFunc2D object class is also equiped with methods to get the support (self.get_SuppRZ) and quadrature points (self.get_quadPoints) of each basis function.

Like the BaseFunc1D object, it provides a method for a least square fit of an input function. In the following example, the coefficients are determined using this method and then fed to various plotting methods used to visalise the function itself or some of its derivatives:

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 106-139,142

.. figure:: figures_doc/Fig_Tutor_ToFuMesh_BF2.png
   :height: 300px
   :width: 500 px
   :scale: 100 %
   :alt: Input 2D emissivity model and fitted BaseFunc2D
   :align: center
   
   Input 2D emissivity model and fitted BaseFunc2D


.. figure:: figures_doc/Fig_Tutor_ToFuMesh_BF2_Deriv.png
   :height: 800px
   :width: 1200 px
   :scale: 100 %
   :alt: Series of derivatives or local quantities of interest of the fitted BaseFunc2D object
   :align: center
   
   Series of derivatives or local quantities of interest of the fitted BaseFunc2D object

Like for the BaseFunc1D object, and in order to facilitate detailed analysis and possibly debugging, you can also plot the key points, support and value of some selected basis functions of your choice:

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 146,148,150

.. figure:: figures_doc/Fig_Tutor_ToFuMesh_BF2_Int1.png
   :height: 400px
   :width: 400 px
   :scale: 100 %
   :alt: Local values of the selected local basis functions, with the underlying mesh
   :align: center
   
   Local values of the selected local basis functions, with the underlying mesh

.. figure:: figures_doc/Fig_Tutor_ToFuMesh_BF2_Int2.png
   :height: 400px
   :width: 400 px
   :scale: 100 %
   :alt: Support and PMax of the selected local basis functions, with the underlying mesh and centers and knots associated to the selected local basis functions
   :align: center
   
   Support and PMax of the selected local basis functions, with the underlying mesh and centers and knots associated to the selected local basis functions

Finally, you can access values and operators of interest regarding some integrated quantities like the squared norm of the gradient, the squared laplacian (to be finished)...

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 154-158

.. literalinclude:: ../../src/Tutorial_ToFu_Mesh.py
   :language: python
   :lines: 159-163

The following table lists the operators which are available in **ToFu_Mesh**, depending on the value of the kwdarg 'Deriv':

.. list-table:: The available values of the 'Deriv' keyword argument for integral computation
   :widths: 10 40
   :header-rows: 1

   * - Value
     - Description
   * - 0 or 'D0'
     - The integrals of the B-splines themselves (integrals of higher order derivatives are all zero)
   * - 'D0N2', 'D1N2', 'D2N2', 'D3N2'
     - The integrals of the squared norm of the 0th, 1st, 2nd and 3rd order derivatives (only 0-th order derivative implemented so far, for Deg=0,1, but not for Deg=2,3)
   * - 'D1FI'
     - The integrated Fisher Information, not implemented so far



.. Local Variables:
.. mode: rst
.. End:
