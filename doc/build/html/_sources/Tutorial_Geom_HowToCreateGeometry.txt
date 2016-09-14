.. role:: envvar(literal)
.. role:: command(literal)
.. role:: file(literal)
.. role:: ref(title-reference)

How to create a diagnostic geometry
====================================

It is the geometry module that provides all the necessary tools for creating a new diagnostic.
A diagnostic comprises a set of detectors (ToFu creates one object for each detector and you can then group them into a larger object to represent cameras).
Each detector is basically defined by its active surface, which should be a planar polygon, and by a set of optics through which it 'sees' the plasma.
The optics can be a converging spherical lens or an arbitrary number of apertures (of arbitrary shape).
Each detector is also assigned to a vessel, which defines the linear or toroidal volume in which the plasma can exist.

The following guides you through the creation of these objects in the famous 'hello-world' example:

To find out more about what you can do with the geometry module check out the advanced_ tutorial.

.. _advanced: Tutorial_Geom_Advanced.html


As a pre-requisite, let's load some basic useful libraries in a ipython session, as well as the geometry module of ToFu:

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> plt.ion()
>>> # tofu-specific
>>> import tofu.geom as tfg


Creating, plotting and saving a vessel
--------------------------------------

If a vessel object does not exist yet, you have to create one (otherwise you can just load it an existing one).
A vessel object is basically defined by a 2D simple polygon (i.e.: non self-intersecting), that is then expanded linearly or toroidally depending on the desired configuration.
This polygon limits the volume available for the plasma, where the emissivity can be non-zero. It is typically defined by the inner wall in a tokamak.

Let's define the polygon limiting the vessel as a circle with a divertor-like shape at the bottom:

>>> # Define the center, radius and lower limit
>>> R0, Z0, rad, ZL = 2., 0., 1., -0.85
>>> # Define the key points in the divertor region below ZL
>>> Div_R, Div_Z = [R0-0.2, R0, R0+0.2], [-1.2, -0.9, -1.2]
>>> # Find the angles corresponding to ZL and span the rest
>>> thet1 = np.arcsin((ZL-Z0)/rad)
>>> thet2 = np.pi - thet1
>>> thet = np.linspace(thet1,thet2,100)
>>> # Assemble the polygon
>>> poly_R = np.append(R0+rad*np.cos(thet), Div_R)
>>> poly_Z = np.append(Z0+rad*np.sin(thet), Div_Z)
>>> # Plot for checking
>>> f, l, a = plt.figure(facecolor='w', figsize=(6,6)), plt.plot(poly_R, poly_Z), plt.axis('equal')

.. figure:: figures_doc/Fig_Tutor_Geom_Basic_01.png
   :height: 300px
   :width: 300 px
   :scale: 100 %
   :alt: Polygon used for defining the vaccum vessel where the plasma may live
   :align: center

   Polygon used for defining the vaccum vessel where the plasma may live

Notice that the polygon does not have to be closed, ToFu will anyway check that and close it automatically if necessary

Now let's feed this 2D polygon to the appropriate ToFu class and specify that it should be a toroidal type (if linear type is chosen, the length should be specified by the 'DLong' keyword argument).
ToFu also asks for a name to be associated to this instance, and an experiment ('Exp') and a shot number (useful when the same experiment changes geometry in time).

>>> # Create a toroidal Ves instance with name 'World', associated to experiment 'Misc' (for 'Miscellaneous') and shot number 0
>>> ves = tfg.Ves('HelloWorld', [poly_R,poly_Z], Type='Tor', Exp='Misc', shot=0) 

Now the vessel instance is created. I provides you with several key attributes and methods (see :class:`~tofu.geom.Ves` for details).
Among them the Id attribute is itself a class instance that contains all useful information about this vessel instance for identification, saving... In particular, that's where the name, the default saving path, the Type, the experiment, the shot number... are all stored. 
A default name for saving was also created that automatically includes not only the name you gave but also the module from which this instance was created (tofu.geom or tfg), the type of object, the experiment, the shot number...
This recommended default pattern is useful for quick identification of saved object, it is advised not to modify it.

>>> print ves.Id.SaveName
TFG_VesTor_Misc_World_sh0

Now, we can simply visualise the created vessel by using the dedicated method (keyword argument 'Elt' specifies the elements of the instance we want to plot, typically one letter corresponds to one element, here we just want the polygon):

>>> # Plot the polygon, by default in two projections (cross-section and horizontal) and return the list of axes
>>> Lax = ves.plot(Elt='P')

.. figure:: figures_doc/Fig_Tutor_Geom_Basic_02.png
   :height: 300px
   :width: 600 px
   :scale: 100 %
   :alt: The created vessel instance, plotted in cross-section and horizontal projections
   :align: center

   The created vessel instance, plotted in cross-section and horizontal projections


Since the vessel is an important object (it defines where the plasma can live), all the other ToFu objects rely on it. It is thus important that you save it so that it can be used by other ToFu objects when necessary.

>>> ves.save(Path='./')

This method will save the instance as a numpy compressed file (.npz), using the path and file name found in ves.Id.SavePath and ves.Id.SaveName.
While it is highly recommended to stick to the default value for the SaveName, but you can easily modify the saving path if you want by specifying it using keyword argument Path. 



Creating, plotting and saving structural elements
-------------------------------------------------

Unlike the vessel, which is important for physics reasons, the structural elements that ToFu allows to create are purely for illustrative purposes. They are entirely passive and have no effect whatsoever on the computation of the volume of sight of the detectors or on the plasma volume and are just made available for illustrations.

Like for a vessel, a structural element is mostly defined by a 2D polygon. If a vessel instance is provided, the type of the structural element (toroidal or linear) is automatically the same as the type of the vessel, otherwise the type must be specified.
For plotting, structural elements that enclose the entirety of a vessel are automatically transparent, and gray if they don't.

>>> # Define two polygons, one that does not enclose the vessel and one that does
>>> thet = np.linspace(0.,2.*np.pi,100)
>>> poly1 = [[2.5,3.5,3.5,2.5],[0.,0.,0.5,0.5]]
>>> poly2 = [R0+1.5*np.cos(thet),1.5*np.sin(thet)]
>>> # Create the structural elements with the appropriate ToFu class, specifying the experiment and a shot number for keeping track of changes
>>> s1 = tfg.Struct('S1', poly1, Ves=ves, Exp='Misc', shot=0)
>>> s2 = tfg.Struct('S2', poly2, Ves=ves, Exp='Misc', shot=0)
>>> # Plot them on top of the vessel
>>> Lax = ves.plot(Elt='P')
>>> # Re-use the same list of axes to overlay the plots
>>> Lax = s1.plot(Lax=Lax)
>>> Lax = s2.plot(Lax=Lax)

.. figure:: figures_doc/Fig_Tutor_Geom_Basic_03.png
   :height: 300px
   :width: 600 px
   :scale: 100 %
   :alt: The created structural elements, plotted over the structural elements on both projections
   :align: center

   The created structural elements, plotted over the structural elements on both projections


It is not necessary for ToFu (since structural elements are used by no other objects) but for convenience you can save a structral element using the same save() method as for any other object.



Creating apertures
------------------

An aperture is also mosly defined by a planar polygon, except that the polygon coordinates should be provided in 3D cartesian coordinates (even though the polygon is planar, it mey not live in the same plane as other apertures or as the detector).

We can easily define two different polygons for two different apertures

>>> # Define the planes in which they will live by a point (O) and a vector (n)
>>> O1, n1 = (3.0,0.00,0.52), (-1.,0.1,-0.9)
>>> O2, n2 = (2.9,0.01,0.48), (-1.,0.0,-1.0)
>>> # Compute local orthogonal basis vectors in the planes
>>> e11, e21 = np.cross(n1,(0.,0.,1.)), np.cross(n2,(0.,0.,1.))
>>> e12, e22 = np.cross(e11,n1), np.cross(e21,n2)
>>> # Normalize
>>> e11, e12 = e11/np.linalg.norm(e11), e12/np.linalg.norm(e12)
>>> e21, e22 = e21/np.linalg.norm(e21), e22/np.linalg.norm(e22)
>>> # Implement the planar polygons 2D coordinates
>>> p1_2D = 0.005*np.array([[-1.,1.,1.,-1],[-1.,-1.,1.,1.]])
>>> p2_2D = 0.01*np.array([[-1.,1.,1.,-1],[-1.,-1.,1.,1.]])
>>> # Compute the 3D coordinates
>>> p1 = [O1[0] + e11[0]*p1_2D[0,:] + e12[0]*p1_2D[1,:], O1[1] + e11[1]*p1_2D[0,:] + e12[1]*p1_2D[1,:], O1[2] + e11[2]*p1_2D[0,:] + e12[2]*p1_2D[1,:]]
>>> p2 = [O2[0] + e21[0]*p2_2D[0,:] + e22[0]*p2_2D[1,:], O2[1] + e21[1]*p2_2D[0,:] + e22[1]*p2_2D[1,:], O2[2] + e21[2]*p2_2D[0,:] + e22[2]*p2_2D[1,:]]
>>> # Create the apertures, specifying also the diagnostic the apertures belong to
>>> a1 = tfg.Apert('A1', p1, Ves=ves, Exp='Misc', shot=0, Diag='misc')
>>> a2 = tfg.Apert('A2', p2, Ves=ves, Exp='Misc', shot=0, Diag='misc')
>>> # Plot them, both the polygon and the vector, with the associated vessel (using EltVes), in 3D
>>> Lax = a1.plot(Elt='PV', EltVes='P')
>>> Lax = a2.plot(Lax=Lax, Elt='PV')

.. figure:: figures_doc/Fig_Tutor_Geom_Basic_04.png
   :height: 300px
   :width: 600 px
   :scale: 100 %
   :alt: The created apertures, plotted over the vessel on both projections
   :align: center

   The created apertures, plotted over the vessel on both projections

ToFu allows you to save the apertures, if you wish, but if you created then only to pass tem on to detectors, you can also skip saving them. Indeed, once the detector associated to these apertures is created, you will save the detector object instead, and ToFu will automatically store all information about the apertures (everything necessary to re-create them when loading the detector object).



Creating, plotting and saving detectors objects
-----------------------------------------------

A detector object is defined in the same way as an aperture, except that it needs to know which optics it is associated to. The optics can be either a converging spherical lens or, as in this case, a list of apertures.
In the folloing we will thus create two detectors (re-using the same planes as for the apertures for simplicity, but they could lie in any plane).

>>> # Choose different reference points for the 2 planes
>>> Od1, Od2 = (3.05,0.00,0.54), (3.05,0.00,0.50)
>>> # Implement the planar polygons 2D coordinates
>>> pd1_2D = 0.005*np.array([[-1.,1.,1.,-1],[-1.,-1.,1.,1.]])
>>> pd2_2D = 0.005*np.array([[-1.,1.,1.,-1],[-1.,-1.,1.,1.]])
>>> # Compute the 3D coordinates
>>> pd1 = [Od1[0] + e11[0]*pd1_2D[0,:] + e12[0]*pd1_2D[1,:], Od1[1] + e11[1]*pd1_2D[0,:] + e12[1]*pd1_2D[1,:], Od1[2] + e11[2]*pd1_2D[0,:] + e12[2]*pd1_2D[1,:]]
>>> pd2 = [Od2[0] + e21[0]*pd2_2D[0,:] + e22[0]*pd2_2D[1,:], Od2[1] + e21[1]*pd2_2D[0,:] + e22[1]*pd2_2D[1,:], Od2[2] + e21[2]*pd2_2D[0,:] + e22[2]*pd2_2D[1,:]]
>>> # Create the detectors, specifying also the diagnostic and the Optics
>>> d1 = tfg.Detect('D1', pd1, Optics=[a1,a2], Ves=ves, Exp='Misc', shot=0, Diag='misc')
>>> d2 = tfg.Detect('D2', pd2, Optics=[a2], Ves=ves, Exp='Misc', shot=0, Diag='misc')

The computation of the detectors may take a while (~3 min) because ToFu automatically computes the natural Line Of Sight (LOS) and its etendue, the Volume Of Sight (VOS), a pre-computed 3D grid of the VOS for faster computation of synthetic signal...
Some of these automatic computations can be de-activacted using the proper keyword arguments, or the resolution of the discretization can downgraded for faster computation (see :class:`~tofu.geom.Detect` for details).

A Detect object is at the core of the added value of ToFu: all relevant quantities are automatically computed, and can be obtained and plotted via attributes and methods. 

>>> # Plot the detectors, specifying we want not only the polygon but also the perpendicular vector and the viewing cone ('C'), as well as elements of the LOS, Optics and vessel
>>> Lax = d1.plot(Elt='PVC', EltOptics='P', EltLOS='L', EltVes='P')
>>> Lax = d2.plot(Lax=Lax, Elt='PVC', EltOptics='P', EltLOS='L')

.. figure:: figures_doc/Fig_Tutor_Geom_Basic_05.png
   :height: 300px
   :width: 600 px
   :scale: 100 %
   :alt: The created detectors, with associated apertures and vessel, on both projections
   :align: center

   The created detectors, with associated apertures and vessel, on both projections

Using d1.save() would save detector 1 and all necessary info about its associated optics (i.e.:apertures) will also be included in the file so it is not necessary to save the apertures separately (unless you need to for something else).
Usually, tomography diagnostics do not have a few but many different detectors, grouped in cameras (often a group of detectors sharing a common aperture).
ToFu provides an GDetect object that allows you to group a list of detectors and treat them like a single object (each method is automatically applied to all the detectors included in the GDetect object).



Creating, plotting and saving GDetect objects
---------------------------------------------

Once several Detect objects are created, they can be fed to a GDetect object to be handle as a single object.

>>> # Create the group of detectors by feeding a list of detectors
>>> gd = tfg.GDetect('GD', [d1,d2], Exp='Misc', shot=0)
>>> # Plot the group of detectors as a single set
>>> Lax = gd.plot(Elt='PVC', EltOptics='P', EltLOS='L', EltVes='P')

The last command yields the same result as the previous figure.


Congratulations ! You completed the basic tutorial for getting started and creating your own geometry, take you time now to explore all the methods and attributes of the classes introduced in :mod:`tofu.geom`.



Indices and tables
------------------
* Homepage_
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Homepage: index.html

