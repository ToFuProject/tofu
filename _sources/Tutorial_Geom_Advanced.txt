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

.. _advanced: 

Creating a vessel
-----------------

If a vessel object does not exist yet, you have to create one (otherwise you can just load it an existing one).
A vessel object is basically defined by a 2D simple polygon (i.e.: non self-intersecting), 












**Open-source:**
    ToFu is distributed under the very permissive MIT_ license, thus allowing free use, keeping in mind that neither the author nor any of the laboratories in which he worked can be held responsible for unwanted behaviour or results. 
    It is instead transparency that is considered for as a warranty of quality on the long-term.

    ToFu is hosted on github_.

.. _MIT: https://opensource.org/licenses/MIT
.. _github: https://github.com/

**Versions:**
    A list of the successive versions of ToFu,  with a brief description can be found here_.

.. _here: Versions.html

**Dependences:**
    ToFu uses the following python packages_.

.. _packages: Dependencies.html


**Citing ToFu:**
    If you decide to use ToFu for research and published results please acknowledge this work by citing_ the project.

.. _citing: Citation.html


Contents:
---------

**Description of the library modules:**

.. toctree::
   :maxdepth: 1

   overview

**Code documentation:**

.. toctree::
   :maxdepth: 1

   Auto_tofu.geom
   Auto_tofu.plugins.AUG.SXR.geom
   Auto_tofu.plugins.ITER.Bolo.geom


**Tutorials and how to's:**
    * How to build a diagnostic geometry 
        Create apertures and detectors to test a new configuration, to apply ToFu to your own problems, to design a prospetive diagnostic...
    * How to compute integrated signal from 2D or 3D synthetic emissivity
        Use an already-existing diagnostic geometry in a synthetic diagnostic approach to solve the direct problem and compute the line Of Sight and / or Volume of Sight integrated signals from a  simulated emissivity field that you provide as an input.
    * How to compute tomographic inversions
        Use existing diagnostic geometry and signals to solve the inverse problem and compute tomographic inversions using a choice of discretization basis functions and regularisation functionals.
    * How to contribute (to do's)


Indices and tables
==================
* Homepage_
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Homepage: index.html

