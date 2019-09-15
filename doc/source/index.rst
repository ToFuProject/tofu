================================
Welcome to tofu's documentation!
================================


**tofu** stands for **To**\mography for **Fu**\sion, it is an IMAS-compatible
open-source machine-independent python library
with non-open source plugins containing all machine-dependent routines.

It aims at providing the **fusion** and **plasma** community with an object-
oriented, transparent and documented tool for designing
**tomography diagnostics**, computing **synthetic signal** (direct problem)
as well as **tomographic inversions** (inverse problem). It gives access to a
full 3D description of the diagnostic geometry, thus reducing the impact of
geometrical approximations on the direct and, most importantly, on the inverse
problem.

**tofu** is relevant for all diagnostics integrating, in a finitie field of view or along a set of lines of sight, a quantity (scalar or vector) for which the plasma can be considered transparent (e.g.: light in the visible, UV, soft and hard X-ray ranges, or electron density for interferometers).

**tofu** is **command-line oriented**, for maximum flexibility and scriptability.
The absence of a GUI is compensated by built-in one-liners for interactive plots.



**Open-source:**

**tofu** is distributed under the very permissive MIT_ license, thus allowing free use, keeping in mind that neither the author nor any of the laboratories in which he worked can be held responsible for unwanted behaviour or results.
 It is instead transparency, reproducibility and incremental improvements that guarantee quality on the long-term.

 **tofu** is hosted on github_.

.. _MIT: https://opensource.org/licenses/MIT
.. _github: https://github.com/

**Versions:**
    A list of the successive versions of **tofu**,  with a brief description can be found here_.

.. _here: Versions.html

**Dependencies:**
    **tofu** uses the following python packages_.

.. _packages: Dependencies.html


**Citing tofu:**
    If you decide to use **tofu** for research and published results please acknowledge this work by citing_ the project.

.. _citing: Citation.html

**Feedback - bug report - wish list**
    To provide feedback on **tofu** itself please use the github_ page.
    .. _github: https://github.com/

To provide feedback on a specific plugin, please refer to that plugin's webpage where a contact will be indicated.


**Miscellaneous**
    **tofu** is tested with the nose_/1.3.4 library (not all methods are tested yet, in process...)
    **tofu** can be installed using the distutils_ library.
    .. _nose: https://pypi.python.org/pypi/nose

.. _distutils: https://docs.python.org/2/distutils/


Contents:
---------

**Description of the library structure:**

.. toctree::
   :maxdepth: 1

   overview
   auto_examples/index.rst

**Code documentation:**

Notice that the main **tofu** classes and methods have docstrings so you can access contextual help with the usual python syntax from a iython console (print <method>.__doc__, or <method>?).

.. toctree::
   :maxdepth: 2
   :numbered:
   :titlesonly:

   tofu.geom
   tofu.data
   tofu.dumpro

**Tutorials and how to's:**
    * How to create / handle a diagnostic geometry
        - Visit the basic_ tutorial for getting started: create, plot and save a vessel, apertures and detectors and group them
        - Check out the complete set of detailed_ tutorials for more info on each of these aspects and on others.
    * How to compute integrated signal from 2D or 3D synthetic emissivity
        - Visit the tutorial_ for getting started: load an already-existing diagnostic geometry in a synthetic diagnostic approach to solve the direct problem and compute the line Of Sight and / or Volume of Sight integrated signals from a  simulated emissivity field that you provide as an input.
    * How to compute tomographic inversions (to do)
        Use existing diagnostic geometry and signals to solve the inverse problem and compute tomographic inversions using a choice of discretization basis functions and regularisation functionals.
    * How to contribute (todos_)

.. _basic: Tutorial_Geom_HowToCreateGeometry.html
.. _detailed: Tutorial_Geom_Advanced.html
.. _tutorial: Tutorial_Geom_SynthDiag_Basic.html
.. _todos: Todos.html


**Plugin-specific tutorials:**
    * AUG_ : load the existing geometry, load data
    * ITER_ : load existing geometry...

.. _AUG : Tutorial_AUG.html
.. _ITER : Tutorial_ITER.html


.. raw:: html
   :file: columns.html

----------------

.. _Homepage: index.html
