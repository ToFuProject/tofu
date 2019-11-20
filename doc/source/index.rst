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

**tofu** is relevant for all diagnostics integrating, in a finitie field of
view or along a set of lines of sight, a quantity (scalar or vector) for which
the plasma can be considered transparent (e.g.: light in the visible, UV, soft
and hard X-ray ranges, or electron density for interferometers).

**tofu** is **command-line oriented**, for maximum flexibility and
scriptability. The absence of a GUI is compensated by built-in one-liners for
interactive plots.

**tofu** is hosted on github_.

.. _github: https://github.com/tofuproject/tofu

..
   Adding thumbnails to some tutorials
.. raw:: html

    <div style="clear: both"></div>
    <div class="container-fluid hidden-xs hidden-sm">
      <div class="row">
        <a href="auto_examples/tutorials/tuto_plot_basic.html">
          <div class="col-md-3 thumbnail">
            <img src="_images/sphx_glr_tuto_plot_basic_thumb.png">
          </div>
        </a>
	<a href="auto_examples/tutorials/tuto_plot_create_geometry.html">
          <div class="col-md-3 thumbnail">
            <img src="_images/sphx_glr_tuto_plot_create_geometry_thumb.png">
          </div>
        </a>
	<a href="auto_examples/tutorials/tuto_plot_custom_emissivity.html">
          <div class="col-md-3 thumbnail">
            <img src="_images/sphx_glr_tuto_plot_custom_emissivity_thumb.png">
          </div>
        </a>
      </div>
    </div>

Contents
---------

**Tutorials and how to's:**

.. toctree::
   :maxdepth: 1

   How to install tofu <installation.rst>
   A guide to contributing to tofu <contributing.rst>
   Tutorials and examples <auto_examples/index.rst>

* How to create / handle a diagnostic geometry
   - Visit the basic_ tutorial for getting started: create, plot and save
     your first configuration: a vessel and its structures;
   - To know how to load a configuration and create 1D and 2D cameras,
     see the cameras_ tutorial.
* How to compute integrated signal from 2D or 3D synthetic emissivity
   - Visit the tutorial_ for getting started: load an already-existing
     diagnostic geometry in a synthetic diagnostic approach to solve the direct
     problem and compute the line Of Sight and / or Volume of Sight integrated
     signals from a  simulated emissivity field that you provide as an input.
* How to compute tomographic inversions (to do)
   - Use existing diagnostic geometry and signals to solve the inverse problem
     and compute tomographic inversions using a choice of discretization basis
     functions and regularisation functionals.

.. _basic: auto_examples/tutorials/tuto_plot_create_geometry.html
.. _cameras: auto_examples/tutorials/tuto_plot_basic.html
.. _tutorial: auto_examples/tutorials/tuto_plot_custom_emissivity.html
.. _todos: Todos.html


**Code documentation:**

.. note::
   Main **tofu** classes and methods have docstrings so you can
   access contextual help with the usual python syntax from a `ipython` console
   (`print <method>.__doc__`, or `<method>?`).

.. toctree::
   :maxdepth: 1
   :titlesonly:

   tofu
   tofu.geom
   tofu.dumpro
   tofu.data
   tofu.dust


.. raw:: html
   :file: columns.html


----------------

.. _Homepage: index.html
