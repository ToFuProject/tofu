.. role:: envvar(literal)
.. role:: command(literal)
.. role:: file(literal)
.. role:: ref(title-reference)

AUG-specific tutorial
=====================

How to access the ToFu library
------------------------------

The plugin for ASDEX Upgrade (AUG) is hosted on the theory (Tok) clusters of the Max-Planck Institute for Plasma Physics (IPP) in Garching.
If you have an account in IPP, you can then connect to one of the tok clusters where the library is hosted, via the command:

>>> ssh toki01

Enter your password and then you need to load the module in the terminal

>>> module load tofu

You may need to load other modules on which ToFu depends (see the dependencies_).

.. _dependencies : Dependencies.html

You can then start a ipython console and load the AUG plugin for ToFu:

>>> import tofu.plugins.AUG as tfAUG


How to load existing geometry
------------------------------

You can now load the geometry that was already computed and stored for some diagnostics (only the Soft X-Ray diagnostic at this date).
In general loading the geometry means using a method of the plugin that will load and return a list of :class:`tofu.geom.GDetect` instances.
On AUG, each :class:`tofu.geom.GDetect` instance corresponds to a camera head.
Since the geometry (position, aperture size...) of each camera head may change in time (changes are sometimes implemented between experimental campaigns), you can specify a shot number and the plugin will return the latest geometry that was computed before that shot number (only a few have been computed so far, but more will come). 

>>> LGD = tfAUG.SXR.geom.load(shot=31801)

This command returns a list of :class:`tofu.geom.GDetect` instances with the latest geometry computed before shot 31801.
 

How to load SXR data
--------------------

the ToFu plugin for AUG also comes with a data loading and formatting routine.
In addition to a shot number and a time window, you can choose whether the data is returned as numpy arrays or as a ToFu-compatible object for data pre-treatment via the Tofu keyword argument.

>>> pre = tfAUG.SXR.data.load(shot=30801, Dt=[2.,2.01], Method='SX', Tofu=True)

This command will return the SXR data for shot 30801 in the [2.,2.01] s time window using diagnostic 'SX.' (instead of SSX), in the form of a ToFu-compatible object.
See the documentation of the method for more details.


Indices and tables
------------------
* Homepage_
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Homepage: index.html

