.. _command_line:

Bash access to tofu
===================

tofu is a python library, and it through a python console that you'll get the most of it.
However, it also provides a few bash commands to be used straight form the terminal.
These commands provide a quick and simple access to a few very common features of tofu.
So far they include:

-  :ref:`tofuplot`: for interactive plotting of data from IMAS
-  :ref:`tofucalc`: for computing and interactive plotting of synthetic data from IMAS
-  :ref:`tofu-custom`: for setting-up your own tofu preferences


.. _tofuplot:

tofuplot
--------

tofuplot is available only if IMAS is also installed on your environment.
In that case, the sub-package imas2tofu will be operational.
This sub-package provides an interface between tofu and IMAS, and allows,
among other things, to use tofu to plot experimental data stored in IMAS in
interactive figures.

This feature is typically used as follows:

::

   $ tofuplot -s 54178 -i ece

The line above calls tofuplot with the following arguments:

- -s / --shot : the shot number of the imas data entry (here 54178)
- -i / --ids  : the name of the ids we want to get data from (here ece)

The ids names that can be used are diagnostic ids, they include:

- soft_x_rays
- bolometer
- interferometer
- polarimeter
- reflectometer_profile
- barometry
- spectrometer_visible
- bremsstrahlung_visible

Note that you can combine the plots from several ids in the same figure by
simply adding more ids (they will have common time axis):

::

   $ tofuplot -s 54178 -i ece soft_x_rays interferometer


In all cases, what tofuplot does is simply:

- read the tokamak geometry from ids wall
- read the diagnostic geometry from the provided ids
- compute the Lines of Sight (LOS)
- read the diagnostic experimental data from the provided ids
- display the data (time traces per LOS) and the geometry (tokamak + LOS) in an interactive figure

There are many other parameters that can be specified, like in particular:

- -tok / --tokamak: the name of the tokamak of the imas data entry
- -u / --user     : the user of the imas data entry
- -t0 / --t0      : the name of the time event used as origin (can be a float)

When a parameter is not specified, a default value is used.

For help on the other parameters, type:

::

   $ tofuplot --help

Here is an example of the interactive figure



.. _tofucalc:

tofucalc
--------

tofuplot simply reads and plot data.

By comparison, tofucalc also reads the diagnostics geometry,
but more importantly, it reads plasma profiles (1d radial profiles or 2d maps)
of the quantity of interest for the chosen diagnostic
(i.e.: electron density for interferometer, total radiated power for
bolometer...) and calculates the synthetic data of the
diagnostic (it performs the Line Of Sight integrationi of the quantity).
Finally, it displays the result in the same interactive figure as tofuplot.

Once you have understood the parameters of tofuplot, tofucalc is intuitive
as it requires the same input:

::

   $ tofucalc -s 54178 -i interferometer

Note however, that tofucalc is available for a limited number of diagnostics.
Indeed, it requires pre-tabulating the quantity of interest for each ids and
implementing proper 2D interpolation methods for each type of profile.
So far, it is available for:

- interefreometer
- polarimeter
- bolometer
- bremsstrhalung_visible

Other diagnostics / ids will be added to this list as tofu is developped and
tested on WEST.

Also note that there is a default profile tabulated for each diagostics.
For example interferometer used the 1d electron density profile stored in ids
core_profiles.
But one could of course object that electron density can be stored as a 2D map
in another ids, produced, for example by a plasma edge code.
Computing synthetic data from an alternative source than the tabulated default
o ne is possible, but through the python console only as it requires a more
advanced use of the features offered by tofu.

For some specific users, an alternative way of providing the profiles has been
implemented: it is possible to pass the ids not by its identification
parameters (shot, user, tokamak, ids name...), but via an input file saved with
matlab (.mat).
The input file shall contain an exact representation of the ids.
Likewise, the result can be saved into a .mat output file.
This feature is only available for ids bremsstrahlung_visible so far and it
only the 1d radiation profile that is passed throught the input file.
The equilibrium (for interpolation) and diagnostic geometry are still read from
regular IMAS ids.

::

   $ tofucalc -s 54178 -i interferometer




.. _tofu-custom:

tofu-custom
-----------

tofu used a lot of default parameters, such that providing no parameter at all
is ok when using most method / functions.
However, you may want to customize some of these default parameters to better
suit your usage or liking.

If tofu is installed on a shared cluster, you can't access tofu's default
parametersas modifying them would also affect everybody else's usage.

In order to allow for user-specific customization, run:

::

   $ tofu-custom


This will create a .tofu/ directory in your home (~/), in which tofu will copy
default parameters and data that you are free to edit.
This local copy is thus user-specific and will always have precedence when
importing tofu.

Not all parameters can be customized, and this effort is on-going, but to,
as of tofu 1.4.3, you can edit:

- the imas shortcuts in _imas2tofu-def.py
- the default parameters of tofuplot and tofcalc in _scripts_def.py

Other parameters will be available for customization in future versions.

This hidden directory also holds a openadas2tofu/ sub-directory where all data
downloaded by tofu from `openadas <https://open.adas.ac.uk/>`__
(a free online atomic database) is stored.
