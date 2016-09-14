.. role:: envvar(literal)
.. role:: command(literal)
.. role:: file(literal)
.. role:: ref(title-reference)

How to compute integrated signal from synthetic emissivity
==========================================================


We are assuming here that you have access to an existing geometry (i.e. to :class:`~tofu.geom.Detect` or :class:`~tofu.geom.GDetect` objects that you or someone else created or that you can load).
It if is not the case you should first create the geometry you need, by following the basic_ geometry tutorial.

.. _basic: Tutorial_Geom_HowToCreateGeometry.html

We are also assuming that you have a code that can produce as output a simulated isotropic emissivity. Either directly or by spacial interpolation, you should be able to write a python function that computes an emissivity value in any arbitrary point inside the vessel volume.


As a prerequisite load the necessary modules:

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> plt.ion()
>>> # tofu-specific
>>> import tofu.pathfile as tfpf


Writing the input function
--------------------------

In the following, all is done assuming **gd** is a :class:`~tofu.geom.GDetect` object, but the same would apply if it is just a :class:`~tofu.geom.Detect` object.
If the :class:`~tofu.geom.GDetect` object you want to use is not already existing in your session, you can load it (with its absolute path and file name) using the :meth:`tofu.pathfile.Open()` function or a dedicated plugin function.

Instances of :class:`~tofu.geom.GDetect` have a method called :meth:`~tofu.geom.GDetect.calc_Sig()`, which takes as input **ff** a python function able to evaluate the emissivity value in any number of points provided in 3D cartesian coordinates.

This function should obey the following constraints:
    * It is a callable with one input argument and optionally keyword arguments
    * The input argument is a (3,N) numpy.ndarray, where N is the number of points at which one wants to evaluate the emissivity, provided in 3D cartesian coordinates (X,Y,Z)

Hence, suppose that we simulate a 2D (i.e.: invariant along the 3rd dimension) gaussian emissivity centered on point (2.,0.), we can define ff as 

>>> def ff(Pts, A=1., DR=1., DZ=1.):
>>>     R = np.hypot(Pts[0,:],Pts[1,:])
>>>     Z = Pts[2,:]
>>>     Emiss = A*np.exp(-(R-2.)**2/DR**2 - (Z-0.)**2/DZ**2)
>>>     return Emiss

What will happen when we feed ff to :meth:`~tofu.geom.GDetect.calc_Sig()` depends on the choice of method for the integration:
    * If we want a volumic integration, the VOS of each detector will be discretized and ff will be called to evaluate the emissivity at each point before perfoming the integration
    * If a Line Of Sight integration is desired, only the LOS is discretized for integration and the result is multiplied by the etendue

By default, the method uses a pre-computed discretization of the VOS (because re-computing the solid angle for each point every time is costly), but this feature can be suppressed by setting PreComp=False if you want to use customized integration parameters.
For example, in both cases, the numerical integration can be done by choosing the resolution of the discretization, or by using an iterative algorithm that only stops when the required relative error on the integral value is reached.
In our case:

>>> # Compute synthetic signal using a volume approach with resolution-fixed numerical integration method
>>> sigVOS, ldet = gd.calc_Sig(ff, extargs={'A':1.,'DR':1.,'DZ':1.}, Method='Vol', Mode='simps', PreComp=False)
>>> sigLOS, ldet = gd.calc_Sig(ff, extargs={'A':1.}, Method='LOS', Mode='quad', PreComp=False)
>>> print sigVOS, sigLOS
[[  1.31675917e-06   1.40620027e-06]] [[  1.31408026e-06   1.39941326e-06]]

Notice that when using the 'quad' numerical integration method, only one extra argument can be passed on to ff.
Notice the small differences in the volume and LOS approaches, due to the small non-zero second derivative of the emissivity field and to boundary effects (where there is small partial obstruction of the VOS).


If your code gives a tabulated emissivity field
-----------------------------------------------

Then you simply have to include an intermediate function that interpolates your emissivity field to compute it at any point. Like in the following example:

>>> def ff(Pts):
>>>     R = np.hypot(Pts[0,:],Pts[1,:])
>>>     Z = Pts[2,:]
>>>     Emiss = ff_interp(R,Z)
>>>     return Emiss

Where ff_interp() is an interpolating function using tabulated output from your code.

Plotting the result
-------------------

The :meth:`~tofu.geom.GDetect.plot_Sig()` method provides a way of plotting the result, either by feeding it the output signal of :meth:`~tofu.geom.GDetect.calc_Sig()` or directly **ff** (in which case it simply calls :meth:`~tofu.geom.GDetect.plot_Sig()` for you).
This feature is only available for :class:`~tofu.geom.GDetect` objects since the signal of a single detector is just a single value that does not really require plotting...



Indices and tables
------------------
* Homepage_
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Homepage: index.html

