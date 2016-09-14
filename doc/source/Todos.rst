.. role:: envvar(literal)
.. role:: command(literal)
.. role:: file(literal)
.. role:: ref(title-reference)
.. _overview:

**To do for contributors**
==========================

This to do list includes aspects that require a few hours up to a few months of work.
Whenever possible, a link to a document describing the problem in details is provided.


Math and geometry:
------------------
* Write a C routine for (very) fast computation of solid angle in non-trival cases using spherical geometry and write a python / cython wrapper (weeks)


Coding:
-------
* Parallelize (frist CPU then GPU) the key functions of the geometry module (months)
* Branch the meshing module to allow compatibility with CAID/Pigasus (months)
* Branch the matrix computation module for the same reason (months)
* Branch the inversion module for the same reason (months)

Long term:
----------
* Create a parallel library called ToFuG, which provides all ToFu functionalities through a GUI for each ToFu module (year)



.. Local Variables:
.. mode: rst
.. End:
