.. _installation:

Installing tofu
================

As of October 2019, `tofu` is still under heavy development. Therefore, installation
instructions are provided officialy for Linux and ITER users. However, instructions
for other platforms are also provided since they have been known to work out for some users.
If installation fails for your use-case, please open an issue over
at `Github. <https://github.com/ToFuProject/tofu/>`__ so that we can try adressing it.

-  :ref:`installing-tofu-on-linux`
-  :ref:`installing-tofu-on-mac`
-  :ref:`installing-tofu-on-windows`
-  :ref:`iter-users`
-  :ref:`installing-as-a-developer`


.. _installing-tofu-on-linux:

Linux
-----

To install and use `tofu` on Linux, we recommend to proceed in two steps: install the
Python package manager conda and then install tofu.
We recommend ``Miniconda`` (light version of the Anaconda Python distribution for data science,
but you can also work with ``pip`` or another Python package manager of
your choice).

-  `Get the latest Miniconda version and install
   it. <https://docs.conda.io/en/latest/miniconda.html>`__
- Install tofu

::

   $ conda install -c tofuproject tofu

- Check that tofu works by printing its version number:

::

   $ python -c "import tofu; print(tofu.__version__)"

Now you can `follow a tutorial. <auto_examples/index.html>`__

.. _installing-tofu-on-mac:

Mac OS X
--------

See :ref:`installing-as-a-developer`.

Additional *caveat*: if you are using a version of `gcc < 8` be sure to
turn off all parallelizations since there is a `known bug with cython
<https://github.com/ToFuProject/tofu/issues/183>`__.

.. _installing-tofu-on-windows:

Windows
-------

See :ref:`installing-as-a-developer`.

Additional *caveat*: you may need to open an ``Anaconda prompt`` (usually found by pressing
the Windows key) to run the commands described in the linked section.


.. _iter-users:

Using tofu on a ITER cluster
----------------------------

If you have an **ITER** account, you can use **tofu** directly from ITER
Computing Cluster. No need to install tofu !

-  Ask for access to ITER servers, if you don't have them already.
-  For information about the cluster, see `this
   link. <https://confluence.iter.org/display/IMP/ITER+Computing+Cluster>`__
-  Open a new terminal and connect to server (see link above)
-  Create a new file in your ``$HOME`` directory, you can name it
   ``load_tofu_modules.sh``
-  Open it and add the following lines:

::

   module refresh
   source /etc/profile.d/modules.sh # make sure you have the module environment
   module purge # unload any previously loaded modules
   module load IMAS/3.24.0-4.1.5 # for IMAS data base
   module load IPython/6.3.1-intel-2018a-Python-3.6.4 # for iPython
   module load PySide2/5.12.0-intel-2018a-Python-3.6.4
   module load ToFu/1.4.0-intel-2018a-Python-3.6.4 # Load tofu :)

-  Convert it to an executable, from the terminal:
   ``chmod +x ~/load_tofu_modules.sh``
-  Execute it: ``./load_tofu_modules.sh``
-  If you are going to use *tofu* often, you might want to add the
   execution of the script to your ``.bash_profile`` (or ``.bashrc``
   file):

::

   echo './load_tofu_modules.sh' >> .bash_profile

You are all set, open a Python/IPython console and try importing tofu.

::

   $ python
   In [1]: import tofu as tf

You can now `follow a tutorial. <auto_examples/index.html>`__


.. _installing-as-a-developer:

Installing tofu as a developer
------------------------------

To install tofu as a developer, we recommend using the conda ecosystem (Miniconda in particular):

-  `Get the latest Miniconda version and install
   it. <https://docs.conda.io/en/latest/miniconda.html>`__

- create a dedicated (Python 3) environment for tofu development and activate it

::

   $ conda create -n tofu3 python=3.6 scipy numpy cython git
   $ conda activate tofu3

-  Move to where you would like to install your local copy of ToFu ``$ cd some_path``
-  ``$ git clone https://github.com/ToFuProject/tofu.git`` (make sure you
   remember the path where you are installing, if you want to install it
   into your home repository, just make sure to ``$ cd ~`` before the
   ``git clone...``)
-  Move to the "cloned" tofu directory that has been created by the git clone command:
   ``cd ~/tofu``
-  Switch to the ``git`` branch you will be working on. If you are just
   starting you probably want to start from the latest develop branch:
   ``git checkout devel``. If you are not familiar with **git** take a
   look at `this tutorial
   (long) <https://www.atlassian.com/git/tutorials>`__ or `this short
   one <https://rogerdudler.github.io/git-guide/>`__
-  Run ``pip install -e .[dev]``. This will install dependencies, compile the
   tofu cython extensions and install it into your conda environment while you can still
   modify the source files in the current repository.`
-  Make sure tofu tests are running by typing ``pytest tofu/tests``
