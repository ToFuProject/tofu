.. _installation:

Installing tofu
================

As of January 2022, `tofu` is still under development. Since binary packages of tofu are regularly released
on `conda-forge` for Linux, Mac and Windows, this should be the preferred way for users to install `tofu` (see below),
although it is not the only one.

Special installation instructions are provided for *ITER users* as well as *developers*.

If installation fails for your use-case, please open an issue over
at `Github. <https://github.com/ToFuProject/tofu/>`__ so that we can try adressing it.

-  :ref:`installing-tofu-using-conda-forge`
-  :ref:`iter-users`
-  :ref:`installing-as-a-developer`

.. note::
   If you encounter problems during the installation, check our list of
   :ref:`known bugs <Knownbugs>` or `open an issue. <https://github.com/ToFuProject/tofu/issues>`__.

.. _installing-tofu-using-conda-forge:

Installing `tofu` from conda-forge
-----------------------------------

`tofu` can be installed using the `conda` package manager and the `conda-forge` software channel.
We recommend to proceed in two steps: install the `conda` package manager  and then install `tofu`.
We recommend `Miniconda` (a light version of the Anaconda Python distribution for data science,
but you can also work with `pip` or another Python package manager of
your choice).

-  `get the latest Miniconda version and install
   it. <https://docs.conda.io/en/latest/miniconda.html>`__
- install tofu with::

   $ conda install -c conda-forge tofu

- check that tofu works by printing its version number:::

   $ python -c "import tofu; print(tofu.__version__)"

If the version number printed correctly, congratulations, you can now `follow a tutorial! <auto_examples/index.html>`__

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

- Create a dedicated (Python 3) environment for tofu development and activate it::

   $ conda create -n tofu_env python=3.9 scipy numpy cython git
   $ conda activate tofu_env

-  Move to where you would like to install your local copy of ToFu ::

    $ cd some_path

-  Clone the repository source code (make sure you
   remember the path where you are installing, if you want to install it
   into your home repository, just make sure to ``$ cd ~`` before the
   ``git clone...``)::

    $ git clone https://github.com/ToFuProject/tofu.git

-  Move to the "cloned" tofu directory that has been created by the git clone command::

   $ cd ~/tofu

-  Switch to the ``git`` branch you will be working on. If you are just
   starting you probably want to start from the latest develop branch, which is
   normally checked out by the above ``git clone`` command. If you were trying to fix a
   bug or code up a new feature you would create or check out a dedicated branch at this point.
   If you are not familiar with **git**, take a look at `this tutorial (long) <https://www.atlassian.com/git/tutorials>`__
   or `this short one <https://rogerdudler.github.io/git-guide/>`__)::

    $ git checkout devel

-  We are now ready to install ``tofu``. The following command will install dependencies,
   compile the tofu cython extensions and install it into your conda environment.
   We use the ``-e`` flag to tell ``pip`` that it should install the package in editable mode,
   which will allow you to modify the source files and run the modified code (although you sometimes need to reload your interpreter)::

    $ pip install -e .[dev]

-  To make sure the installation is working, we run the tofu test suite. This should yield
   a report that indicates which tests passed at the end, for example ``==== 198 passed, 927 warnings in 670.06s (0:11:10) ====``.
   All tests are expected to pass to indicate that tofu installed correctly::

   $ pytest tofu/tests

-  If you would like to contribute to `tofu`, check out our dedicated guide, :ref:`contributing-to-tofu`.
   Alternatively, we also have some developer guides :ref:`devtutos`.
