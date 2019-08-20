=============================
 INSTALLING TOFU ON WINDOWS
=============================

This is a quick tutorial on how to install and use ToFu on Windows.
It supposes you are an absolute beginner in coding. If you have some experience you can
probably skip some steps.


Before installing ToFu
======================

First things first: you will need a bash terminal. If you are running Windows 10, there are already some solutions
integrated with your system. Please google "Linux bash shell on Windows 10" to see the latest solutions available.
If you are running an older Windows, you will need to `install cygwin. <https://cygwin.com/install.html>`__
There are other solutions available, but you should have a bash terminal with **git** and **python** by the end of this step.

Required packages for cygwin
----------------------------

When installing ``Cygwin`` you will need to install the following packages (you can also install them after installing Cygwin, using ``MinGW Installer``). List of packages:

* ``base-cygwin``
* ``bash``
* ``cmake``
* ``curl``
* ``git``
* ``emacs`` or ``vim`` (for editting text/code)
* ``gcc-core``
* ``gcc-g++``
* ``gdb`` (debugger)
* ``make``
* ``openssh``
* ``python3`` (or ``python2``)
* ``python3-pip`` (or ``python2-pip``, depending on the chosen python version)
* ``python3-devel`` (or ``python2-devel``, depending on the chosen python version)
  
If you want a more in-detail documentation, we suggest `this link. <https://www.davidbaumgold.com/tutorials/set-up-python-windows/>`__


Installing Anaconda
===================

There are some python packages that cannot be installed via pip nor the cygwin package manager. 
For this reason, we are going to use **Miniconda**.
* `Get the latest version and install it. <https://docs.conda.io/en/latest/miniconda.html/>`__ 
* Add ``conda`` to your ``$PATH`` variable (this might be slighlty different for you). In your ``.bash_profile``::
  PATH="${HOME}/miniconda3:${PATH}" # where conda's python binary is
  PATH="${HOME}/miniconda3/condabin:${PATH}" # where conda's conda binary is 
  alias conda="conda.bat" # as the conda command binary is in the conda.bat exectuable

Creating a conda environment
----------------------------

Now ``conda`` should be accessible from ``cygwan``::

 $ conda create -n tofu3 scipy numpy cython 
 
This creates a conda environment named "tofu3" and installs scipy, numpy and cython



Get the repository
==================

* Create a ssh public key and add it to your GitHub account: `follow this tutorial. <https://help.github.com/en/articles/adding-a-new-ssh-key-to-your-github-account>`__
* Go to ToFu's GitHub repository:  `here. <https://github.com/ToFuProject/tofu/>`__
* Click on "clone or download" and swith to the option "Use SSH". Copy the link.
* On a new **Cygwin Terminal** window: ``git clone git@github.com:ToFuProject/tofu.git`` (make sure you remember the path where you are installing, if you want to install it into your home repository, just make sure to ``cd ~`` before the ``git clone...``)


Installing ToFu
===============

* Move to the tofu directory, probably: ``cd ~/tofu``
* Switch to the ``git`` branch you will be working on. If you are just starting you probably want to start from the latest develop branch: ``git checkout devel``. If you are not familiar with **git** take a look at  `this tutorial (long)  <https://www.atlassian.com/git/tutorials>`__ or `this short one <https://rogerdudler.github.io/git-guide/>`__
* Installing extra dependencies:
- ``pip install Cython``
- ``pip install numpy``
* ``pip install .`` (it installs the library where you are, so ToFu)
