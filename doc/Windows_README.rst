=============================
 INSTALLING TOFU ON WINDOWS
=============================

This is a quick tutorial on how to install and use ToFu on Windows.
It supposes you are an absolute beginner in coding. If you have some experience you can
probably skip some steps.


Before installing ToFu
======================

Installing Anaconda
--------------------

We will use Miniconda (light version of Anaconda, but you can also work with Anaconda) not only to install and manage the packages necessary for installing ToFu, but also to have a working bash-like Terminal.

* `Get the latest version and install it. <https://docs.conda.io/en/latest/miniconda.html/>`__ 
* Follow the directions (you can use default options)
* Open an ``Anaconda prompt``


Creating a conda environment
----------------------------

We are going to create an environment specific for ToFu. ::

 $ conda create -n tofu3 python=3.6 scipy numpy cython git
 $ conda activate tofu3
 $ conda install m2-base # Get some basic Linux/bash commands (ls, cd, mv, ...)
 
This creates a conda environment named "tofu3" and installs scipy, numpy and cython. The second command activates this environment.


Get the repository
==================

* Create a ssh public key and add it to your GitHub account: `follow this tutorial. <https://help.github.com/en/articles/adding-a-new-ssh-key-to-your-github-account>`__
* Go to ToFu's GitHub repository:  `here. <https://github.com/ToFuProject/tofu/>`__
* Click on "clone or download" and swith to the option "Use SSH". Copy the link.
* Move to where you would like to install **ToFu** ``$ cd some_path``
* ``$ git clone git@github.com:ToFuProject/tofu.git`` (make sure you remember the path where you are installing, if you want to install it into your home repository, just make sure to ``cd ~`` before the ``git clone...``)


Installing ToFu
===============

* Move to the tofu directory, probably: ``cd ~/tofu``
* Switch to the ``git`` branch you will be working on. If you are just starting you probably want to start from the latest develop branch: ``git checkout devel``. If you are not familiar with **git** take a look at  `this tutorial (long)  <https://www.atlassian.com/git/tutorials>`__ or `this short one <https://rogerdudler.github.io/git-guide/>`__
* Compile ``python setup.py build_ext --inplace``
* Install ``python setup.py install``
* Make sure tests are running ``nosetests``
 
 
 Contribute
 ===========

If you wish to contribute, you will probably need a text editor::

 # you can choose emacs or vim as a text editor:
 conda install -c conda-forge emacs 
 conda install -c conda-forge vim 

