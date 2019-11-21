.. _devtutos:

=========================
Tutorials For Developers
=========================

Some tutorials for developers.

-----------------


-------------------
How To Do A Release
-------------------

1. Write down the release_notes:

   a) in **rST** format: ``realeas_notes_XYZ.rst``
   b) Start with a summary **Main Changes** (TL;DR version of the whole file)
   c) add all issues and PR related references
   d) add list of contributors
   e) add list of perspectives and tag contributors

2. Release an alpha version:

   a) Create an annotaded tag (do not use GitHub's release system) ``git tag -a X.Y.Z-alpha0 -m "annotation"``
   b) Update the version of the repo: ``python -c "import _updateversion as up; out = up.updateversion(); print(out)"``
   c) push on GitHub server
   d) Verify that ONE build passes on travis:
      the deployment on conda should pass for all travis-build, but for pypi only
      the first build will pass
   e) Install it locally on all available platforms and environments (by hand),
      with all different packaging tools. And try some tests.
   f) Easybuild: try to create a tarball for Jira machines
   g) For IRFM servers, build the conda_recipe by hand:

      .. code-block:: python

	 $ if [[ "$TRAVIS_PYTHON_VERSION" == "3.7-dev" ]]; then export VADD="py37"; else export VADD="py36";  fi
         $ export CONDA_BLD_PATH=$(pwd)/conda-bld/ # you should be in the tofu dir
         $ export REV=$(python -c "import _updateversion as up; out=up.updateversion(); print(out)")
         $ export VERSION=$(echo $REV | tr - .)
         $ echo $REV

#. Update the ``releases.rst`` file in the web documentation:

   #. create a symbolic link from ``$TOFU_DIR/release_notes/release_notes_XYZ.rst`` to ``$TOFU_DIR/doc/source/release_notes/release_notes_XYZ.rst``
   #. Update the ``.rst`` file
   #. Make and publish doc (see tutorial below)

#. Release the real version:

   a) Follow the same steps as for alpha version, just change the tag and annotation.

#. Send an email to users.


------------------------
How To Construct The Doc
------------------------

Adding text, updating the API:

#. The easiest way is to have two different repo: ``$TOFU_DIR``, for building the doc and ``$TOFU_DIR/../test_doc/tofu/`` for seeing the resulting html (always in the ``gh-pages`` branch)
#. If updating the API, make sure that you have installed the right version of **tofu**
#. Change to the ``new-doc`` branch
#. Go to ``$ cd $TOFU_DIR/doc/``
#. Modify the corresponding file (somewhere in ``src/a-file.rst``)
#. If you added a file
#.  ``sphinx-apidoc -Mf -P -d 5 -o source/ ../tofu/ {../*test*,../*mag*,../*imas2tofu*,../*geom/inputs}``
#. ``$ make clean`` (to make sure only the needed files will be stored)
#. ``$ make html``
#. Go to the ``gh-pages`` branch and open the ``index.html`` file (from the top)
#. Commit and push the changes to the ``new-doc`` branch: to keep the changes in the sources
#. Commit and push the changes to the ``gh-pages``, this will update the website.


Adding a tutorial:

#. Add your tutorial in ``$TOFU_DIR/examples/tutorials`` with the proper naming conventions
#. The sphinx gallery is automatically updated when building the doc (as above)
#. Change the ``index.rst`` file to include a thumbnail of your tutorial
