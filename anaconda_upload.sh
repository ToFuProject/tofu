#!/bin/bash
set -e

conda config --set anaconda_upload no
conda install anaconda-client conda-build
conda build conda_recipe
export PKG_REAL=$(conda build . --output | tail -1)
echo $PKG_REAL

echo "Deploying to anaconda.org..."
export USER=ToFuProject
export PKG_DIR=$HOME/miniconda/conda-bld/$OS/
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l main $PKG_DIR/tofu-*.tar.bz2
echo "Successfully uploaded !"
exit 0
