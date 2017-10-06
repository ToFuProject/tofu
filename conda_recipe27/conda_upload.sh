# Only need to change these two variables
PKG_NAME=tofu
USER=ToFuProject

OS=linux-64
mkdir ~/conda-bld
conda config --set anaconda_upload no
conda config --add channels pypi
export CONDA_BLD_PATH=~/conda-bld
#export VERSION=`date +%Y.%m.%d`
export VERSION=$(head -n 1 ../version.txt)
conda build .
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l nightly $CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-0.tar.bz2 --force
