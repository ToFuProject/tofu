# Only need to change these two variables
PKG_NAME=tofu
USER=ToFuProject

OS=linux-64
mkdir ~/conda-bld
conda config --set anaconda_upload no
conda update -n root conda-build
conda config --add channels conda-forge
conda config --add channels tofuproject
export CONDA_BLD_PATH=~/conda-bld
#export VERSION=`date +%Y.%m.%d`
#export VERSION=$(head -n 1 version.txt)

conda build $RECIPE
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l main $CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-$VADD.tar.bz2 --force
