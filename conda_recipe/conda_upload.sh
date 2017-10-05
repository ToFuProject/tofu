# Only need to change these two variables
PKG_NAME=tofu
USER=ToFuProject

OS=linux-64
mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
#export VERSION=`date +%Y.%m.%d`
export VERSION="""$(python -c "import version; print(version.__version__.replace('-','.'))")"""
conda build .
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l nightly $CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-0.tar.bz2 --force
