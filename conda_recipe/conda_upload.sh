# Only need to change these two variables
PKG_NAME=tofu
USER=ToFuProject
OS=linux-64

#mkdir ~/conda-bld
#conda config --set anaconda_upload no
#conda update -n root conda-build
#conda config --append channels conda-forge
#conda config --append channels tofuproject
#export CONDA_BLD_PATH=~/conda-bld
#export VERSION=`date +%Y.%m.%d`
#export VERSION=$(head -n 1 version.txt)

#conda build conda_recipe
echo "Available conda packages:"
echo $(find $CONDA_BLD_PATH/$OS/ -type f -name $PKG_NAME*.tar.bz2)
PKG_REAL=$(find $CONDA_BLD_PATH/$OS/ -type f -name $PKG_NAME-$VERSION-$VADD*.tar.bz2)
echo $PKG_REAL
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l main $PKG_REAL --force
