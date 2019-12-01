# Only need to change these two variables
PKG_NAME=tofu
USER=ToFuProject

conda config --set anaconda_upload no
conda install anaconda-client conda-build
conda build conda_recipe
export PKG_REAL=$(conda build . --output | tail -1)
echo "Available conda packages:"
echo $PKG_REAL
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l main $PKG_REAL --force
