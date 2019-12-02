# Only need to change these two variables
USER=ToFuProject

echo "Available conda packages:"
echo $PKG_REAL
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l main $PKG_REAL --force
