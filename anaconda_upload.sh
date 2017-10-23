#!/bin/bash
set -e

#echo "Converting conda package..."
#conda convert --platform all $PKG_DIR/tofu-*.tar.bz2 --output-dir $PKG_DIR
echo $USER
export USER=ToFuProject
echo "Deploying to anaconda.org..."
echo $CONDA_UPLOAD_TOKEN
echo $USER
echo $PKG_DIR
echo $(ls $PKG_DIR/tofu-*.tar.bz2)
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l main $PKG_DIR/tofu-*.tar.bz2
echo "Successfully uploaded !"
exit 0
