#!/bin/bash
set -e

#echo "Converting conda package..."
#conda convert --platform all $PKG_DIR/tofu-*.tar.bz2 --output-dir $PKG_DIR

echo "Deploying to anaconda.org..."
export USER=ToFuProject
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l main $PKG_DIR/tofu-*.tar.bz2
echo "Successfully uploaded !"
exit 0
