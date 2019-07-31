if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
-O miniconda.sh; export VADD="py27"; else wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
-O miniconda.sh; export VADD="py36";  fi
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda install conda-build
conda update -q conda
conda update -n root conda-build
conda config --set anaconda_upload no
conda config --append channels conda-forge
conda config --append channels tofuproject
conda info -a
if [[ "$TRAVIS_PYTHON_VERSION" == "3.7-dev" ]]; then export THIS_PY_VERSION="3.7";
else THIS_PY_VERSION=$TRAVIS_PYTHON_VERSION;
fi
conda install -q python=$THIS_PY_VERSION conda-build anaconda-client nose
nose-timer coverage codecov
export REV=$(python -c "import _updateversion as up; out=up.updateversion(); print(out)")
export VERSION=$(echo $REV | tr - .)
echo $REV
conda build conda_recipe
export PKG_DIR=$HOME/miniconda/conda-bld/linux-64/
conda install tofu --use-local

nosetests tofu.tests --nocapture -v --with-id --with-timer --with-coverage --cover-package=tofu
