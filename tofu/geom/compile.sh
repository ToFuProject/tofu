
cp GG02.pyx GG03.pyx

echo ""
env_tofu2
python setup.py build_ext --inplace

echo ""
env_tofu3
python setup.py build_ext --inplace




