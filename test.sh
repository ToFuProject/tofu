
echo ""
echo "--------------------------------"
echo "--------------------------------"
echo "    TEST PYTHON 2.7"
echo "--------------------------------"
echo ""
env_tofu2
nosetests --nocapture -v --with-id tests/tests01_geom/

echo ""
echo "--------------------------------"
echo "--------------------------------"
echo "    TEST PYTHON 3.6"
echo "--------------------------------"
echo ""
env_tofu3
nosetests --nocapture -v --with-id tests/tests01_geom/




