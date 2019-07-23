#!/bin/bash

echo " "
echo "Loading modules"
echo "---------------"
source /etc/profile.d/modules.sh
module use /Applications/Modules/compilers
module use /Applications/Modules/soft
module use /work/imas/etc/modulefiles
# Get module list from file
file="modules_INTRA_IRFM.txt"
while IFS= read -r line
do
    module load $line
    #printf '%s\n' "$line"
done < "$file"
echo " "
echo "Modules list:"
module li

echo " "
echo "Compile with optimization"
echo "-------------------------"
f2py -c -m mag_ripple mag_ripple.f
