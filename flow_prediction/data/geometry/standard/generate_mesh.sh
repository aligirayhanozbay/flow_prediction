#!/bin/bash

for file in "$(pwd)"/*.geo
do 
    echo "********************* start *****************************"
    echo "geo file:$file"
    gmsh -3 $file -o ${file/.geo/.msh} -format msh2
    echo "msh file:${file/.geo/.msh}"
    echo "********************* end *****************************"
done
echo "Generated all 3D meshes"
