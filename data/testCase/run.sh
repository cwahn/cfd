#!/bin/bash
cartesianMesh >> ./log;
decomposePar >> ./log;
mpirun -np 96 simpleFoam -parallel >> ./log;
reconstructPar >> ./log;
