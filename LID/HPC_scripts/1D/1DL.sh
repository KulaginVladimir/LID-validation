#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --array=1-20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00

export DIJITSO_CACHE_DIR=./cache
#export MPLCONFIGDIR=./cache

name=${SLURM_JOB_ID}
mkdir ${name}

cp -rf properties.py ./${name}
cp -rf 1DL.py ./${name}
cp -rf properties.py ./${name}
cd ${name}

mpirun -np 1 python3 1DL.py ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_MAX}

rm -rf ./cache