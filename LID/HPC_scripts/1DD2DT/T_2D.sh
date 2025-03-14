#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --array=1-20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00

source /mnt/pool/6/vvkulagin/FESTIM/miniconda3/bin/activate
conda activate festim-env2
export DIJITSO_CACHE_DIR=./cache
#export MPLCONFIGDIR=./cache

name=${SLURM_JOB_ID}
mkdir ${name}
duration=1ms

cp -rf properties.py ./${name}
cp -rf T_2D.py ./${name}
cp -rf properties.py ./${name}
cp -rf mesh_T2D ./${name}/mesh
cd ${name}

mpirun -np 1 python3 T_2D.py ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_MAX} ${duration}

rm -rf ./cache
cd ..
rm -rf ${name}