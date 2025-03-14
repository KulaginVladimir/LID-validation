#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --array=1-20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=300:00:00

source /mnt/pool/6/vvkulagin/FESTIM/miniconda3/bin/activate
conda activate festim-env2
export DIJITSO_CACHE_DIR=./cache
#export MPLCONFIGDIR=./cache

name=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir ${name}
duration=1ms

cp -rf properties.py ./${name}
cp -rf LID_profiling.py ./${name}
cd ${name}

mpirun -np 1 python3 LID_profiling.py ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_MAX} ${duration}

cd ../
rm -rf ${name}