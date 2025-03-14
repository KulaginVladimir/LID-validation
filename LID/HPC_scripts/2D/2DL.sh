#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00

source /mnt/pool/6/vvkulagin/FESTIM/miniconda3/bin/activate
conda activate festim-env2
export DIJITSO_CACHE_DIR=./cache

#export MPLCONFIGDIR=./cache


E=1.003
duration=1ms
name=flux_${duration}_E${E}_${SLURM_JOB_ID}
#name=flux_1ms_E${E}_${SLURM_JOB_ID}
mkdir ${name}

cp -rf properties.py ./${name}
cp -rf 2DL.py ./${name}
cp -rf properties.py ./${name}
cp -rf ./mesh ./${name}/mesh
cd ${name}

mpirun -np 8 python3 2DL.py ${E}

rm -rf ./cache