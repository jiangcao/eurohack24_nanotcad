#!/bin/bash -l
#SBATCH --job-name="nanotcad"
#SBATCH --account=hck
##SBATCH --mail-type=ALL
##SBATCH --mail-user=jiacao@ethz.ch
#SBATCH --time=00:10:00
####SBATCH --output=log_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --constraint=gpu
#SBATCH --uenv=prgenv-gnu/24.7:v3
#SBATCH --view=modules
#SBATCH --view=modules
#SBATCH --partition=debug
##SBATCH --reservation=eurohack24
set -e -u

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export NSYS=1
export NSYS_FILE=bse_dist_${SLURM_JOBID}_${OMP_NUM_THREADS}_numRanks${SLURM_NPROCS}.qdrep

# export CUDA_LAUNCH_BLOCKING=1

# export ARRAY_MODULE=numpy

source ~/load_modules.sh
conda activate quatrex

srun python /users/hck26/repos/eurohack24_nanotcad/src/testing_dist.py
# srun ./nsys.sh python /users/hck26/repos/eurohack24_nanotcad/src/testing_dist.py
