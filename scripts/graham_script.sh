#!/bin/sh
#SBATCH --account=rrg-rgmelko-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --job-name=vita-2d
#SBATCH --output=/home/mbeach/vita-2d.log
#SBATCH --array=1-60

module load julia

julia -p32 -O3 --check-bounds=no --math-mode=fast run_parallel.jl $SLURM_ARRAY_TASK_ID 
