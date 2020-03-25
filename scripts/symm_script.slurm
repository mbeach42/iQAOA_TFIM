#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --job-name=vita-2d-p2
#SBATCH --output=vita2d-p2.log
#SBATCH --array=1-2%50

module load julia
julia --project=@. -O3 --check-bounds=no --math-mode=fast run_parallel.jl $SLURM_ARRAY_TASK_ID 
