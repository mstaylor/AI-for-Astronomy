#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH -J cosmic_ai
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --account=mlsys
#SBATCH --output=outputs/gpu_%j.out
#---SBATCH --cpus-per-task=6 # number of workers + 1 for main process + 1 for buffer

# Load modules
module load gcc nccl
# Add other dependencies (like CUDA) if needed
source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn miniforge
conda activate cosmic_ai

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python inference_infinite.py --device cuda --disable_progress