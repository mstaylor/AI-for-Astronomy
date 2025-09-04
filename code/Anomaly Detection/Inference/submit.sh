#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH -J cosmic_ai
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --account=mlsys
#SBATCH --output=outputs/cpu_%j.out

# Load modules
module load gcc nccl
# Add other dependencies (like CUDA) if needed
source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn miniforge
conda activate cosmic_ai

python inference_infinite.py --disable_progress