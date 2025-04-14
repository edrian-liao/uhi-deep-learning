#!/bin/bash

#SBATCH --mail-user=edrianpaul.liao@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./gen_gp.out
#SBATCH --error=./gen_gp.err
#SBATCH --mem=100G
#SBATCH -p gpu-common
#SBATCH --gres=gpu:1
#SBATCH --exclusive

source ~/.bashrc
conda activate wildfire-ai

python generate_gp.py --city Atlanta --resolution 300 --num_points 4
