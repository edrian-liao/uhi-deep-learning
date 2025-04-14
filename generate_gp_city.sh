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

python generate_gp.py --city "Atlanta" --resolution 500 --num_points 4
python generate_gp.py --city "Baltimore_A" --resolution 500 --num_points 4
python generate_gp.py --city "Baltimore_B" --resolution 500 --num_points 4
python generate_gp.py --city "Boston" --resolution 500 --num_points 4
python generate_gp.py --city "Chicago" --resolution 500 --num_points 4
python generate_gp.py --city "Detroit" --resolution 500 --num_points 4
python generate_gp.py --city "Durham" --resolution 500 --num_points 4
python generate_gp.py --city "Houston" --resolution 500 --num_points 4
python generate_gp.py --city "Kansas" --resolution 500 --num_points 4
python generate_gp.py --city "Las Vegas" --resolution 500 --num_points 4
python generate_gp.py --city "Los Angeles" --resolution 500 --num_points 4
python generate_gp.py --city "Miami" --resolution 500 --num_points 4
python generate_gp.py --city "Nashville" --resolution 500 --num_points 4
python generate_gp.py --city "New Orleans" --resolution 500 --num_points 4
python generate_gp.py --city "NYC" --resolution 500 --num_points 4
python generate_gp.py --city "Oklahoma" --resolution 500 --num_points 4
python generate_gp.py --city "Raleigh" --resolution 500 --num_points 4
python generate_gp.py --city "San Francisco" --resolution 500 --num_points 4
python generate_gp.py --city "Seattle" --resolution 500 --num_points 4
python generate_gp.py --city "Washington DC" --resolution 500 --num_points 4
