#!/bin/bash

#SBATCH --mail-user=edrianpaul.liao@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./gen_gp.out
#SBATCH --error=./gen_gp.err
#SBATCH --mem=64G

source ~/.bashrc
conda activate wildfire-ai

# Array of city names based on processed CSV filenames
cities=(
  Albuquerque
  Atlanta
  Baltimore_A
  Baltimore_B
  Boston
  Boulder
  Chicago
  Detroit
  Durham
  Houston
  Kansas
  "Las Vegas"
  "Los Angeles"
  Miami
  Nashville
  "New Orleans"
  NYC
  Oklahoma
  Raleigh
  "San Francisco"
  Seattle
  "Washington DC"
)

# Loop through each city
for city in "${cities[@]}"
do
  echo "🚀 Starting GP generation for: $city"
  python generate_gp.py --city "$city" --resolution 500 --num_points 9
done
