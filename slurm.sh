#!/bin/bash

#SBATCH --job-name=flwr-pwr
#SBATCH --output=./logs/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:3
#SBATCH --time=1-00:00:00

source ~/.bashrc
source venv/bin/activate

flwr run . standard-gpuq
