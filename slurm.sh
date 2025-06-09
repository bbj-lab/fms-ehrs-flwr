#!/bin/bash

#SBATCH --job-name=flwr-pwr
#SBATCH --output=./logs/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:3
#SBATCH --time=1-00:00:00

source ~/.bashrc 2> /dev/null
source ".venv/bin/activate" 2> /dev/null

flwr run .
