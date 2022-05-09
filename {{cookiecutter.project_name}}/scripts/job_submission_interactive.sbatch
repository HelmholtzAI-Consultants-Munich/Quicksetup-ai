#!/bin/bash

#SBATCH -o slurm_out_%j.job
#SBATCH -e slurm_error_%j.job
#SBATCH -p interactive_gpu_p 
#SBATCH --gres=gpu:1
#SBATCH --qos interactive_gpu
#SBATCH -c 3
#SBATCH --mem=5G
#SBATCH -t 00:30:00
#SBATCH --nice=10000

source $HOME/.bashrc
echo 'Starting script'

conda activate ml_template_env

python train.py