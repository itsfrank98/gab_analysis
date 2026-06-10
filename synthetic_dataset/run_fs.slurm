#!/bin/bash

#SBATCH --job-name=user_creation              # Descriptive job name
#SBATCH --time=00:20:00                       # Maximum wall time (hh:mm:ss)
#SBATCH --nodes=1                             # Number of nodes to use
#SBATCH --gres=gpu:2                          # Number of GPUs per node (adjust to match hardware)
#SBATCH --partition=boost_usr_prod           # GPU-enabled partition
#SBATCH --output=out_fs.out              # File for standard output
#SBATCH --error=err_fs.err               # File for standard error
#SBATCH --account=IscrC_GNG26          # Project account number

# Load necessary modules (adjust to your environment)
module purge
module load gcc/12.2.0
module load cuda/12.2                         # Load CUDA toolkit
module load python/3.11.7

source /leonardo/home/userexternal/fbenedet/user_creation/.venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
accelerate launch --num_processes 2 /leonardo/home/userexternal/fbenedet/user_creation/user_posts_creation.py --n_posts 10 --users_profiles_path synthetic_user_profiles.csv --output_fname synthetic_posts_snowpiercer.csv --model_path /leonardo_scratch/large/userexternal/fbenedet/models/snowpiercer --real_posts_path gab_posts_labeled_qwen.csv 

#/leonardo_scratch/large/userexternal/fbenedet/models/irix12b
