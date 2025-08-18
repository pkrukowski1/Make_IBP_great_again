#!/bin/bash
#SBATCH --job-name=RAVEN_affine_arithmetic_training
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rtx4090

source scripts/main.sh

run_sweep_and_agent "scripts/training/mnist_dm_medium_eps_train_04_test_03/affine_arithmetic"