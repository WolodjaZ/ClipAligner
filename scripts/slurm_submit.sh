#!/bin/bash

#!/usr/bin/env bash
#SBATCH --account=???
#SBATCH --job-name=ClipAlign
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=???

# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
srun python src/train.py