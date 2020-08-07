#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=a_rate%j
#SBATCH --mail-type=END
#SBATCH --mail-user=ls3817@nyu.edu
#SBATCH --output=slurm_%j.out
  
module purge

VIRTUALENV=$SCRATCH
RUNDIR=$SCRATCH/low_latency_streaming/rate_adaption_torch

cd $VIRTUALENV

source ./torch/bin/activate
  
cd $RUNDIR
python main.py -a
