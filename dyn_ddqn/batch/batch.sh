#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=ddqn
#SBATCH --mail-type=END
#SBATCH --mail-user=ls3817@nyu.edu
#SBATCH --output=slurm_%j.out
  
module purge

VIRTUALENV=$SCRATCH
RUNDIR=$SCRATCH/low_latency_streaming/infocom21_rate/dyn_ddqn

cd $VIRTUALENV

source ./torch/bin/activate
  
cd $RUNDIR
python main.py
