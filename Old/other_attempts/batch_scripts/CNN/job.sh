#!/bin/bash
#!#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-tas=2
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=myImageLoader
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:p40:1
#SBATCH --mail-type=END
#SBATCH --mail-user=grivam01@nyu.edu

#module purge
source /beegfs/ga4493/projects/groupb/environments/torch/bin/activate
#module swap python3/intel  python/intel/2.7.12
#python imageloader.py
#python subset.py
#python deep_learning.py 
python main.py
chgrp -R ga4493b ./*
chmod -R g+rwx ./*
