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

source /beegfs/ga4493/projects/groupb/environments/torch/bin/activate
python main.py ${1} ${2} ${3}
chgrp -R ga4493b ./*
chmod -R g+rwx ./*
