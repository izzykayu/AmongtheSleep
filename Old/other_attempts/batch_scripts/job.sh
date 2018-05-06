#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-tas=2
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=myImageLoader
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=grivam01@nyu.edu

module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
source ~/scripts/deep_learning_hw2/dl_env/bin/activate

#python imageloader.py
#python subset.py
#python deep_learning.py 
python model_loader.py
