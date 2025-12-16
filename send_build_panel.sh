#!/bin/bash
#
#SBATCH --job-name=rob_joint

#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=ncpu
#SBATCH --mail-user=colasa@crick.ac.uk
#SBATCH --mail-type=END,FAIL
ml purge

ml Anaconda3/2022.05

source activate base

conda activate spapros

cd /nemo/lab/znamenskiyp/home/users/colasa/code/spapros

python /nemo/lab/znamenskiyp/home/users/colasa/code/spapros/build_panel_2021_rob.py --n_jobs 32