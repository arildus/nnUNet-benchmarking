#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --job-name="nnunet_benchmarking"
#SBATCH --output=nnunet_benchmarking.out


cd ${SLURM_SUBMIT_DIR}/

module purge
module load Anaconda3/2024.02-1
conda activate nnunet_env
set CUDA_VISIBLE_DEVICES=0
set nnUNet_raw=./content/data/nnUNet_raw_data_base/
set nnUNet_preprocessed=./content/data/nnUNet_preprocessed/
set nnUNet_results=./content/data/nnUNet_results/

# srun python trainer.py # srun python trainer.py
srun nnUNetv2_train 137 3d_fullres 0 -tr nnUNetTrainerUMambaBot