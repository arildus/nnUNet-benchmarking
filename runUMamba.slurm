#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=gpu32g
#SBATCH --job-name="nnunet_benchmarking"
#SBATCH --output=nnunet_benchmarking.out


cd ${SLURM_SUBMIT_DIR}/

module purge
module load Anaconda3/2024.02-1
module load CUDA/12.1.1
conda activate nnUNet_env

export CUDA_VISIBLE_DEVICES=0
export nnUNet_n_proc_DA=180

export nnUNet_raw=content/data/nnUNet_raw_data_base/
export nnUNet_preprocessed=content/data/nnUNet_preprocessed/
export nnUNet_results=content/data/nnUNet_results/

# srun python trainer.py # srun python trainer.py
srun nnUNetv2_train 137 3d_fullres 0 -tr nnUNetTrainerUMambaBot