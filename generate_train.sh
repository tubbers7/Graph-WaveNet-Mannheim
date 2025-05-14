#!/bin/bash
#SBATCH -p dev_gpu_4  # Use the dev_gpu_4_a100 partition with A100 GPUs dev_gpu_4, gpu_8, gpu_4
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 00:10:00            # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=10000             # Memory request (adjust as needed)
#SBATCH --gres=gpu:1           # Request 1 GPU (adjust if you need more)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)

module load devel/miniforge
module purge
module load compiler/gnu/13.3 #need to load explicitly otherwise tries to find gcc complier for pandas

conda activate /home/ma/ma_ma/ma_tofuchs/.conda/envs/GraphWaveEnv

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "Python version: $(python --version)"
echo "Environment: $(conda info --envs)"

module load devel/cuda/12.4

python ~/Graph-WaveNet-Mannheim/generate_training_data.py --seq_length_x 6 --seq_length_y 6 --output_dir /pfs/work9/workspace/scratch/ma_tofuchs-GraphWave-Seminar/Datasets/Mannheim/train_data --traffic_df_filename /pfs/work9/workspace/scratch/ma_tofuchs-GraphWave-Seminar/Datasets/Mannheim/imputed_dataset_grin_full.csv