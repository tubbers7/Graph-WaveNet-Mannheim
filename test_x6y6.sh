#!/bin/bash
#SBATCH -p dev_gpu_h100  # Use the dev_gpu_4_a100 partition with A100 GPUs dev_gpu_4, gpu_8, gpu_4
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 00:10:00            # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=10000             # Memory request (adjust as needed)
#SBATCH --gres=gpu:1           # Request 1 GPU (adjust if you need more)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)

#module load devel/miniforge
#module purge
#module load compiler/gnu/13.3 #need to load explicitly otherwise tries to find gcc complier for pandas

#conda activate /home/ma/ma_ma/ma_tofuchs/.conda/envs/GraphWaveEnv
source GraphWave/bin/activate

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "Python version: $(python --version)"
#echo "Environment: $(conda info --envs)"

module load devel/cuda/12.8



python ~/Graph-WaveNet-Mannheim/val.py \
  --device cuda:0 \
  --data /pfs/work9/workspace/scratch/ma_tofuchs-GraphWave-Seminar/Datasets/Mannheim/train_data \
  --adjdata /pfs/work9/workspace/scratch/ma_tofuchs-GraphWave-Seminar/Datasets/Mannheim/train_data/sensor_graph/adj_mx.csv \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --randomadj \
  --checkpoint /pfs/work9/workspace/scratch/ma_tofuchs-GraphWave-Seminar/models/mannheim_exp1_best_135.79.pth \
  --plotheatmap False \
  --num_nodes 25 \
  --seq_length 6 