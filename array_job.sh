#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem=4GB
#SBATCH --array=2-2
#SBATCH --job-name=DGCN-nf_rank-b
#SBATCH --mail-type=END
#SBATCH --mail-user=npj226@nyu.edu
#SBATCH --output=slurm_out/DGCN-nf_rank-%ab.out
#SBATCH --error=slurm_out/DGCN-nf_rank-%ab.err
  
module load cuda/10.1.105
module load cudnn/10.1v7.6.5.32

source /home/npj226/py3env/bin/activate
cd /scratch/npj226/DGCN

python train.py args/DGCN-nf_rank-${SLURM_ARRAY_TASK_ID}b.json
