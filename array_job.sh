#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00
#SBATCH --mem=4GB
#SBATCH --array=0-3
#SBATCH --job-name=GCDLNet-nf_topK-b
#SBATCH --mail-type=END
#SBATCH --mail-user=npj226@nyu.edu
#SBATCH --output=slurm_out/GCDLNet-nf_topK-%ab.out
#SBATCH --error=slurm_out/GCDLNet-nf_topK-%ab.err
  
module load cuda/10.1.105
module load cudnn/10.1v7.6.5.32

source /home/npj226/py3env/bin/activate
cd /scratch/npj226/DGCN

python train.py args/GCDLNet-nf_topK-${SLURM_ARRAY_TASK_ID}b.json
