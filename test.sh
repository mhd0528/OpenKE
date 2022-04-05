#!/bin/bash
#SBATCH --job-name=OpenKE_ComplEx_Test
#SBATCH --output=/blue/daisyw/ma.haodi/OpenKE/logs/Complex_nne_aer-FB237-0_shot-12.07.out
#SBATCH --error=/blue/daisyw/ma.haodi/OpenKE/logs/Complex_nne_aer-FB237-0_shot-12.07.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ma.haodi@ufl.edu
#SBATCH --nodes=1 # nodes allocated to the job
#SBATCH --ntasks=1 # one process per node
#SBATCH --cpus-per-task=1 # the number of CPUs allocated per task, i.e., number of threads in each process
#SBATCH --mem=5gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00               # Time limit hrs:min:sec
#SBATCH --qos=daisyw

module load singularity
echo "starting job"
T1=$(date +%s)
echo "$T1"
singularity exec --nv --bind /blue/daisyw/ma.haodi/OpenKE/:/home/OpenKE /blue/daisyw/ma.haodi/kbc_models_version1.8.sif python -u /home/OpenKE/train_complex_nne_aer_FB237.py
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
