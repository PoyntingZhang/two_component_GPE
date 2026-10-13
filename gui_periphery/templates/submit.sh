#!/usr/bin/env bash
# Template for one GPU on CSD3 Wilkes3. Set the account and wall time before use.
# Compare with /usr/local/Cluster-Docs/SLURM/slurm_submit.wilkes3 on CSD3.
#SBATCH --job-name=GPE
#SBATCH --account=CHANGE_ME_GPU_ACCOUNT
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
# Submit using `sbatch submit.sh` from the prepared simulation folder.
cd -- "${SLURM_SUBMIT_DIR:?Submit from the simulation folder using sbatch}"
srun bash run.sh
