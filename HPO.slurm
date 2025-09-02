#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH --output=slurm_logs/job_%A_%a.out
#SBATCH --error=slurm_logs/job_%A_%a.err
#SBATCH --time=1:00:00        # Adjust time limit as needed
#SBATCH --cpus-per-task=4     # Adjust CPU count as needed
#SBATCH --mem=2G              # Adjust memory as needed
#SBATCH --gres=gpu:1          # Request 1 GPU
# Note: The array size will be calculated and set dynamically below
# Placeholder for now - will be overridden by the submission wrapper

# Print detailed GPU/driver information for debugging
# Environment setup
echo "========== ENVIRONMENT SETUP =========="
echo "Loading required modules..."
module load anaconda3/2024.6
module load cudatoolkit/12.6

echo "Activating conda environment: difflogic"
conda activate difflogic

echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "========================================"

echo "========== HOST GPU INFORMATION =========="
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi is available on the host"
    echo "Full nvidia-smi output:"
    nvidia-smi
    echo "NVIDIA driver version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
    echo "CUDA version:"
    nvidia-smi --query-gpu=cuda_version --format=csv,noheader
else
    echo "nvidia-smi command not found on the host"
    echo "Checking for GPU devices:"
    ls -la /dev/nvidia* 2>/dev/null || echo "No NVIDIA device files found"
    echo "Checking CUDA installation:"
    ls -la /usr/local/cuda* 2>/dev/null || echo "No CUDA installation found in /usr/local"
fi
echo "==========================================="

# Create logs directory
mkdir -p slurm_logs

# Define the parameter values
thresholds=("0" "1" "0 1" "1 2" "2 3" "0 1 2" "1 2 3" "1 2 4" "1 2 4 8" "1 2 4 8 16")
nkernels=(64 128 256)
strides=(1 2)
treedepths=(3 4)

# Define the parameter values
thresholds=("0" "1" "0 1" "1 2" "2 3" "0 1 2")
nkernels=(64 128 256)
strides=(1 2)
treedepths=(3 4)

# Calculate total combinations automatically
total_combinations=$((${#thresholds[@]} * ${#nkernels[@]} * ${#strides[@]} * ${#treedepths[@]}))
max_array_index=$((total_combinations - 1))

echo "========== PARAMETER SPACE =========="
echo "Thresholds: ${#thresholds[@]} options: (${thresholds[*]})"
echo "N-kernels: ${#nkernels[@]} options: (${nkernels[*]})"
echo "Strides: ${#strides[@]} options: (${strides[*]})"
echo "Tree depths: ${#treedepths[@]} options: (${treedepths[*]})"
echo "Total parameter combinations: $total_combinations (array indices 0-$max_array_index)"
echo "Current SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Validate that the current task ID is within bounds
if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_combinations" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds maximum index ($max_array_index)"
    echo "Please set --array=0-$max_array_index when submitting this job"
    exit 1
fi
echo "====================================="

# Calculate indices based on SLURM_ARRAY_TASK_ID
# Using nested indexing: task_id = ((threshold * nkernels + nkernel) * strides + stride) * treedepths + treedepth
treedepth_idx=$((SLURM_ARRAY_TASK_ID % ${#treedepths[@]}))
remainder=$((SLURM_ARRAY_TASK_ID / ${#treedepths[@]}))

stride_idx=$((remainder % ${#strides[@]}))
remainder=$((remainder / ${#strides[@]}))

nkernel_idx=$((remainder % ${#nkernels[@]}))
remainder=$((remainder / ${#nkernels[@]}))

threshold_idx=$((remainder % ${#thresholds[@]}))

# Get parameter values
threshold=${thresholds[$threshold_idx]}
nkernel=${nkernels[$nkernel_idx]}
stride=${strides[$stride_idx]}
treedepth=${treedepths[$treedepth_idx]}

# Base output directory
BASE_OUTPUT_DIR="/scratch/network/lg0508/lgn/HPO"

# Use SLURM_ARRAY_JOB_ID as the campaign identifier (consistent across all array tasks)
# This ensures all jobs in the same array submission use the same folder
CAMPAIGN_ID="${SLURM_ARRAY_JOB_ID}"
OUTPUT_DIR="$BASE_OUTPUT_DIR/campaign_${CAMPAIGN_ID}/t_${threshold// /_}_k_${nkernel}_s_${stride}_d_${treedepth}"
mkdir -p "$OUTPUT_DIR"

echo "========== EXPERIMENT PARAMETERS =========="
echo "Running experiment with parameters:"
echo "  threshold: '$threshold'"
echo "  nkernel: $nkernel"
echo "  stride: $stride"
echo "  treedepth: $treedepth"
echo "  Output directory: $OUTPUT_DIR"
echo "  Array indices: threshold=$threshold_idx, nkernel=$nkernel_idx, stride=$stride_idx, treedepth=$treedepth_idx"
echo "==========================================="

# Run your Python script with the parameters
python train-clgn-model.py \
    --num-iterations 10000 \
    --batch-size 128 \
    --eval-freq 100 \
    --save-freq 500 \
    --learning-rate 0.05 \
    --device cuda \
    --thresholds $threshold \
    --n-kernels "$nkernel" \
    --stride "$stride" \
    --tree-depth "$treedepth" \
    --output "$OUTPUT_DIR"

# Capture the exit status
EXIT_STATUS=$?

# Add logging
echo "========== EXPERIMENT COMPLETION =========="
if [ $EXIT_STATUS -eq 0 ]; then
    echo "SUCCESS: Experiment completed successfully"
    log_message="SUCCESS - threshold='$threshold', nkernel=$nkernel, stride=$stride, treedepth=$treedepth, Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}, Output: $OUTPUT_DIR"
else
    echo "FAILED: Experiment failed with exit status $EXIT_STATUS"
    log_message="FAILED - threshold='$threshold', nkernel=$nkernel, stride=$stride, treedepth=$treedepth, Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}, Exit Status: $EXIT_STATUS"
fi

# Thread-safe logging
{
    flock -x 200
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $log_message" >> "$BASE_OUTPUT_DIR/experiment_log.txt"
} 200>>"$BASE_OUTPUT_DIR/experiment_log.txt.lock"

echo "Experiment completed with status: $EXIT_STATUS"
echo "==========================================="

exit $EXIT_STATUS