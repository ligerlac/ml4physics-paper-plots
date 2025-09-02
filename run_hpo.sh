#!/bin/bash

# HPO Configuration and Submission Script
# Define ALL parameters here - this is the single source of truth

# =============================================================================
# PARAMETER CONFIGURATION (Edit these as needed)
# =============================================================================
THRESHOLDS=("0" "1" "0 1" "1 2" "2 3" "0 1 2")
NKERNELS=(64 128 256)
STRIDES=(1 2)
TREEDEPTHS=(3 4)

# SLURM Configuration
JOB_NAME="model_training"
TIME_LIMIT="1:00:00"
CPUS_PER_TASK=4
MEMORY="2G"
GPUS=1

# Paths
BASE_OUTPUT_DIR="/scratch/network/lg0508/lgn/HPO"
PYTHON_SCRIPT="train-clgn-model.py"
# =============================================================================

# Calculate array size
TOTAL_COMBINATIONS=$((${#THRESHOLDS[@]} * ${#NKERNELS[@]} * ${#STRIDES[@]} * ${#TREEDEPTHS[@]}))
MAX_ARRAY_INDEX=$((TOTAL_COMBINATIONS - 1))

echo "========== HPO CAMPAIGN CONFIGURATION =========="
echo "Parameter space:"
echo "  Thresholds: ${#THRESHOLDS[@]} options: (${THRESHOLDS[*]})"
echo "  N-kernels: ${#NKERNELS[@]} options: (${NKERNELS[*]})"
echo "  Strides: ${#STRIDES[@]} options: (${STRIDES[*]})"
echo "  Tree depths: ${#TREEDEPTHS[@]} options: (${TREEDEPTHS[*]})"
echo "  Total combinations: $TOTAL_COMBINATIONS"
echo "  Array range: 0-$MAX_ARRAY_INDEX"
echo "================================================="

# Function to generate the SLURM script content
generate_slurm_script() {
    cat << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=slurm_logs/job_%A_%a.out
#SBATCH --error=slurm_logs/job_%A_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEMORY
#SBATCH --gres=gpu:$GPUS
#SBATCH --array=0-$MAX_ARRAY_INDEX

# Environment setup
echo "========== ENVIRONMENT SETUP =========="
echo "Loading required modules..."
module load anaconda3/2024.6
module load cudatoolkit/12.6

echo "Activating conda environment: difflogic"
conda activate difflogic

echo "Python path: \$(which python)"
echo "Python version: \$(python --version)"
echo "Conda environment: \$CONDA_DEFAULT_ENV"
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

# Parameter arrays (auto-generated from config)
thresholds=($(printf '"%s" ' "${THRESHOLDS[@]}"))
nkernels=(${NKERNELS[@]})
strides=(${STRIDES[@]})
treedepths=(${TREEDEPTHS[@]})

# Calculate indices
treedepth_idx=\$((SLURM_ARRAY_TASK_ID % \${#treedepths[@]}))
remainder=\$((SLURM_ARRAY_TASK_ID / \${#treedepths[@]}))

stride_idx=\$((remainder % \${#strides[@]}))
remainder=\$((remainder / \${#strides[@]}))

nkernel_idx=\$((remainder % \${#nkernels[@]}))
remainder=\$((remainder / \${#nkernels[@]}))

threshold_idx=\$((remainder % \${#thresholds[@]}))

# Get parameter values
threshold=\${thresholds[\$threshold_idx]}
nkernel=\${nkernels[\$nkernel_idx]}
stride=\${strides[\$stride_idx]}
treedepth=\${treedepths[\$treedepth_idx]}

# Output directory
CAMPAIGN_ID="\${SLURM_ARRAY_JOB_ID}"
OUTPUT_DIR="$BASE_OUTPUT_DIR/campaign_\${CAMPAIGN_ID}/t_\${threshold// /_}_k_\${nkernel}_s_\${stride}_d_\${treedepth}"
mkdir -p "\$OUTPUT_DIR"

echo "========== EXPERIMENT PARAMETERS =========="
echo "Running experiment with parameters:"
echo "  threshold: '\$threshold'"
echo "  nkernel: \$nkernel"
echo "  stride: \$stride"
echo "  treedepth: \$treedepth"
echo "  Output directory: \$OUTPUT_DIR"
echo "  Array indices: threshold=\$threshold_idx, nkernel=\$nkernel_idx, stride=\$stride_idx, treedepth=\$treedepth_idx"
echo "==========================================="

# Run Python script
python $PYTHON_SCRIPT \\
    --num-iterations 10000 \\
    --batch-size 128 \\
    --eval-freq 100 \\
    --save-freq 500 \\
    --learning-rate 0.05 \\
    --device cuda \\
    --thresholds \$threshold \\
    --n-kernels "\$nkernel" \\
    --stride "\$stride" \\
    --tree-depth "\$treedepth" \\
    --output "\$OUTPUT_DIR"

# Capture exit status and log
EXIT_STATUS=\$?

echo "========== EXPERIMENT COMPLETION =========="
if [ \$EXIT_STATUS -eq 0 ]; then
    echo "SUCCESS: Experiment completed successfully"
    log_message="SUCCESS - threshold='\$threshold', nkernel=\$nkernel, stride=\$stride, treedepth=\$treedepth, Job ID: \${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}, Output: \$OUTPUT_DIR"
else
    echo "FAILED: Experiment failed with exit status \$EXIT_STATUS"
    log_message="FAILED - threshold='\$threshold', nkernel=\$nkernel, stride=\$stride, treedepth=\$treedepth, Job ID: \${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}, Exit Status: \$EXIT_STATUS"
fi

# Thread-safe logging
{
    flock -x 200
    echo "\$(date '+%Y-%m-%d %H:%M:%S') - \$log_message" >> "$BASE_OUTPUT_DIR/experiment_log.txt"
} 200>>"$BASE_OUTPUT_DIR/experiment_log.txt.lock"

echo "Experiment completed with status: \$EXIT_STATUS"
echo "==========================================="

exit \$EXIT_STATUS
EOF
}

# Ask for confirmation
read -p "Generate and submit $TOTAL_COMBINATIONS HPO jobs? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Generate the SLURM script
    SLURM_FILE="submission-file.slurm"
    generate_slurm_script > "$SLURM_FILE"
    chmod +x "$SLURM_FILE"
    
    echo "Generated SLURM script: $SLURM_FILE"
    echo "Submitting job array..."
    
    sbatch "$SLURM_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✅ Job array submitted successfully!"
        echo "Monitor with: squeue -u \$USER"
        echo "Generated script saved as: $SLURM_FILE"
    else
        echo "❌ Submission failed"
        exit 1
    fi
else
    echo "Submission cancelled"
    echo "You can run this script anytime to generate and submit the HPO campaign"
fi
