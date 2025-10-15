#!/bin/bash

# Pipeline execution script for uncertainty analysis
# This script runs segmentation and uncertainty analysis with timestamped output directories

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Generate datetime stamp
DATETIME_STAMP=$(date +"%Y%m%d_%H%M%S")
echo "Using datetime stamp: $DATETIME_STAMP"

# Create results directory structure
mkdir -p "results/${DATETIME_STAMP}/seg"
mkdir -p "results/${DATETIME_STAMP}/skel_uncertainty"

echo "=== Step 1: Navigate to segmentation_unet3d directory ==="
cd segmentation_unet3d

echo "=== Step 2: Modify segmentation config.json ==="
python3 ../modify_config.py segmentation config.json "$DATETIME_STAMP"

echo "=== Step 3: Run segmentation ==="
CUDA_VISIBLE_DEVICES=0 python3 test_unet_3D.py --params config.json

echo "=== Step 4: Navigate to uncertainty directory ==="
cd ../uncertainty

echo "=== Step 5: Modify uncertainty config.json ==="
python3 ../modify_config.py uncertainty config.json "$DATETIME_STAMP"

echo "=== Step 6: Run uncertainty analysis ==="
CUDA_VISIBLE_DEVICES=0 python3 test.py --params config.json

echo "=== Step 7: Run centerline to vessel conversion ==="
python3 cl_to_vessel.py

echo "=== Pipeline completed successfully! ==="
echo "Results saved in: results/${DATETIME_STAMP}/final_uncertainty/"
