# Structure-wise Uncertainty for 3D Segmentation

This repository provides code to compute structure-wise uncertainty for 3D volumetric data (e.g., .tiff stacks).
Sample datasets for testing and verification will be made available soon.

The implementation is based on the inference pipeline described in the NeurIPS 2023 paper:
[Topology-Aware Uncertainty for Image Segmentation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/19ded4cfc36a7feb7fce975393d378fd-Abstract-Conference.html)

For reference, the original source code can be found in the official repository:
[GitHub.com/Saumya-Gupta-26/struct-uncertainty](https://github.com/Saumya-Gupta-26/struct-uncertainty/)

## Table of Contents

- [1. Environment Setup](#1-environment-setup)
- [2. Running the Pipeline](#2-running-the-pipeline)
- [3. Notes](#3-notes)
- [4. Citation](#4-citation)
- [5. Contact](#5-contact)

## 1. Environment Setup

### 1.1) Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- CUDA-compatible GPU (recommended for optimal performance)
- CMake (for building DIPHA)

### 1.2) Use environment.yml

```bash
# Clone or download this repository
cd uncertainty-code-share

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate mito
```

**Note:** The provided `environment.yml` includes PyTorch with CUDA 12.1 support, which was compatible with the development system. If your system has a different CUDA version, you may need to modify the PyTorch installation in the environment file or install PyTorch separately with the appropriate CUDA version for your system. Check your CUDA version with `nvidia-smi` and visit [PyTorch's installation page](https://pytorch.org/get-started/locally/) for the correct installation command.

### 1.3) Verification

Test your installation:

```python
import torch
import numpy as np

# Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

### 1.4) Building dependency: DIPHA (Distributed Persistent Homology Algorithm)

DIPHA is required for topological analysis. Build it in the `dipha-graph-recon/` folder:

```bash
cd dipha-graph-recon

# Clean any existing build
rm -rf build/

# Create and enter build directory
mkdir build
cd build

# Configure and build
cmake ..
make
```

## 2. Running the Pipeline

### 2.1) Data Preparation

1. **Place your input images** in the `data/` directory
2. **Update the sample list** in `test-samples.csv` with the images you want to process
3. **Ensure your data format** is compatible with the pipeline (the current code is for TIFF file format)

### 2.2) Automated Pipeline Execution

The repository includes an automated pipeline script that handles the entire workflow. You might need to give following files execute rights beforehand:

```bash
# Give execute rights
chmod +x dipha-graph-recon/src/loop.out
chmod +x dipha-graph-recon/src/manifold.out
chmod +x dipha-graph-recon/build/dipha
chmod +x run_pipeline.sh

# Activate the environment
conda activate mito

# Run the complete pipeline
./run_pipeline.sh
```

### 2.3) Output Structure

The uncertainty results are organized by timestamp to prevent overwriting. They can be found in the `results/YYYYMMDD_HHMMSS/final_uncertainty` folder. The output is stored in both `.npy` and `.tiff` formats.

```
results/
└── YYYYMMDD_HHMMSS/
    └── final_uncertainty/
        ├── *.npy           # NumPy arrays for programmatic access
        ├── *.tiff          # TIFF images for visualization
```

Each run creates a unique timestamped directory.


## 3. Notes

- The pipeline requires one GPU
- Processing time depends on image size and complexity. Some commands in the bash script `run_pipeline.sh` can take 15-20 minutes each
- Results include both`.npy` and `.tiff` formats
- Each run is timestamped to maintain result history


## 4. Citation
The citation for this work is
```
@article{gupta2024topology,
  title={Topology-aware uncertainty for image segmentation},
  author={Gupta, Saumya and Zhang, Yikai and Hu, Xiaoling and Prasanna, Prateek and Chen, Chao},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

## 5. Contact
For any issues, please email saumgupta@cs.stonybrook.edu