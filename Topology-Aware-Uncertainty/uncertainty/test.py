"""
3D Structure-wise Uncertainty Quantification Testing Module

This module implements testing and inference for 3D structure-wise uncertainty quantification
using Discrete Morse Theory (DMT) and deep learning models. It processes 3D
medical images in cubes, applies uncertainty models, and reconstructs full
uncertainty heatmaps.

Key functionalities:
- 3D volume processing in non-overlapping cubes
- Monte Carlo sampling for uncertainty estimation
- Uncertainty heatmap reconstruction from cube predictions
- Integration with DMT-based topological analysis
- Support for large 3D volumes through cube-based processing

The module handles large 3D volumes by splitting them into manageable cubes,
processing each cube independently, and then reconstructing the full volume
from the processed cubes.
"""

import torch
import numpy as np
import argparse, json
import os, glob, sys, shutil
from time import time
from tqdm import tqdm

from dataloader import Dataset3D  
from dmt_trainer import getData_val, reconstruct_uncertainty_heatmap
from unc_model import UncertaintyModel

def parse_func(args):
    """
    Parse command line arguments and configuration file.
    
    This function reads the parameters file and extracts all necessary
    configuration parameters for the uncertainty quantification testing.
    
    Args:
        args: Command line arguments containing the path to parameters file
        
    Returns:
        dict: Dictionary containing all configuration parameters
    """
    print("Reading params file {}...".format(args.params))
    with open(args.params, 'r') as f:
        params = json.load(f)

    # Extract and organize configuration parameters
    mydict = {}
    mydict['num_classes'] = int(params['num_classes'])
    mydict['uncmodel_checkpoint_restore'] = params['uncmodel_checkpoint_restore']    
    mydict['test_datalist'] = params['test_datalist']
    mydict['npy_seg_dir'] = params['npy_seg_dir']
    mydict['MCSamples'] = int(params['MCSamples'])
    mydict['output_folder'] = params['output_folder']
    mydict['cube_size'] = params.get('cube_size', 64)  # Default cube size is 64
    mydict['image_dir'] = params['image_dir']

    return mydict

def process_volume_in_cubes(volume, cube_size=64):
    """
    Split 3D volume into non-overlapping cubes with special handling for borders.
    
    This function divides a 3D volume into cubes of specified size, ensuring that
    all regions of the volume are covered, including border regions that don't
    fit perfectly into the cube size. It handles all possible border and corner
    cases to ensure complete coverage.
    
    Args:
        volume (torch.Tensor): 3D volume tensor with shape (N, C, D, H, W)
        cube_size (int): Size of each cube (default: 64)
        
    Returns:
        tuple: (cubes, coords) where:
            - cubes: List of cube tensors
            - coords: List of coordinate tuples (d_start, d_end, h_start, h_end, w_start, w_end)
    """
    D, H, W = volume.shape[2:]  # Extract spatial dimensions from NCHW format
    cubes = []
    coords = []
    
    # Process regular non-overlapping cubes
    for d in range(0, D - cube_size + 1, cube_size):
        for h in range(0, H - cube_size + 1, cube_size):
            for w in range(0, W - cube_size + 1, cube_size):
                cube = volume[:, :, d:d+cube_size, h:h+cube_size, w:w+cube_size]
                cubes.append(cube)
                coords.append((d, d+cube_size, h, h+cube_size, w, w+cube_size))
    
    # Handle border regions that don't fit perfectly into cube size
    # D border (depth dimension)
    if D % cube_size != 0:
        d_start = D - cube_size
        for h in range(0, H - cube_size + 1, cube_size):
            for w in range(0, W - cube_size + 1, cube_size):
                cube = volume[:, :, d_start:D, h:h+cube_size, w:w+cube_size]
                cubes.append(cube)
                coords.append((d_start, D, h, h+cube_size, w, w+cube_size))
    
    # H border (height dimension)
    if H % cube_size != 0:
        h_start = H - cube_size
        for d in range(0, D - cube_size + 1, cube_size):
            for w in range(0, W - cube_size + 1, cube_size):
                cube = volume[:, :, d:d+cube_size, h_start:H, w:w+cube_size]
                cubes.append(cube)
                coords.append((d, d+cube_size, h_start, H, w, w+cube_size))
    
    # W border (width dimension)
    if W % cube_size != 0:
        w_start = W - cube_size
        for d in range(0, D - cube_size + 1, cube_size):
            for h in range(0, H - cube_size + 1, cube_size):
                cube = volume[:, :, d:d+cube_size, h:h+cube_size, w_start:W]
                cubes.append(cube)
                coords.append((d, d+cube_size, h, h+cube_size, w_start, W))
    
    # Handle corner cases where multiple dimensions don't fit perfectly
    # D-H corner
    if D % cube_size != 0 and H % cube_size != 0:
        d_start = D - cube_size
        h_start = H - cube_size
        for w in range(0, W - cube_size + 1, cube_size):
            cube = volume[:, :, d_start:D, h_start:H, w:w+cube_size]
            cubes.append(cube)
            coords.append((d_start, D, h_start, H, w, w+cube_size))
    
    # D-W corner
    if D % cube_size != 0 and W % cube_size != 0:
        d_start = D - cube_size
        w_start = W - cube_size
        for h in range(0, H - cube_size + 1, cube_size):
            cube = volume[:, :, d_start:D, h:h+cube_size, w_start:W]
            cubes.append(cube)
            coords.append((d_start, D, h, h+cube_size, w_start, W))
    
    # H-W corner
    if H % cube_size != 0 and W % cube_size != 0:
        h_start = H - cube_size
        w_start = W - cube_size
        for d in range(0, D - cube_size + 1, cube_size):
            cube = volume[:, :, d:d+cube_size, h_start:H, w_start:W]
            cubes.append(cube)
            coords.append((d, d+cube_size, h_start, H, w_start, W))
    
    # D-H-W corner (all three dimensions don't fit perfectly)
    if D % cube_size != 0 and H % cube_size != 0 and W % cube_size != 0:
        d_start = D - cube_size
        h_start = H - cube_size
        w_start = W - cube_size
        cube = volume[:, :, d_start:D, h_start:H, w_start:W]
        cubes.append(cube)
        coords.append((d_start, D, h_start, H, w_start, W))
    
    return cubes, coords

def reconstruct_from_cubes(cubes, coords, full_shape):
    """
    Reconstruct full volume from processed cubes.
    
    This function takes a list of processed cubes and their coordinates and
    reconstructs the full 3D volume. Border cubes take precedence over regular
    cubes in overlapping regions to ensure proper coverage.
    
    Args:
        cubes (list): List of processed cube arrays
        coords (list): List of coordinate tuples for each cube
        full_shape (tuple): Target shape of the full volume (D, H, W)
        
    Returns:
        numpy.ndarray: Reconstructed full volume with shape full_shape
    """
    D, H, W = full_shape
    full_volume = np.zeros((D, H, W))
    
    # Place each cube in its corresponding location
    # Border cubes overwrite regular cubes in overlapping regions
    for cube, (d_start, d_end, h_start, h_end, w_start, w_end) in zip(cubes, coords):
        full_volume[d_start:d_end, h_start:h_end, w_start:w_end] = cube
    
    return full_volume

if __name__ == "__main__":
    """
    Main execution block for 3D uncertainty quantification testing.
    
    This script performs uncertainty quantification on 3D medical images by:
    1. Loading test data and uncertainty model
    2. Processing volumes in cubes to handle large datasets
    3. Applying Monte Carlo sampling for uncertainty estimation
    4. Reconstructing full uncertainty heatmaps
    5. Saving results for analysis
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the parameters file")
    parser.add_argument('--dataset', type=str, default="YourDataset")

    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")
    else:
        args = parser.parse_args()
        mydict = parse_func(args)

    print("3D Inference!")

    # Set up device (prefer CUDA if available)
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    # Create output directories
    os.makedirs(mydict['output_folder'], exist_ok=True)

    # Initialize test dataset and data loader
    test_set = Dataset3D(mydict['image_dir'], mydict['npy_seg_dir'], mydict['test_datalist'])
    n_channels = 1  # Typically 1 for 3D medical images
    in_channels = 3  # Adjusted for 3D (image + likelihood + path channels)
    
    test_generator = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )
    
    # Initialize uncertainty model
    binary_classifier = UncertaintyModel(
        in_channels=in_channels,
        num_features=36,
        hidden_units=48
    ).float().to(device)

    # Load pre-trained model checkpoint
    if mydict['uncmodel_checkpoint_restore']:
        checkpoint = torch.load(mydict['uncmodel_checkpoint_restore'], map_location=device)
        binary_classifier.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"Loaded uncertainty model checkpoint: {mydict['uncmodel_checkpoint_restore']}")
    else:
        print("No uncertainty model found!")
        sys.exit()

    test_start_time = time()

    # Perform inference on test data
    with torch.no_grad():
        binary_classifier.train()  # Enable dropout for Monte Carlo sampling

        for i, (volume, likelihood, filename) in enumerate(tqdm(test_generator)):
            # Get original volume dimensions
            original_shape = volume.shape[2:]  # NCDHW format
            
            # Split volume into cubes for processing
            volume_cubes, coords = process_volume_in_cubes(volume, mydict['cube_size'])
            likelihood_cubes, _ = process_volume_in_cubes(likelihood, mydict['cube_size'])
            
            # Process each cube independently
            processed_cubes = []
            mycnt = 0
            for vol_cube, like_cube in zip(volume_cubes, likelihood_cubes):
                mycnt += 1
                print(f"Processing cube {mycnt}/{len(volume_cubes)}")
                vol_cube = vol_cube.to(device)
                like_cube = like_cube.to(device)
                
                # Extract manifold features using DMT analysis
                imgbatch, unc_input = getData_val(
                    mydict['num_classes'],
                    vol_cube,
                    like_cube
                )

                if unc_input is not None:
                    imgbatch = imgbatch.float().to(device)
                    unc_input = unc_input.float().to(device)

                    # Monte Carlo sampling for uncertainty estimation
                    unc_pred_mu = []
                    unc_pred_logvar = []
                    for _ in range(mydict['MCSamples']):
                        temp = binary_classifier(imgbatch, unc_input)  # Returns (mu, log_var)
                        unc_pred_mu.append(torch.squeeze(temp[0], dim=1).detach().cpu().numpy()) 
                        unc_pred_logvar.append(torch.squeeze(temp[1], dim=1).detach().cpu().numpy())

                    print(f"Lengths: {len(unc_pred_mu)}, {len(unc_pred_logvar)}")
                    print(f"Uncertainty prediction: {unc_pred_mu[-1].shape}, {unc_pred_logvar[-1].shape}")

                    # Reconstruct uncertainty map for this cube
                    cube_map = reconstruct_uncertainty_heatmap(
                        os.path.join(mydict['output_folder'], os.path.basename(filename[0]).replace('.tiff', '')),
                        unc_pred_mu, 
                        unc_pred_logvar,
                        (mydict['cube_size'],)*3,  # cube dimensions
                        f"{os.path.basename(filename[0])}_cube_{len(processed_cubes)}"
                    )
                    processed_cubes.append(cube_map)

            # Reconstruct full volume from processed cubes
            full_uncertainty_map = reconstruct_from_cubes(processed_cubes, coords, original_shape)

            # Save results
            output_path = os.path.join(mydict['output_folder'], f'{os.path.basename(filename[0])}_uncertainty_3d.npy')
            np.save(output_path, np.clip(full_uncertainty_map, 0., 1.))
            print(f"Processed {os.path.basename(filename[0])}, saved to {output_path}")

            test_end_time = time()
            print(f"Processing took {(test_end_time - test_start_time)/60:.2f} minutes")