"""
3D UNet Inference Script for Medical Image Segmentation

This script performs inference on 3D medical images using a trained UNet model.
It uses a patch-based approach to handle large volumes that may not fit in GPU memory.
The script loads a trained model checkpoint and processes test volumes to generate
segmentation predictions.

Key Features:
- Patch-based inference for memory-efficient processing of large 3D volumes
- Overlapping patches with weighted averaging to reduce boundary artifacts
- Support for both float probability maps and binary segmentation outputs
- GPU acceleration with CUDA support
- Reproducible results through seed setting
"""

import torch
import numpy as np
import argparse, json
import os, glob, sys
from time import time
from scipy import ndimage
import imageio
import SimpleITK as sitk
from dataloader import Dataset3D
from model import UNet

def parse_func(args):
    """
    Parse configuration parameters from JSON file.
    
    This function reads a JSON configuration file containing model and inference
    parameters, validates the required fields, and creates the output directory
    if it doesn't exist.
    
    Args:
        args: Parsed command line arguments containing the path to the parameters file
        
    Returns:
        dict: Dictionary containing parsed configuration parameters:
            - num_classes (int): Number of output classes for segmentation
            - checkpoint_restore (str): Path to the trained model checkpoint
            - test_datalist (str): Path to file containing list of test images
            - image_dir (str): Directory containing input images
            - output_folder (str): Directory to save prediction results
            - crop_size (int): Size of patches for patch-based inference
            
    Raises:
        FileNotFoundError: If the parameters file doesn't exist
        KeyError: If required parameters are missing from the JSON file
        ValueError: If numeric parameters cannot be converted to integers
    """
    ### Reading the parameters json file
    print("Reading params file {}...".format(args.params))
    with open(args.params, 'r') as f:
        params = json.load(f)

    # Parse and validate required parameters
    mydict = {}
    mydict['num_classes'] = int(params['num_classes'])
    mydict["checkpoint_restore"] = params['checkpoint_restore']
    mydict['test_datalist'] = params['test_datalist']
    mydict['image_dir'] = params['image_dir']
    mydict['output_folder'] = params['output_folder']
    mydict['crop_size'] = int(params['crop_size'])

    print(mydict)

    # Create output directory if it doesn't exist
    if not os.path.exists(mydict['output_folder']):
        os.makedirs(mydict['output_folder'])

    return mydict

def set_seed():
    """
    Set random seeds for reproducible results.
    
    This function sets the random seed for PyTorch (CPU and CUDA), NumPy,
    and configures cuDNN for deterministic behavior. This ensures that
    inference results are reproducible across multiple runs.
    
    Note:
        Setting deterministic=True and benchmark=False may reduce performance
        but ensures reproducibility. For production inference where speed
        is more important than reproducibility, these can be adjusted.
    """
    seed = 0
    torch.manual_seed(seed)  # Set PyTorch CPU random seed
    torch.cuda.manual_seed_all(seed)  # Set PyTorch CUDA random seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic cuDNN operations
    torch.backends.cudnn.benchmark = False  # Disable cuDNN auto-tuner for reproducibility
    np.random.seed(seed)  # Set NumPy random seed

def force_cudnn_initialization():
    """
    Force cuDNN initialization to avoid first-batch slowdown.
    
    This function performs a dummy convolution operation to initialize cuDNN
    libraries and GPU context. This prevents the first inference batch from
    being significantly slower due to initialization overhead.
    
    The dummy operation uses small tensors to minimize memory usage while
    ensuring that all necessary GPU resources are initialized.
    """
    s = 32  # Size of dummy tensors
    dev = torch.device('cuda')
    # Perform dummy 2D convolution to initialize cuDNN
    torch.nn.functional.conv2d(
        torch.zeros(s, s, s, s, device=dev), 
        torch.zeros(s, s, s, s, device=dev)
    )

def test_3d(mydict):
    """
    Perform 3D segmentation inference on test volumes.
    
    This is the main inference function that processes 3D medical images using
    a trained UNet model. It uses a patch-based approach to handle large volumes
    that may exceed GPU memory limits.
    
    The function performs the following steps:
    1. Set up reproducible environment and GPU initialization
    2. Load test data using the custom Dataset3D loader
    3. Initialize the UNet model and load trained weights
    4. Process each volume using overlapping patches
    5. Stitch patches together with weighted averaging
    6. Save both probability maps (.npy) and binary masks (.tiff)
    
    Args:
        mydict (dict): Configuration dictionary containing:
            - num_classes: Number of output classes
            - checkpoint_restore: Path to model checkpoint
            - test_datalist: Path to test image list
            - image_dir: Directory containing test images
            - output_folder: Directory to save results
            - crop_size: Size of patches for inference
            
    Processing Details:
        - Uses overlapping patches with 25% step size for smooth boundaries
        - Applies sigmoid activation for probability outputs
        - Thresholds at 0.5 for binary segmentation
        - Saves float predictions as .npy and binary as .tiff
        
    Memory Management:
        - Processes one volume at a time to manage GPU memory
        - Clears cache after each volume
        - Uses patch-based processing for large volumes
    """
    # Reproducibility, and Cuda setup
    set_seed()
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run testing on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    force_cudnn_initialization()

    # Test Data - Initialize dataset and dataloader
    test_set = Dataset3D(mydict['image_dir'], mydict['test_datalist'])
    test_generator = torch.utils.data.DataLoader(
        test_set, 
        batch_size=1,  # Process one volume at a time
        shuffle=False,  # Maintain order for consistent results
        num_workers=8,  # Parallel data loading
        drop_last=False  # Process all volumes including incomplete batches
    )

    # Network - Initialize 3D UNet architecture
    single_gpu_network = UNet(
        in_channels=1,  # Single channel input (grayscale medical images)
        out_channels=mydict['num_classes'],  # Number of segmentation classes
        dim=3,  # 3D convolutions
        start_filters=32  # Initial number of filters (doubles each level)
    )
    # Use DataParallel for multi-GPU inference if available
    network = torch.nn.DataParallel(single_gpu_network).to(device)

    # Load trained model checkpoint
    if mydict['checkpoint_restore'] != "":
        checkpoint = torch.load(mydict['checkpoint_restore'], map_location=device)
        network.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded checkpoint: {}".format(mydict['checkpoint_restore']))
    else:
        print("No model found!")
        sys.exit()

    # Parameters for patch-based inference
    crop_size = mydict.get('crop_size', 96)  # Size of each patch
    step_size = mydict.get('step_size', 3*crop_size//4)  # 25% overlap between patches

    def get_patches_3d(volume):
        """
        Extract overlapping 3D patches from a volume for patch-based inference.
        
        This function divides a large 3D volume into smaller overlapping patches
        that can fit in GPU memory. The overlapping design helps reduce boundary
        artifacts when patches are stitched back together.
        
        Args:
            volume (numpy.ndarray): Input 3D volume with shape (D, H, W)
            
        Returns:
            tuple: (patches, coords) where:
                - patches (list): List of 3D numpy arrays, each of size crop_size^3
                - coords (list): List of tuples (z_start, z_end, y_start, y_end, x_start, x_end)
                  indicating the position of each patch in the original volume
                  
        Algorithm:
            - Uses step_size for patch spacing (75% of crop_size for 25% overlap)
            - Ensures full volume coverage by adjusting patch positions at boundaries
            - Maintains consistent patch size by adjusting start coordinates when needed
        """
        D, H, W = volume.shape
        patches = []
        coords = []
        
        # Iterate through volume with step_size spacing
        for z in range(0, D, step_size):
            for y in range(0, H, step_size):
                for x in range(0, W, step_size):
                    # Calculate patch boundaries, ensuring we don't exceed volume bounds
                    z_end = min(z + crop_size, D)
                    y_end = min(y + crop_size, H)
                    x_end = min(x + crop_size, W)
                    
                    # Adjust start coordinates to maintain consistent patch size
                    # This is important for patches near volume boundaries
                    z_start = max(0, z_end - crop_size)
                    y_start = max(0, y_end - crop_size)
                    x_start = max(0, x_end - crop_size)
                    
                    # Extract patch and store coordinates
                    patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]
                    patches.append(patch)
                    coords.append((z_start, z_end, y_start, y_end, x_start, x_end))
        
        return patches, coords

    def stitch_patches_3d(patches, coords, output_shape):
        """
        Reconstruct full volume from overlapping patches using weighted averaging.
        
        This function combines predictions from overlapping patches back into a
        full-size volume. Overlapping regions are averaged to reduce boundary
        artifacts and create smooth transitions between patches.
        
        Args:
            patches (list): List of predicted patch arrays
            coords (list): List of coordinate tuples (z_start, z_end, y_start, y_end, x_start, x_end)
            output_shape (tuple): Target shape (D, H, W) of the reconstructed volume
            
        Returns:
            numpy.ndarray: Reconstructed volume with shape output_shape
            
        Algorithm:
            - Accumulates patch predictions in overlapping regions
            - Tracks weights to compute proper averages
            - Handles edge cases where some voxels might not be covered
            - Uses weighted averaging for smooth blending of overlapping predictions
            
        Quality Assurance:
            - Checks for unprocessed regions and reports them
            - Ensures no division by zero in weight normalization
        """
        D, H, W = output_shape
        output = np.zeros((D, H, W))  # Accumulator for patch predictions
        weight = np.zeros((D, H, W))  # Weight map for averaging
        
        # Accumulate predictions from all patches
        for patch, (z_start, z_end, y_start, y_end, x_start, x_end) in zip(patches, coords):
            output[z_start:z_end, y_start:y_end, x_start:x_end] += patch
            weight[z_start:z_end, y_start:y_end, x_start:x_end] += 1
        
        # Quality check: Identify any unprocessed regions
        zero_regions = np.where(weight == 0)
        if len(zero_regions[0]) > 0:
            print(f"Found {len(zero_regions[0])} unprocessed voxels")
            print(f"Z range of zeros: {np.min(zero_regions[0])} to {np.max(zero_regions[0])}")
        
        # Prevent division by zero and compute weighted average
        weight = np.maximum(weight, 1)  # Ensure minimum weight of 1
        output = output / weight  # Weighted average
        return output

    print("Let the inference begin!")
    print("Total test volumes: {}".format(len(test_generator)))

    inference_start_time = time()
    with torch.no_grad():  # Disable gradient computation for inference
        network.eval()  # Set network to evaluation mode

        for idx, (x, o_filename) in enumerate(test_generator):
            # Extract filename for output naming
            filename = o_filename[0]
            filename = os.path.basename(filename).split('.')[0]
            print(f"Processing volume {idx + 1}/{len(test_generator)}: {filename}")

            # Get original volume dimensions (N=batch, C=channels, D=depth, H=height, W=width)
            N, C, D, H, W = x.shape
            
            # Extract 3D volume from batch and channel dimensions
            volume = x.squeeze(0).squeeze(0).cpu().numpy()  # Convert to numpy for patch extraction
            patches, coords = get_patches_3d(volume)
            
            # Process each patch through the network
            predicted_patches = []
            for i, patch in enumerate(patches):
                # Prepare patch tensor: add batch and channel dimensions
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                
                # Forward pass through network
                pred = torch.sigmoid(network(patch_tensor))  # Apply sigmoid for probability output
                pred = pred.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch/channel dims and move to CPU
                predicted_patches.append(pred)
                
                # Progress reporting every 100 patches
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(patches)} patches for volume {filename}")
            print(f"Processed all {len(patches)} patches for volume {filename}")
            
            # Reconstruct full volume from patches
            final_prediction = stitch_patches_3d(predicted_patches, coords, (D, H, W))
            
            # Save probability maps as numpy arrays (float32, range 0-1)
            np.save(os.path.join(mydict['output_folder'], f'pred_{filename}.npy'), final_prediction)
            
            # Create binary segmentation by thresholding at 0.5
            final_prediction_binary = (final_prediction >= 0.5).astype(np.uint8)
            
            # Save binary masks as TIFF images (uint8, values 0 or 255)
            imageio.volwrite(
                os.path.join(mydict['output_folder'], f'{filename}_pred.tiff'), 
                (final_prediction_binary * 255).astype(np.uint8)
            )

            # Memory cleanup to prevent GPU memory accumulation
            del predicted_patches, final_prediction
            torch.cuda.empty_cache()  # Clear GPU cache

    inference_end_time = time()

    print("\nInference Complete!")
    print(f"Total inference time: {inference_end_time - inference_start_time:.2f} seconds")


if __name__ == "__main__":
    """
    Main execution block for the 3D UNet inference script.
    
    Command line usage:
        python test_unet_3D.py --params config.json
        
    The config.json file should contain:
    {
        "num_classes": 1,
        "checkpoint_restore": "./checkpoint/model_best.pth",
        "image_dir": "../data",
        "test_datalist": "../test-samples.csv",
        "crop_size": 96,
        "output_folder": "../results/seg"
    }
    
    Output files:
        - pred_{filename}.npy: Float probability maps (range 0-1)
        - {filename}_pred.tiff: Binary segmentation masks (values 0 or 255)
    """
    parser = argparse.ArgumentParser(
        description='3D UNet Inference for Medical Image Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example usage: python test_unet_3D.py --params config.json"""
    )
    parser.add_argument('--params', type=str, required=True, 
                       help="Path to the JSON parameters file containing inference configuration")
    
    # Check if parameters file argument is provided
    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")
        parser.print_help()
        sys.exit(1)
    else:
        args = parser.parse_args()
        mydict = parse_func(args)
        test_3d(mydict)
