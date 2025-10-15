"""
Discrete Morse Theory (DMT) Trainer Module

This module implements structure-wise uncertainty quantification for medical image segmentation using
Discrete Morse Theory. It processes likelihood maps to extract topological features based on manifold structures.

Key functionalities:
- DMT-based topological analysis of likelihood maps
- Manifold feature extraction for uncertainty prediction
- Random walk path finding for data augmentation
- Uncertainty heatmap reconstruction from model predictions
- Integration with DIPHA (Discrete Morse Theory) software

The module uses DIPHA for computing persistence diagrams and manifold structures
from 3D likelihood maps, then extracts features for uncertainty quantification.
"""

import subprocess
import sys
import numpy as np
import os, shutil
import torch
import csv
from PIL import Image

# Directory for DIPHA output files
savedir = "../dipha-graph-recon"
if not os.path.exists(savedir): 
    os.makedirs(savedir)

# DIPHA file format constants
DIPHA_CONST = 8067171840          # DIPHA magic number for file format identification
DIPHA_IMAGE_TYPE_CONST = 1        # Image type constant for DIPHA
DIM = 3                           # 3D image dimension

# Image processing parameters
mapHW = 64                        # Map height/width for convolutional layers (half of patch size)
halfHw = int(mapHW/2)             # Half of mapHW for centering operations

# Noise and random walk parameters
s_gaussiid = 0.01                 # Standard deviation for Gaussian IID noise
a_rw = 0.2                        # Alpha parameter for random walk path finding

# Connectivity definitions for 3D path finding
neighbors_6 = [[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]]  # 6-connectivity
neighbors_26 = [  # 26-connectivity (all adjacent voxels in 3x3x3 neighborhood)
    [1,1,1], [1,1,0], [1,1,-1], [1,0,1], [1,0,0], [1,0,-1], [1,-1,1], [1,-1,0], [1,-1,-1],
    [0,1,1], [0,1,0], [0,1,-1], [0,0,1], [0,0,-1], [0,-1,1], [0,-1,0], [0,-1,-1],
    [-1,1,1], [-1,1,0], [-1,1,-1], [-1,0,1], [-1,0,0], [-1,0,-1], [-1,-1,1], [-1,-1,0], [-1,-1,-1]
]
neighbors = neighbors_26  # Using 26-connectivity by default for more flexible path finding

def dmt_3d(lh_map, Th):
    """
    Apply 3D Discrete Morse Theory analysis to a likelihood map.
    
    This function processes a 3D likelihood map using DIPHA (Discrete Morse Theory)
    to compute persistence diagrams and extract topological features. It creates
    the necessary input files for DIPHA and runs the analysis pipeline.
    
    Args:
        lh_map (numpy.ndarray): 3D likelihood map with values in [0,1] range
        Th (float): Threshold value for manifold extraction (scaled by 255)
    """
    # Define file paths for DIPHA input/output
    dipha_diagram_filename = os.path.join(savedir, 'inputs/diagram.bin')
    dipha_output_filename = os.path.join(savedir, 'inputs/complex.bin')
    vert_filename = os.path.join(savedir, 'inputs/vert.txt')
    dipha_edge_filename = os.path.join(savedir, 'inputs/dipha.edges')
    dipha_edge_txt = os.path.join(savedir, 'inputs/dipha_edges.txt')
    dipha_output = os.path.join(savedir, 'output/')

    # Get image dimensions and create working copy
    nx, ny, nz = lh_map.shape
    im_cube = np.zeros([nx, ny, nz])
    im_cube[:, :, :] = lh_map

    # Create DIPHA input file in binary format
    with open(dipha_output_filename, 'wb') as output_file:
        # Write DIPHA file header
        np.int64(DIPHA_CONST).tofile(output_file)           # Magic number
        np.int64(DIPHA_IMAGE_TYPE_CONST).tofile(output_file) # Image type
        np.int64(nx * ny * nz).tofile(output_file)          # Total voxels
        np.int64(DIM).tofile(output_file)                   # Dimensions
        np.int64(nx).tofile(output_file)                    # X dimension
        np.int64(ny).tofile(output_file)                    # Y dimension
        np.int64(nz).tofile(output_file)                    # Z dimension
        
        # Write voxel values (inverted and scaled to [0,255])
        for k in range(nz):
            sys.stdout.flush()
            for j in range(ny):
                for i in range(nx):
                    val = int(-im_cube[i, j, k]*255)  # Invert and scale
                    np.float64(val).tofile(output_file)
        output_file.close()

    # Create vertex file with coordinates and values
    with open(vert_filename, 'w') as vert_file:
        for k in range(nz):
            sys.stdout.flush()
            for j in range(ny):
                for i in range(nx):
                    # Format: x y z value
                    vert_file.write(str(i) + ' ' + str(j) + ' ' + str(k) + ' ' + str(int(-im_cube[i, j, k] * 255)) + '\n')
        vert_file.close()

    # Run DIPHA analysis pipeline
    # Step 1: Compute persistence diagram and edge information
    subprocess.call(["mpiexec", "-n", "1", 
                    "../dipha-graph-recon/build/dipha", 
                    str(dipha_output_filename), 
                    str(dipha_diagram_filename), 
                    str(dipha_edge_filename), 
                    str(nx), str(ny), str(nz)])

    # Step 2: Convert edge file to text format
    subprocess.call(["../dipha-graph-recon/src/loop.out", 
                    str(dipha_edge_filename), str(dipha_edge_txt)])

    # Step 3: Extract manifold structures using threshold
    subprocess.call(["../dipha-graph-recon/src/manifold.out", 
                    str(vert_filename), str(dipha_edge_txt), 
                    str(Th), str(dipha_output)])


def dmt(num_classes, patch, Th=0.02):
    """
    Apply Discrete Morse Theory analysis to a batch of likelihood patches.
    
    This function processes a batch of likelihood maps by extracting the appropriate
    channel and applying DMT analysis to each sample in the batch.
    
    Args:
        num_classes (int): Number of classes (2 for binary, 1 for single class)
        patch (torch.Tensor): Batch of likelihood maps with shape (B, C, D, H, W)
        Th (float): Threshold value in [0,1] range for manifold extraction
    """
    B, C, D, H, W = patch.shape
    
    # Extract appropriate channel based on number of classes
    if num_classes == 2:
        patch = np.array(patch.detach().cpu())[:,1,:,:,:]  # Use channel 1 for binary classification
    else:
        patch = np.array(patch.detach().cpu())[:,0,:,:,:]  # Use channel 0 for single class
    
    # Apply DMT analysis to each sample in the batch
    for i in range(B):
        dmt_3d(patch[i,:,:,:], Th * 255)  # Scale threshold to [0,255] range


def gaussianIID(inp):
    """
    Add Gaussian IID (Independent and Identically Distributed) noise to input.
    
    Even though sigma is fixed, since the noise is IID, the resulting map is 
    different every time, providing data augmentation for training.
    
    Args:
        inp (numpy.ndarray): Input array to add noise to
        
    Returns:
        numpy.ndarray: Input array with added Gaussian noise
    """
    return inp + np.random.normal(loc=0., scale=s_gaussiid, size=inp.shape)

def getdist(srcc, dstc):
    """
    Calculate inverse Euclidean distance between two 3D points.
    
    Args:
        srcc (tuple/list): Source coordinates (x, y, z)
        dstc (tuple/list): Destination coordinates (x, y, z)
        
    Returns:
        float: Inverse Euclidean distance (1/distance)
    """
    ans = np.sqrt(pow(srcc[0]-dstc[0],2) + pow(srcc[1]-dstc[1],2) + pow(srcc[2]-dstc[2],2))
    return 1./ans

def checkbounds(curc, mapshape):
    """
    Check if 3D coordinates are within the bounds of a given map shape.
    
    Args:
        curc (tuple/list): Current coordinates (x, y, z)
        mapshape (tuple): Map shape (x_dim, y_dim, z_dim)
        
    Returns:
        bool: True if coordinates are within bounds, False otherwise
    """
    if curc[0] >= mapshape[0] or curc[1] >= mapshape[1] or curc[2] >= mapshape[2] or curc[0] < 0 or curc[1] < 0 or curc[2] < 0:
        return False
    return True

def getPath(likelihood_map, srcc, dstc): 
    """
    Find a path between two points using random walk with likelihood guidance.
    
    This function implements a random walk algorithm that moves from source to destination
    by choosing the neighbor with the highest combined score of distance to target and
    likelihood value. If the walk gets stuck or exceeds step limit, it terminates.
    
    Args:
        likelihood_map (numpy.ndarray): 3D likelihood map guiding the path
        srcc (tuple/list): Source coordinates (x, y, z)
        dstc (tuple/list): Destination coordinates (x, y, z)
        
    Returns:
        numpy.ndarray: Binary image showing the path taken (1 for path voxels, 0 otherwise)
    """
    mini_image = np.zeros_like(likelihood_map)
    lmshape = likelihood_map.shape
    curc = srcc  # current coordinates
    mini_image[curc[0], curc[1], curc[2]] = 1

    path_cnt = 0

    # Continue walking until destination is reached
    while(np.any(curc != dstc)):
        max_p_val = None
        neighbor_coord = None 

        # Check all 26 neighbors
        for offset in neighbors:
            newc = curc + np.array(offset)

            # If we've reached the destination, stop
            if np.all(newc == dstc):
                neighbor_coord = newc
                break 
                
            # Check if neighbor is within bounds
            if checkbounds(newc, lmshape):
                # Calculate combined score: distance to target + likelihood value
                p_val = a_rw*getdist(newc, dstc) + (1.-a_rw)*likelihood_map[newc[0], newc[1], newc[2]]

                # Choose neighbor with highest score
                if max_p_val is None or max_p_val < p_val:
                    max_p_val = p_val
                    neighbor_coord = newc

        # If no valid neighbor found, terminate walk
        if neighbor_coord is None:
            print("[getPath] Random Walk: No valid neighbor found for next step")
            break
        
        # Move to chosen neighbor
        curc = neighbor_coord
        mini_image[curc[0], curc[1], curc[2]] = 1

        path_cnt += 1
        # Prevent infinite loops - limit to 100 steps in 3D
        if path_cnt > 100:
            break

    return mini_image    


def getImgBatch(img, likelihood, dmt_bimg, srccoord, dstcoord):
    """
    Create an image batch for training by combining image, likelihood, and path information.
    
    This function creates a training sample by either using the original likelihood map
    or generating a new path using random walk with Gaussian noise augmentation.
    The resulting batch is cropped to a fixed size around the source-destination region.
    
    Args:
        img (numpy.ndarray): Input image with shape (C, H, W) where C=3 for RGB
        likelihood (numpy.ndarray): Likelihood map with shape (H, W)
        dmt_bimg (numpy.ndarray): DMT binary image with shape (H, W)
        srccoord (tuple/list): Source coordinates (x, y, z)
        dstcoord (tuple/list): Destination coordinates (x, y, z)
        
    Returns:
        torch.Tensor: Combined image batch with shape (C+2, H, W) where +2 adds likelihood and path channels
    """ 
    global del_cnt

    # Data augmentation: 50% chance to use original likelihood, 50% chance to use noisy version
    if np.random.rand() > 0.5:  # Use original likelihood map
        temp_lm = likelihood
        temp_path = dmt_bimg
    else:  # Use Gaussian noise augmented likelihood and generate new path
        temp_lm = gaussianIID(likelihood)
        temp_path = getPath(temp_lm, srccoord, dstcoord)

    # Stack likelihood and path channels, then concatenate with image
    nstack = np.stack([temp_lm, temp_path])
    nstack = np.concatenate((img, nstack), axis=0)

    # Calculate crop region centered between source and destination
    minx = min(srccoord[0], dstcoord[0])
    maxx = max(srccoord[0], dstcoord[0])
    miny = min(srccoord[1], dstcoord[1])
    maxy = max(srccoord[1], dstcoord[1])
    minz = min(srccoord[2], dstcoord[2])
    maxz = max(srccoord[2], dstcoord[2])
    
    # Find center point
    midx = int(minx + (maxx - minx)/2)
    midy = int(miny + (maxy - miny)/2)
    midz = int(minz + (maxz - minz)/2)

    # Define crop boundaries
    dstx = midx + halfHw
    dsty = midy + halfHw
    dstz = midz + halfHw
    srcx = midx - halfHw
    srcy = midy - halfHw
    srcz = midz - halfHw

    # Handle boundary conditions to ensure crop is within image bounds
    if dstx >= likelihood.shape[0]:
        dstx = likelihood.shape[0]
        srcx = dstx - mapHW

    if dsty >= likelihood.shape[1]:
        dsty = likelihood.shape[1]
        srcy = dsty - mapHW

    if dstz >= likelihood.shape[2]:
        dstz = likelihood.shape[2]
        srcz = dstz - mapHW

    if srcx < 0:
        srcx = 0
        dstx = srcx + mapHW
    
    if srcy < 0:
        srcy = 0
        dsty = srcy + mapHW   

    if srcz < 0:
        srcz = 0
        dstz = srcz + mapHW    

    return torch.from_numpy(nstack[:, srcx:dstx, srcy:dsty, srcz:dstz])  # Return CHWD tensor


def getManifoldFeatures(num_classes, img, likelihood):
    """
    Extract manifold features from DMT analysis results for uncertainty prediction.
    
    This function reads the manifold structures computed by DIPHA and extracts
    features for each manifold, including size, average likelihood, and persistence
    values. It also creates image batches for training the uncertainty model.
    
    Args:
        num_classes (int): Number of classes (2 for binary, 1 for single class)
        img (torch.Tensor): Input image tensor
        likelihood (torch.Tensor): Likelihood map tensor
        
    Returns:
        tuple: (img_batch, unc_input) where:
            - img_batch: Tensor of image patches for each manifold (N, C, H, W, D)
            - unc_input: Tensor of uncertainty features for each manifold (N, 4)
    """
    return_input = []
    return_imgbatch = [] #use .stack on it later

    manifold_filepath = os.path.join(savedir, "output/dimo_manifold.txt")
    vert_filepath = os.path.join(savedir, "output/dimo_vert.txt")

    if num_classes == 2:
        likelihood = torch.squeeze(likelihood).detach().cpu().numpy()[1] # probabilities in channel 1
    else:
        likelihood = torch.squeeze(likelihood).detach().cpu().numpy()
    img = torch.squeeze(img,0).detach().cpu().numpy() # CHWD

    nx, ny, nz = likelihood.shape

    vert_info = np.loadtxt(vert_filepath)
    bin_image = np.zeros([nx, ny, nz])
    pers_image = np.zeros([nx, ny, nz])
    likeli_image = np.zeros([nx, ny, nz])
    srccoord = None 
    dstcoord = None

    manifold_cnt = -1
    with open(manifold_filepath, 'r') as manifold_info:
        reader = csv.reader(manifold_info, delimiter=' ')
        for row in reader:
            if len(row) != 3:
                if bin_image.sum() != 0:
                    manifold_cnt += 1
                    
                    # add to return_input
                    likeli_image = bin_image * likelihood
                    manifold_size = bin_image.sum()

                    dstcoord = [int(vert_info[v1,0]), int(vert_info[v1,1]), int(vert_info[v1,2])]

                    return_imgbatch.append(getImgBatch(img, likelihood, bin_image, np.array(srccoord), np.array(dstcoord))) # returns torch tensor

                    return_input.append(np.array([manifold_size, likeli_image.sum()/manifold_size,pers_image.sum()/manifold_size,0.02]))

                bin_image = np.zeros([nx, ny, nz])
                pers_image = np.zeros([nx, ny, nz])
                srccoord = None
                dstcoord = None
                continue

            v0 = int(row[0])
            v1 = int(row[1])
            pers_value = int(row[2])/255. # in [0,255] range

            if srccoord is None:
                srccoord = [int(vert_info[v0,0]), int(vert_info[v0,1]), int(vert_info[v0,2])]

            bin_image[int(vert_info[v0,0]), int(vert_info[v0,1]), int(vert_info[v0,2])] = 1
            bin_image[int(vert_info[v1,0]), int(vert_info[v1,1]), int(vert_info[v1,2])] = 1

            pers_image[int(vert_info[v0,0]), int(vert_info[v0,1]), int(vert_info[v0,2])] = pers_value
            pers_image[int(vert_info[v1,0]), int(vert_info[v1,1]), int(vert_info[v1,2])] = pers_value

    if return_input == []:
        return_input = None
        return_imgbatch = None
    else:
        return_input = torch.from_numpy(np.array(return_input))
        return_imgbatch = torch.stack(return_imgbatch, dim=0) #NCHWD form

    return return_imgbatch, return_input # torch datatype


def getData(num_classes, img, likelihood):
    """
    Main data processing function for training uncertainty models.
    
    This function applies DMT analysis to likelihood maps and extracts manifold
    features for uncertainty prediction. It's the main entry point for processing
    training data.
    
    Args:
        num_classes (int): Number of classes (2 for binary, 1 for single class)
        img (torch.Tensor): Input image tensor with shape (N, C, H, W, D)
        likelihood (torch.Tensor): Likelihood map tensor with shape (N, C, H, W, D)
        
    Returns:
        tuple: (img_batch, unc_input) where:
            - img_batch: Tensor of image patches for each manifold
            - unc_input: Tensor of uncertainty features for each manifold
    """
    # Clamp likelihood values to [0,1] range
    likelihood = torch.clamp(likelihood, 0., 1.)
    
    # Apply DMT analysis to extract topological features
    dmt(num_classes, likelihood)
    
    # Extract manifold features and create training batches
    img_batch, unc_input = getManifoldFeatures(num_classes, img, likelihood)    
    return img_batch, unc_input


def getData_val(num_classes, img, likelihood):
    """
    Validation data processing function.
    
    This function is identical to getData but is used for validation/testing
    to maintain consistency in the API.
    
    Args:
        num_classes (int): Number of classes (2 for binary, 1 for single class)
        img (torch.Tensor): Input image tensor with shape (N, C, H, W, D)
        likelihood (torch.Tensor): Likelihood map tensor with shape (N, C, H, W, D)
        
    Returns:
        tuple: (img_batch, unc_input) where:
            - img_batch: Tensor of image patches for each manifold
            - unc_input: Tensor of uncertainty features for each manifold
    """
    return getData(num_classes, img, likelihood)


def reconstruct_uncertainty_heatmap(datadir, unc_pred_mu, unc_pred_logvar, img_shape, prefix):
    """
    Reconstruct uncertainty heatmap from model predictions and manifold structures.
    
    This function takes Monte Carlo predictions from the uncertainty model and
    reconstructs a 3D uncertainty heatmap by mapping predictions back to the
    original manifold structures. It computes both epistemic and aleatoric
    uncertainty components.
    
    Args:
        datadir (str): Directory to save output files
        unc_pred_mu (list): List of mean predictions from Monte Carlo sampling
        unc_pred_logvar (list): List of log variance predictions from Monte Carlo sampling
        img_shape (tuple): Shape of the output image (D, H, W)
        prefix (str): Prefix for output filenames
        
    Returns:
        numpy.ndarray: 3D uncertainty heatmap with shape img_shape
    """
    eps = 0.1
    logfile = os.path.join(datadir, prefix+"_structure_info.txt")
    if not os.path.exists(os.path.dirname(logfile)):
        os.makedirs(os.path.dirname(logfile))

    # File paths for DMT analysis results
    manifold_filepath = os.path.join(savedir, "output/dimo_manifold.txt")
    vert_filepath = os.path.join(savedir, "output/dimo_vert.txt")

    # Convert predictions to numpy arrays
    unc_pred_mu = np.array(unc_pred_mu) 
    unc_pred_logvar = np.array(unc_pred_logvar)
    
    # Compute uncertainty components from Monte Carlo samples
    unc_pred_epistemic = np.var(unc_pred_mu, axis=0)           # Epistemic uncertainty (model uncertainty)
    unc_pred_aleatoric = np.exp(np.mean(unc_pred_logvar, axis=0))  # Aleatoric uncertainty (data uncertainty)
    unc_pred_avg = np.mean(unc_pred_mu, axis=0)                # Average prediction

    # Ensure all uncertainty components have the same shape
    assert np.squeeze(unc_pred_aleatoric).shape == np.squeeze(unc_pred_avg).shape
    assert np.squeeze(unc_pred_epistemic).shape == np.squeeze(unc_pred_avg).shape

    # Reshape arrays to ensure proper dimensions
    if unc_pred_avg.shape[0] == 1:
        unc_pred_aleatoric = np.reshape(unc_pred_aleatoric, 1)
        unc_pred_epistemic = np.reshape(unc_pred_epistemic, 1)
        unc_pred_avg = np.reshape(unc_pred_avg, 1)
    else:
        unc_pred_aleatoric = np.reshape(unc_pred_aleatoric, (-1,1))
        unc_pred_epistemic = np.reshape(unc_pred_epistemic, (-1,1))
        unc_pred_avg = np.reshape(unc_pred_avg, (-1,1))

    # Load vertex information from DMT analysis
    vert_info = np.loadtxt(vert_filepath)
    mini_image = np.zeros(img_shape)
    full_image = np.zeros(img_shape)

    # Open log file for writing structure information
    writefile = open(logfile, 'a')

    # Process manifold structures and map predictions back to image space
    manifold_cnt = -1
    with open(manifold_filepath, 'r') as manifold_info:
        reader = csv.reader(manifold_info, delimiter=' ')
        for row in reader:
            if len(row) != 3:  # End of manifold structure
                if mini_image.sum() != 0:
                    manifold_cnt += 1

                    # Write uncertainty information to log file
                    writestr = str(unc_pred_aleatoric[manifold_cnt]) + ',' + str(unc_pred_epistemic[manifold_cnt]) + ',' + str(unc_pred_avg[manifold_cnt]) + '\n'
                    writefile.write(writestr)

                    # Add current manifold to full image
                    full_image += mini_image

                # Reset for next manifold
                mini_image = np.zeros(img_shape)
                continue

            # Process edge in manifold structure
            v0 = int(row[0])
            v1 = int(row[1])

            # Map uncertainty values to vertex coordinates
            mini_image[int(vert_info[v0,0]), int(vert_info[v0,1]), int(vert_info[v0,2])] = unc_pred_avg[manifold_cnt+1]
            mini_image[int(vert_info[v1,0]), int(vert_info[v1,1]), int(vert_info[v1,2])] = unc_pred_avg[manifold_cnt+1]

    # Verify that all manifolds were processed
    assert manifold_cnt+1 == unc_pred_avg.shape[0]

    writefile.close()
    return full_image