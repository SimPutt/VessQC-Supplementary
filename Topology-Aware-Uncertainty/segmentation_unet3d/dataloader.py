"""
3D Medical Image Dataset Loader for PyTorch

This module provides a custom PyTorch Dataset class for loading and preprocessing
3D medical images stored as TIFF files. It handles normalization, file path
management, and provides a clean interface for training and inference pipelines.

Key Features:
- Loads 3D TIFF medical images with automatic normalization
- Supports batch processing through PyTorch DataLoader
- Memory-efficient loading (loads images on-demand)
- Flexible preprocessing pipeline
- Compatible with TIFF file format
"""

import torch
import imageio
import os, glob, sys
import numpy as np

class Dataset3D(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for loading 3D medical images.
    
    This class extends torch.utils.data.Dataset to handle 3D medical images
    stored as TIFF files. It provides on-demand loading, preprocessing, and
    normalization of medical image volumes.
    
    Attributes:
        img_dir (str): Directory containing the image files
        listpath (str): Path to text file containing list of image filenames
        allFiles (list): List of full paths to all image files
        dataCPU (dict): Dictionary for storing file paths and image data
        patches (list): List for storing image patches (currently unused)
    
    File Format Requirements:
        - Images should be in TIFF format (.tiff)
        - List file should contain one filename per line
        - Images should be 3D volumes (D x H x W)
    """
    
    def __init__(self, img_dir, listpath):
        """
        Initialize the Dataset3D loader.
        
        Args:
            img_dir (str): Directory path containing the image files
            listpath (str): Path to text file containing list of image filenames
                          (one filename per line, without directory path)
                                    
        Raises:
            FileNotFoundError: If img_dir or listpath doesn't exist
            IOError: If list file cannot be read
        """
        # Store configuration parameters
        self.listpath = listpath
        self.img_dir = img_dir
        self.allFiles = []  # Will store full paths to all image files

        # Load file paths from the list file
        self.loadCPU()
        
        # Print dataset statistics
        print("Length of dataset (num of 3D volumes): {}".format(len(self.allFiles)))
        print("Length of files on CPU: {}".format(len(self.allFiles)))

    def loadCPU(self):
        """
        Load file paths from the list file and create full paths.
        
        This method reads the list file containing image filenames,
        constructs full paths by joining with the image directory,
        and stores them in self.allFiles.
        
        The method performs the following operations:
        1. Read all lines from the list file
        2. Strip newline characters from each filename
        3. Construct full paths by joining img_dir with each filename
        4. Sort paths in reverse order for consistent ordering

        Raises:
            FileNotFoundError: If the list file doesn't exist
            IOError: If the list file cannot be read
        """
        # Read all filenames from the list file
        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        
        # Remove newline characters from each filename
        mylist = [x.rstrip('\n') for x in mylist]

        # Build full paths by joining directory with filenames
        im_paths = []
        for im_path in mylist:
            full_path = os.path.join(self.img_dir, im_path)
            im_paths.append(full_path)
    
        # Store all file paths and sort for consistent ordering
        self.allFiles.extend(im_paths)
        self.allFiles.sort(reverse=True)  # Reverse sort for consistent iteration


    def interpolate(self, nparr):
        """
        Normalize image intensities to the range [0, 1].
        
        This method performs min-max normalization, scaling image intensities
        from their original range to [0, 1]. This is crucial for neural network
        training as it ensures consistent input ranges and helps with convergence.
        
        The normalization formula is:
        normalized = (x - min) * (new_max - new_min) / (max - min) + new_min
        
        Args:
            nparr (numpy.ndarray): Input image array with arbitrary intensity range
            
        Returns:
            numpy.ndarray: Normalized image array with values in range [0, 1]
            
        Note:
            - If input array is constant (min == max), result will be all zeros
            - This method preserves the relative intensity relationships
            - Suitable for most medical imaging modalities (CT, MRI, etc.)
        """
        # Define target range [0, 1]
        omin = 0    # Output minimum
        omax = 1.0  # Output maximum
        
        # Find input range
        imin = np.min(nparr)  # Input minimum
        imax = np.max(nparr)  # Input maximum

        # Apply min-max normalization formula
        # Handle edge case where imin == imax (constant image)
        if imax == imin:
            return np.zeros_like(nparr)  # Return zeros for constant images
        
        return (nparr - imin) * (omax - omin) / (imax - imin) + omin


    def preprocess(self, im_path):
        """
        Load and preprocess a 3D medical image from file.
        
        This method handles the complete preprocessing pipeline for medical images:
        1. Load the TIFF file using imageio
        2. Convert to float32 for numerical precision
        3. Apply intensity normalization to [0, 1] range
        
        Args:
            im_path (str): Full path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed 3D image array with shape (D, H, W)
                          and values normalized to [0, 1]
                          
        Raises:
            FileNotFoundError: If the image file doesn't exist
            IOError: If the image file cannot be read
            ValueError: If the image format is not supported
        """
        try:
            # Load TIFF file - imageio.v3.imread handles 3D TIFF files automatically
            arrayimage = imageio.v3.imread(im_path).astype(np.float32)
            
            # Apply intensity normalization to [0, 1] range
            arrayimage_norm = self.interpolate(arrayimage)
            
            return arrayimage_norm
            
        except Exception as e:
            print(f"Error loading image {im_path}: {str(e)}")
            raise


    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        This method is required by PyTorch's Dataset interface and is used
        by DataLoader to determine the dataset size for batching and iteration.
        
        Returns:
            int: Total number of 3D volumes in the dataset
        """
        return len(self.allFiles)


    def __getitem__(self, index):
        """
        Retrieve a single sample from the dataset.
        
        This method is called by PyTorch's DataLoader to fetch individual samples
        during training or inference. It loads and preprocesses the image at the
        specified index.
        
        Args:
            index (int): Index of the sample to retrieve (0 <= index < len(dataset))
            
        Returns:
            tuple: (image_tensor, image_path) where:
                - image_tensor (torch.Tensor): Preprocessed image with shape (C, D, H, W)
                  where C=1 (single channel), D=depth, H=height, W=width
                - image_path (str): Full path to the original image file
                  
        Tensor Format:
            - Input: 3D numpy array with shape (D, H, W)
            - Output: 4D PyTorch tensor with shape (1, D, H, W)
            - Data type: torch.float32
            - Value range: [0, 1] (normalized)
            
        Processing Pipeline:
            1. Get file path from index
            2. Load and preprocess image (DHW format)
            3. Convert to PyTorch tensor
            4. Add channel dimension (1, D, H, W)
            5. Return tensor and original file path
            
        Usage in DataLoader:
            When used with batch_size > 1, multiple samples are automatically
            stacked into a batch tensor with shape (N, 1, D, H, W).
        """
        # Get the file path for this index
        im_path = self.allFiles[index]
        
        # Load and preprocess the image (returns DHW numpy array)
        np_img = self.preprocess(im_path)
        
        # Convert to PyTorch tensor and add channel dimension
        # Input: (D, H, W) -> Output: (1, D, H, W) where 1 is the channel dimension
        torch_img = torch.unsqueeze(torch.from_numpy(np_img), dim=0)

        return torch_img, im_path 


