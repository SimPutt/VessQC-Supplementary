"""
3D Dataset Loader for Structure-wise Uncertainty Quantification

This module implements a PyTorch dataset class for loading 3D medical images
and their corresponding segmentation likelihood maps. It handles file loading,
preprocessing, and normalization for uncertainty quantification tasks.

Key functionalities:
- Loading 3D medical images from TIFF files
- Loading segmentation likelihood maps from NumPy files
- Image normalization and preprocessing
"""

import torch
import imageio
import os, glob, sys
import numpy as np

class Dataset3D(torch.utils.data.Dataset):
    """
    3D dataset class for structure-wise uncertainty quantification.
    
    This dataset loads 3D medical images and their corresponding segmentation
    likelihood maps, providing them in a format suitable for uncertainty
    quantification models. It handles file loading, preprocessing, and
    normalization automatically.
    
    Attributes:
        listpath (str): Path to file containing list of images to load
        allFiles (list): List of all image file paths
        dataCPU (dict): Dictionary storing file paths and data on CPU
        seg_dir (str): Directory containing segmentation likelihood files
        img_dir (str): Directory containing input images
        patches (list): List of image patches (currently unused)
    """
    
    def __init__(self, img_dir, seg_dir, listpath):
        """
        Initialize the 3D dataset.
        
        Args:
            img_dir (str): Directory containing input 3D images
            seg_dir (str): Directory containing segmentation likelihood maps
            listpath (str): Path to file containing list of images to load
        """
        self.listpath = listpath

        # Initialize data storage
        self.allFiles = []
        self.seg_dir = seg_dir
        self.img_dir = img_dir

        # Load file paths (only filenames, not actual data)
        self.loadCPU()
        print("Length of dataset (num of 3D files): {}".format(len(self.allFiles)))

    def loadCPU(self):
        """
        Load file paths from the list file.
        
        This method reads the list of image files from the specified path
        and creates full file paths by joining with the image directory.
        It only loads file paths, not the actual image data, for memory efficiency.
        """
        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]

        im_paths = []

        # Create full paths by joining with image directory
        for im_path in mylist:
            im_paths.append(os.path.join(self.img_dir, im_path))
        
        # Store file paths and sort in reverse order
        self.allFiles.extend(im_paths)
        self.allFiles.sort(reverse=True)


    def interpolate(self, nparr):
        """
        Normalize array values to [0, 1] range using min-max scaling.
        
        Args:
            nparr (numpy.ndarray): Input array to normalize
            
        Returns:
            numpy.ndarray: Normalized array with values in [0, 1] range
        """
        omin = 0
        omax = 1.0
        imin = np.min(nparr)
        imax = np.max(nparr)

        return (nparr - imin) * (omax - omin) / (imax - imin) + omin

    def preprocess(self, im_path):
        """
        Preprocess a single image and its corresponding likelihood map.
        
        This method loads an image and its segmentation likelihood map,
        normalizes the image to [0, 1] range, and ensures both arrays
        have the same shape.
        
        Args:
            im_path (str): Path to the input image file
            
        Returns:
            tuple: (arrayimage_norm, array_lh) where:
                - arrayimage_norm: Normalized image array with values in [0, 1]
                - array_lh: Segmentation likelihood array
        """
        # Construct path to corresponding likelihood file
        basename = "pred_" + os.path.basename(im_path)
        likelihood_path = os.path.join(self.seg_dir, basename.replace('.tiff', '.npy'))

        assert os.path.exists(likelihood_path), f"Likelihood file not found: {likelihood_path}"

        # Load image and likelihood map
        arrayimage = imageio.v3.imread(im_path).astype(np.float32)
        array_lh = np.load(likelihood_path)
        
        # Handle compressed NumPy files
        if 'npz' in likelihood_path:
            array_lh = array_lh['probabilities'][1]

        # Ensure both arrays have the same shape
        assert arrayimage.shape == array_lh.shape, f"Shape mismatch: image {arrayimage.shape} vs likelihood {array_lh.shape}"
        
        # Normalize image to [0, 1] range
        arrayimage_norm = self.interpolate(arrayimage)

        return arrayimage_norm, array_lh


    def __len__(self):
        """
        Return the total number of 3D images in the dataset.
        
        Returns:
            int: Number of 3D images in the dataset
        """
        return len(self.allFiles)

    def __getitem__(self, index):
        """
        Get a single item from the dataset.
        
        This method loads and preprocesses a single 3D image and its
        corresponding likelihood map, converting them to PyTorch tensors
        with the appropriate channel dimension.
        
        Args:
            index (int): Index of the item to retrieve
            
        Returns:
            tuple: (torch_img, torch_lh, im_path) where:
                - torch_img: Image tensor with shape (1, D, H, W)
                - torch_lh: Likelihood tensor with shape (1, D, H, W)
                - im_path: Path to the original image file
        """
        im_path = self.allFiles[index]
        np_img, np_lh = self.preprocess(im_path)  # Returns (D, H, W) arrays

        # Add channel dimension and convert to PyTorch tensors
        torch_img = torch.unsqueeze(torch.from_numpy(np_img), dim=0)  # (1, D, H, W)
        torch_lh = torch.unsqueeze(torch.from_numpy(np_lh), dim=0)    # (1, D, H, W)

        return torch_img, torch_lh, im_path 


