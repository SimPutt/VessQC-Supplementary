"""
Uncertainty Model for 3D Structure-wise Uncertainty Quantification

This module implements a deep learning model for uncertainty quantification in
3D medical image segmentation. The model combines 3D convolutional features
extracted from image patches with topological features from Discrete Morse Theory
to predict both epistemic and aleatoric uncertainty.

The model architecture consists of:
- 3D convolutional layers for spatial feature extraction
- Adaptive max pooling for global feature aggregation
- Fully connected layers for feature fusion
- Dual output heads for mean and log variance prediction
"""

import torch

class UncertaintyModel(torch.nn.Module):
    """
    Uncertainty quantification model for 3D structure-wise uncertainty quantification.
    
    This model predicts uncertainty in medical image segmentation by combining
    spatial features from 3D convolutions with topological features from
    Discrete Morse Theory analysis. It outputs both mean predictions and
    log variance for uncertainty estimation.
    
    Attributes:
        imgconv1 (torch.nn.Conv3d): First 3D convolutional layer
        imgconv2 (torch.nn.Conv3d): Second 3D convolutional layer
        maxpool (torch.nn.AdaptiveMaxPool3d): Adaptive max pooling layer
        relu (torch.nn.ReLU): ReLU activation function
        fc1 (torch.nn.Linear): First fully connected layer
        fc2 (torch.nn.Linear): Second fully connected layer
        fc3 (torch.nn.Linear): Third fully connected layer
        fc4_1 (torch.nn.Linear): Mean prediction head
        fc4_2 (torch.nn.Linear): Log variance prediction head
        dropout (torch.nn.Dropout): Dropout layer for regularization
    """
    
    def __init__(self, in_channels, num_features, hidden_units, p=0.2):
        """
        Initialize the uncertainty model.
        
        Args:
            in_channels (int): Number of input channels (image + likelihood + path)
            num_features (int): Number of topological features from DMT analysis
            hidden_units (int): Number of hidden units in fully connected layers
            p (float): Dropout probability (default: 0.2)
        """
        super(UncertaintyModel, self).__init__()

        # 3D convolutional layers for spatial feature extraction
        self.imgconv1 = torch.nn.Conv3d(
            in_channels=in_channels, 
            out_channels=24, 
            kernel_size=3, 
            padding='same'
        )
        self.imgconv2 = torch.nn.Conv3d(
            in_channels=24, 
            out_channels=32, 
            kernel_size=3, 
            padding='same'
        )
        
        # Adaptive max pooling to extract global features
        # Returns N,C,1,1,1 which will be concatenated with topological features
        self.maxpool = torch.nn.AdaptiveMaxPool3d(1)
        self.relu = torch.nn.ReLU()

        # Fully connected layers for feature fusion and prediction
        self.fc1 = torch.nn.Linear(num_features, hidden_units)
        self.fc2 = torch.nn.Linear(hidden_units, hidden_units*2)
        self.fc3 = torch.nn.Linear(hidden_units*2, hidden_units)
        
        # Dual output heads for uncertainty prediction
        self.fc4_1 = torch.nn.Linear(hidden_units, 1)  # Mean prediction
        self.fc4_2 = torch.nn.Linear(hidden_units, 1)  # Log variance prediction
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(p)   
    
    def forward(self, imgbatch, x):
        """
        Forward pass through the uncertainty model.
        
        Args:
            imgbatch (torch.Tensor): 3D image patches with shape (N, C, D, H, W)
                where N is the number of manifolds, C includes image, likelihood, and path channels
            x (torch.Tensor): Topological features from DMT analysis with shape (N, F)
                where F is the number of topological features
                
        Returns:
            tuple: (mu, log_var) where:
                - mu (torch.Tensor): Mean predictions with shape (N, 1)
                - log_var (torch.Tensor): Log variance predictions with shape (N, 1)
        """
        # Extract spatial features using 3D convolutions
        imgbatch = self.dropout(self.relu(self.imgconv1(imgbatch)))
        imgbatch = self.dropout(self.relu(self.imgconv2(imgbatch)))
        
        # Global feature aggregation using adaptive max pooling
        imgbatch = self.maxpool(imgbatch)
        
        # Remove spatial dimensions (D, H, W) to get global features
        imgbatch = torch.squeeze(torch.squeeze(torch.squeeze(imgbatch, dim=4), dim=3), dim=2)

        # Concatenate spatial features with topological features
        x = torch.concat((imgbatch, x), dim=1)
        
        # Process through fully connected layers
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))

        # Dual output heads for uncertainty prediction
        mu = self.fc4_1(x)      # Mean prediction
        log_var = self.fc4_2(x) # Log variance prediction
        
        return mu, log_var