"""
3D UNet Architecture Implementation for Medical Image Segmentation

This module implements a flexible 3D UNet architecture for medical image segmentation.
The UNet follows the encoder-decoder structure with skip connections, making it
particularly effective for biomedical image segmentation tasks.

Key Features:
- Multiple activation functions (ReLU, LeakyReLU, ELU)
- Various normalization options (Batch, Instance, Group)
- Configurable upsampling modes (transposed convolution, interpolation)
- Skip connections for preserving fine-grained details
- Automatic weight initialization

Tensor Format Conventions:
- 3D: NCDHW (Batch, Channels, Depth, Height, Width)

Architecture:
    Input → Encoder (DownBlocks) → Bottleneck → Decoder (UpBlocks) → Output
    
    Skip connections connect corresponding encoder and decoder levels to
    preserve spatial information lost during downsampling.

Typical Usage:
    model = UNet(in_channels=1, out_channels=2, dim=3, start_filters=32)
    prediction = model(input_tensor)  # input: (N, 1, D, H, W)
"""

import torch
from torch import nn

@torch.jit.script
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """
    Center-crops the encoder_layer to the size of the decoder_layer,
    so that merging (concatenation) between levels/blocks is possible.
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
    """
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4:  # 2D
            encoder_layer = encoder_layer[
                :,
                :,
                ((ds[0] - es[0]) // 2) : ((ds[0] + es[0]) // 2),
                ((ds[1] - es[1]) // 2) : ((ds[1] + es[1]) // 2),
            ]
        elif encoder_layer.dim() == 5:  # 3D
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                :,
                :,
                ((ds[0] - es[0]) // 2) : ((ds[0] + es[0]) // 2),
                ((ds[1] - es[1]) // 2) : ((ds[1] + es[1]) // 2),
                ((ds[2] - es[2]) // 2) : ((ds[2] + es[2]) // 2),
            ]
    return encoder_layer, decoder_layer


def conv_layer(dim: int):
    """
    Return the appropriate convolution layer class based on dimensionality.
    
    This factory function provides the correct PyTorch convolution layer
    for the specified number of spatial dimensions.
    
    Args:
        dim (int): Number of spatial dimensions (2 or 3)
        
    Returns:
        torch.nn.Module: Convolution layer class
            - nn.Conv2d for 2D (operates on H×W)
            - nn.Conv3d for 3D (operates on D×H×W)
            
    Raises:
        ValueError: If dim is not 2 or 3
    """
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d
    else:
        raise ValueError(f"Unsupported dimension: {dim}. Only 2D and 3D are supported.")


def get_conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = True,
    dim: int = 2,
):
    """
    Create a convolution layer with specified parameters.
    
    This function creates either a 2D or 3D convolution layer based on the
    dimensionality parameter, with consistent parameter handling.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int, optional): Size of convolution kernel. Defaults to 3.
        stride (int, optional): Convolution stride. Defaults to 1.
        padding (int, optional): Padding size. Defaults to 1.
            - padding=1 with kernel_size=3 maintains spatial dimensions
            - padding=0 reduces spatial dimensions
        bias (bool, optional): Whether to use bias term. Defaults to True.
        dim (int, optional): Spatial dimensions (2 or 3). Defaults to 2.
        
    Returns:
        torch.nn.Module: Configured convolution layer
        
    Common Configurations:
        - Standard conv: kernel_size=3, stride=1, padding=1 (preserves size)
        - Downsampling: kernel_size=3, stride=2, padding=1 (halves size)
        - Point-wise: kernel_size=1, stride=1, padding=0 (changes channels only)
    """
    return conv_layer(dim)(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )


def conv_transpose_layer(dim: int):
    """
    Return the appropriate transposed convolution layer class based on dimensionality.
    
    Transposed convolutions (also called deconvolutions) are used for upsampling
    in the decoder path of UNet. They learn upsampling parameters rather than
    using fixed interpolation methods.
    
    Args:
        dim (int): Number of spatial dimensions (2 or 3)
        
    Returns:
        torch.nn.Module: Transposed convolution layer class
            - nn.ConvTranspose2d for 2D upsampling
            - nn.ConvTranspose3d for 3D upsampling
            
    Raises:
        ValueError: If dim is not 2 or 3
    """
    if dim == 3:
        return nn.ConvTranspose3d
    elif dim == 2:
        return nn.ConvTranspose2d
    else:
        raise ValueError(f"Unsupported dimension: {dim}. Only 2D and 3D are supported.")


def get_up_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 2,
    stride: int = 2,
    dim: int = 3,
    up_mode: str = "transposed",
):
    """
    Create an upsampling layer for the decoder path.
    
    This function provides different upsampling strategies for the UNet decoder:
    1. Transposed convolution: Learnable upsampling with parameters
    2. Interpolation: Fixed upsampling methods (nearest, linear, etc.)
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (only used for transposed conv)
        kernel_size (int, optional): Kernel size for transposed conv. Defaults to 2.
        stride (int, optional): Stride for transposed conv. Defaults to 2.
        dim (int, optional): Spatial dimensions (2 or 3). Defaults to 3.
        up_mode (str, optional): Upsampling method. Defaults to "transposed".
            - "transposed": Learnable transposed convolution
            - "nearest": Nearest neighbor interpolation
            - "linear": Linear interpolation (1D)
            - "bilinear": Bilinear interpolation (2D)
            - "trilinear": Trilinear interpolation (3D)
            - "bicubic": Bicubic interpolation (2D)
            
    Returns:
        torch.nn.Module: Upsampling layer
        
    Notes:
        - Transposed convolution: More parameters, learnable, may introduce artifacts
        - Interpolation: Fewer parameters, fixed algorithm, smoother results
        - For 3D medical images, "trilinear" interpolation is often preferred
    """
    if up_mode == "transposed":
        # Learnable upsampling using transposed convolution
        return conv_transpose_layer(dim)(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
    else:
        # Fixed upsampling using interpolation
        return nn.Upsample(scale_factor=2.0, mode=up_mode)


def maxpool_layer(dim: int):
    """
    Return the appropriate max pooling layer class based on dimensionality.
    
    Max pooling is used in the encoder path for downsampling while preserving
    the most important features. It reduces spatial dimensions while maintaining
    translation invariance.
    
    Args:
        dim (int): Number of spatial dimensions (2 or 3)
        
    Returns:
        torch.nn.Module: Max pooling layer class
            - nn.MaxPool2d for 2D pooling
            - nn.MaxPool3d for 3D pooling
            
    Raises:
        ValueError: If dim is not 2 or 3
    """
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d
    else:
        raise ValueError(f"Unsupported dimension: {dim}. Only 2D and 3D are supported.")


def get_maxpool_layer(
    kernel_size: int = 2, stride: int = 2, padding: int = 0, dim: int = 2
):
    """
    Create a max pooling layer with specified parameters.
    
    Max pooling reduces spatial dimensions by taking the maximum value within
    each pooling window. This provides downsampling with some translation
    invariance and helps the network focus on the most prominent features.
    
    Args:
        kernel_size (int, optional): Size of pooling window. Defaults to 2.
        stride (int, optional): Pooling stride. Defaults to 2.
            - stride=kernel_size: Non-overlapping pooling (typical)
            - stride<kernel_size: Overlapping pooling
        padding (int, optional): Padding size. Defaults to 0.
        dim (int, optional): Spatial dimensions (2 or 3). Defaults to 2.
        
    Returns:
        torch.nn.Module: Configured max pooling layer
        
    Common Configuration:
        - kernel_size=2, stride=2: Reduces each spatial dimension by half
        - This is the standard configuration in UNet encoder path
    """
    return maxpool_layer(dim=dim)(
        kernel_size=kernel_size, stride=stride, padding=padding
    )


def get_activation(activation: str):
    """
    Create an activation function based on the specified type.
    
    Activation functions introduce non-linearity to the network, enabling it
    to learn complex patterns. Different activations have different properties
    and may work better for different types of data.
    
    Args:
        activation (str): Type of activation function
            - "relu": Rectified Linear Unit - most common, fast
            - "leaky": Leaky ReLU - prevents dying neurons
            - "elu": Exponential Linear Unit - smooth, zero-centered
            
    Returns:
        torch.nn.Module: Configured activation function
        
    Activation Properties:
        - ReLU: f(x) = max(0, x)
            + Fast computation, widely used
            - Can cause "dying neurons" (always output 0)
            
        - Leaky ReLU: f(x) = max(0.1*x, x)
            + Prevents dying neurons
            + Small gradient for negative inputs
            
        - ELU: f(x) = x if x>0, α*(exp(x)-1) if x≤0
            + Smooth function, zero-centered output
            + More computationally expensive
            
    Raises:
        ValueError: If activation type is not supported
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "leaky":
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == "elu":
        return nn.ELU()


def get_normalization(normalization: str, num_channels: int, dim: int):
    """
    Create a normalization layer based on the specified type.
    
    Normalization layers stabilize training by normalizing feature distributions,
    reducing internal covariate shift and enabling higher learning rates.
    
    Args:
        normalization (str): Type of normalization
            - "batch": Batch Normalization
            - "instance": Instance Normalization
            - "group{N}": Group Normalization with N groups (e.g., "group8")
        num_channels (int): Number of input channels
        dim (int): Spatial dimensions (2 or 3)
        
    Returns:
        torch.nn.Module: Configured normalization layer
        
    Normalization Types:
        - Batch Norm: Normalizes across batch dimension
            + Good for large batches, stable training
            - Performance degrades with small batches
            
        - Instance Norm: Normalizes each sample independently
            + Batch size independent, good for style transfer
            + Often preferred for medical imaging
            
        - Group Norm: Normalizes within channel groups
            + Batch size independent
            + Good compromise between batch and instance norm
            + Specify groups as "group8", "group16", etc.
            
    Raises:
        ValueError: If normalization type is not supported
    """
    if normalization == "batch":
        if dim == 3:
            return nn.BatchNorm3d(num_channels)
        elif dim == 2:
            return nn.BatchNorm2d(num_channels)
        else:
            raise ValueError(f"Batch norm not supported for {dim}D")
            
    elif normalization == "instance":
        if dim == 3:
            return nn.InstanceNorm3d(num_channels)
        elif dim == 2:
            return nn.InstanceNorm2d(num_channels)
        else:
            raise ValueError(f"Instance norm not supported for {dim}D")
            
    elif "group" in normalization:
        num_groups = int(
            normalization.partition("group")[-1]
        )  # get the group size from string
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class Concatenate(nn.Module):
    """
    Simple concatenation layer for skip connections.
    
    This module concatenates two tensors along the channel dimension (dim=1).
    It's used in UNet to combine encoder features with decoder features
    through skip connections.
    
    The concatenation preserves spatial information from the encoder path
    while allowing the decoder to use both low-level (encoder) and high-level
    (decoder) features for accurate segmentation.
    """
    
    def __init__(self):
        """
        Initialize the Concatenate module.
        
        No parameters are needed as this is a simple concatenation operation.
        """
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)

        return x


class DownBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 MaxPool.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling: bool = True,
        activation: str = "relu",
        normalization: str = None,
        dim: str = 2,
        conv_mode: str = "same",
    ):
        super().__init__()

        # Store configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        if conv_mode == "same":
            self.padding = 1
        elif conv_mode == "valid":
            self.padding = 0
        self.dim = dim
        self.activation = activation

        # conv layers
        self.conv1 = get_conv_layer(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=self.padding,
            bias=True,
            dim=self.dim,
        )
        self.conv2 = get_conv_layer(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=self.padding,
            bias=True,
            dim=self.dim,
        )

        # Max pooling layer for downsampling (optional)
        if self.pooling:
            self.pool = get_maxpool_layer(
                kernel_size=2, stride=2, padding=0, dim=self.dim
            )

        # Activation layers
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # Normalization layers (optional)
        if self.normalization:
            self.norm1 = get_normalization(
                normalization=self.normalization,
                num_channels=self.out_channels,
                dim=self.dim,
            )
            self.norm2 = get_normalization(
                normalization=self.normalization,
                num_channels=self.out_channels,
                dim=self.dim,
            )

    def forward(self, x):
        y = self.conv1(x)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1
        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # activation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2

        before_pooling = y  # save the outputs before the pooling operation
        if self.pooling:
            y = self.pool(y)  # pooling
        return y, before_pooling


class UpBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 UpConvolution/Upsample.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu",
        normalization: str = None,
        dim: int = 3,
        conv_mode: str = "same",
        up_mode: str = "transposed",
    ):
        """
        Initialize the UpBlock.
        
        Creates all necessary layers for upsampling, concatenation, and
        feature processing in the decoder path.
        """
        super().__init__()

        # Store configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        if conv_mode == "same":
            self.padding = 1
        elif conv_mode == "valid":
            self.padding = 0
        self.dim = dim
        self.activation = activation
        self.up_mode = up_mode

        # upconvolution/upsample layer
        self.up = get_up_layer(
            self.in_channels,
            self.out_channels,
            kernel_size=2,
            stride=2,
            dim=self.dim,
            up_mode=self.up_mode,
        )

        # Convolution layers
        # conv0: Used only for interpolation upsampling to reduce channels
        self.conv0 = get_conv_layer(
            self.in_channels,
            self.out_channels,
            kernel_size=1,  # 1x1 convolution for channel reduction
            stride=1,
            padding=0,
            bias=True,
            dim=self.dim,
        )
        
        # conv1: Processes concatenated features (2*out_channels from concat)
        self.conv1 = get_conv_layer(
            2 * self.out_channels,  # Double channels due to concatenation
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=self.padding,
            bias=True,
            dim=self.dim,
        )
        
        # conv2: Second convolution for feature refinement
        self.conv2 = get_conv_layer(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=self.padding,
            bias=True,
            dim=self.dim,
        )

        # Activation layers
        self.act0 = get_activation(self.activation)
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # Normalization layers (optional)
        if self.normalization:
            self.norm0 = get_normalization(
                normalization=self.normalization,
                num_channels=self.out_channels,
                dim=self.dim,
            )
            self.norm1 = get_normalization(
                normalization=self.normalization,
                num_channels=self.out_channels,
                dim=self.dim,
            )
            self.norm2 = get_normalization(
                normalization=self.normalization,
                num_channels=self.out_channels,
                dim=self.dim,
            )

        # Concatenation layer for skip connections
        self.concat = Concatenate()

    def forward(self, encoder_layer, decoder_layer):
        """Forward pass
        Arguments:
            encoder_layer: Tensor from the encoder pathway
            decoder_layer: Tensor from the decoder pathway (to be up'd)
        """
        up_layer = self.up(decoder_layer)  # up-convolution/up-sampling
        cropped_encoder_layer, dec_layer = autocrop(encoder_layer, up_layer)  # cropping

        if self.up_mode != "transposed":
            # We need to reduce the channel dimension with a conv layer
            up_layer = self.conv0(up_layer)  # convolution 0
        up_layer = self.act0(up_layer)  # activation 0
        if self.normalization:
            up_layer = self.norm0(up_layer)  # normalization 0

        merged_layer = self.concat(up_layer, cropped_encoder_layer)  # concatenation
        y = self.conv1(merged_layer)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1
        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # acivation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2
        return y


class UNet(nn.Module):
    """
    activation: 'relu', 'leaky', 'elu'
    normalization: 'batch', 'instance', 'group{group_size}'
    conv_mode: 'same', 'valid'
    dim: 2, 3
    up_mode: 'transposed', 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        n_blocks: int = 4,
        start_filters: int = 32,
        activation: str = "leaky",
        normalization: str = "instance",
        conv_mode: str = "same",
        dim: int = 2,
        up_mode: str = "transposed",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filters = start_filters
        self.activation = activation
        self.normalization = normalization
        self.conv_mode = conv_mode
        self.dim = dim
        self.up_mode = up_mode

        self.down_blocks = []
        self.up_blocks = []

        # Create encoder path (contracting/downsampling path)
        for i in range(self.n_blocks):
            # Calculate input and output channels for this level
            num_filters_in = self.in_channels if i == 0 else num_filters_out
            num_filters_out = self.start_filters * (2 ** i)  # Double filters each level
            
            # Last block (bottleneck) doesn't have pooling
            pooling = True if i < self.n_blocks - 1 else False

            # Create encoder block
            down_block = DownBlock(
                in_channels=num_filters_in,
                out_channels=num_filters_out,
                pooling=pooling,
                activation=self.activation,
                normalization=self.normalization,
                conv_mode=self.conv_mode,
                dim=self.dim,
            )

            self.down_blocks.append(down_block)

        # Create decoder path (expanding/upsampling path)
        # Requires only n_blocks-1 blocks (bottleneck doesn't need corresponding decoder)
        for i in range(n_blocks - 1):
            # Calculate input and output channels for this level
            num_filters_in = num_filters_out  # From previous level
            num_filters_out = num_filters_in // 2  # Halve filters each level

            # Create decoder block
            up_block = UpBlock(
                in_channels=num_filters_in,
                out_channels=num_filters_out,
                activation=self.activation,
                normalization=self.normalization,
                conv_mode=self.conv_mode,
                dim=self.dim,
                up_mode=self.up_mode,
            )

            self.up_blocks.append(up_block)

        # Final 1x1 convolution to produce output classes
        # Maps from final decoder features to desired number of output channels
        self.conv_final = get_conv_layer(
            num_filters_out,    # Input: final decoder features
            self.out_channels,  # Output: number of classes
            kernel_size=1,      # 1x1 convolution (point-wise)
            stride=1,
            padding=0,          # No padding needed for 1x1 conv
            bias=True,
            dim=self.dim,
        )

        # Convert lists to ModuleList for proper parameter registration
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        # Initialize network weights
        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(
            module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)
        ):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(
            module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)
        ):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(
        self,
        method_weights=nn.init.xavier_uniform_,
        method_bias=nn.init.zeros_,
        kwargs_weights={},
        kwargs_bias={},
    ):
        """
        Initialize all network parameters.
        
        This method applies weight and bias initialization to all applicable
        layers in the network. Proper initialization is crucial for stable
        training and convergence.
        
        Args:
            method_weights: Weight initialization method. Defaults to Xavier uniform.
            method_bias: Bias initialization method. Defaults to zeros.
            kwargs_weights (dict): Additional arguments for weight initialization.
            kwargs_bias (dict): Additional arguments for bias initialization.
            
        Default Initialization:
            - Weights: Xavier uniform (good for sigmoid/tanh activations)
            - Biases: Zeros (standard practice)
            
        Alternative Options:
            - Kaiming/He initialization for ReLU activations:
              method_weights=nn.init.kaiming_uniform_
            - Normal initialization:
              method_weights=nn.init.normal_, kwargs_weights={'std': 0.01}
        """
        for module in self.modules():
            self.weight_init(
                module, method_weights, **kwargs_weights
            )  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    def forward(self, x: torch.tensor):
        encoder_output = []
        #print("\tIn Model: input size", x.size()) # printing to check whether data parallel works
        # Encoder pathway
        for module in self.down_blocks:
            x, before_pooling = module(x)
            # Save features before pooling for skip connections
            encoder_output.append(before_pooling)

        # Decoder pathway
        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)

        return x

    def __repr__(self):
        """
        String representation of the UNet model.
        
        Returns a formatted string showing the model's configuration parameters,
        useful for debugging and logging.
        
        Returns:
            str: Formatted string with model configuration
        """
        # Extract public attributes (excluding private and training-related)
        attributes = {
            attr_key: self.__dict__[attr_key]
            for attr_key in self.__dict__.keys()
            if "_" not in attr_key[0] and "training" not in attr_key
        }
        
        # Create formatted representation
        d = {self.__class__.__name__: attributes}
        return f"{d}"




if __name__ == "__main__":
    device = torch.device("cuda")
    network = UNet(in_channels=1, out_channels=2, dim=3, start_filters=32)
    network = network.to(device)
