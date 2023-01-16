import torch
import torch.nn.functional as F
from math import floor
from torch import nn 

def mask_features(x, masks):
    masks_adapted = masks.repeat((1,x.shape[1],1))
    return x * masks_adapted

# Reimplementation of the model ProtCNN from "Using deep learning to annotate the protein universe", Bileschi et. al. Nature (2022)
class CustomConv(nn.Module):
 
    def __init__(self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int,
            dilation: float
        ) :
        """ Custom convolution layer. 
        Zero-out the features before and after applying the 
        convolution given masks that account for the real length of each sequence.

        Args:
            in_channels (int): Number of channels of the inputs
            out_channels (int):  Number of channels of the outputs
            kernel_size (int): Size of the kernel
            dilation (float): Dilation of the convolution 
        """
        super().__init__()

        self.layer = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            stride=1, 
            padding='same'
        )
    
    def forward(self, x, masks):
        x_masked = mask_features(x, masks)
        x_conv = self.layer(x_masked)
        x_conv_masked = mask_features(x_conv, masks)
        return x_conv_masked

class ResidualBlock(nn.Module):

    def __init__(self, 
        num_layer: int,
        in_channels: int, 
        filters: int, 
        kernel_size: int, 
        dilation_rate: float, 
        bottleneck_factor: float
        ) :
        """ Residual Block. 

        Args:
            num_layer (int):  index of the given layer 
            in_channels (int):  Number of channels of the inputs
            filters (int): Number of filters
            kernel_size (int): Size of the kernel
            dilation_rate (float): rate of the dilation
            bottleneck_factor (float): bottleneck factor
        """
        super().__init__()

        self.batch_norm1 = nn.BatchNorm1d(num_features=in_channels)
        bottleneck_dim = floor(bottleneck_factor * filters)
        updated_dilation = max(1, dilation_rate * num_layer)
        self.dilate_conv = CustomConv(
            in_channels=in_channels, 
            out_channels= bottleneck_dim, 
            kernel_size=kernel_size, 
            dilation=updated_dilation
        )
        self.batch_norm2 = nn.BatchNorm1d(num_features=bottleneck_dim)
        self.channels_conv = CustomConv(
            in_channels=bottleneck_dim, 
            out_channels=filters, 
            kernel_size=1, 
            dilation=1
        )

    def forward(self, x, masks):
        features = F.relu(self.batch_norm1(x))
        features = self.dilate_conv(features, masks)
        features = F.relu(self.batch_norm2(features)) 
        residual = self.channels_conv(features, masks)
        output = residual + x # skip-connection
        return output

class ProtCNN(nn.Module): 

    def __init__(self, 
            in_channels: int,
            num_classes: int, 
            filters: int, 
            kernel_size: int, 
            dilation_rate: float, 
            bottleneck_factor: float, 
            num_residual_block: int
            ) -> None:
        """ The ProtCNN architecture as presented in "Using deep learning to annotate the protein universe", Bileschi et. al. Nature (2022). 

        Args:
            in_channels (int): Number of channels of the inputs
            num_classes (int): Number of classes 
            filters (int): Number of filters 
            kernel_size (int): Size of the kernel of the convolution layers
            dilation_rate (float): Dilation rate
            bottleneck_factor (float): Bottleneck factor
            num_residual_block (int): Number of residual blocks
        """
        super().__init__()

        self.initial_conv = CustomConv(
            in_channels=in_channels, 
            out_channels=filters, 
            kernel_size=kernel_size, 
            dilation=1
        )

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                num_layer=i, 
                in_channels=filters, 
                filters=filters, 
                kernel_size=kernel_size, 
                dilation_rate=dilation_rate, 
                bottleneck_factor=bottleneck_factor
            ) for i in range(num_residual_block)
        ])
        self.linear = nn.Linear(
            in_features=filters, 
            out_features=num_classes
        )

    def forward(self, input, masks):
        features = self.initial_conv(input, masks)
        for layer in self.residual_blocks: 
            features = layer(features, masks)

        features = features.max(dim=-1).values # pool along the length of the sequence 
        logits = self.linear(features)
        probs = F.softmax(logits, dim=-1)
        return probs 

    