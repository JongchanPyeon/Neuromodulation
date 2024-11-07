# Utilize the encoding part of the pre-trained DensetNet for the US array shape estimation.
# Use the pre-trained encoder as the encoder of UNet. 
# No bottleneck layer
# Train Decoder part only.
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

densenet_path = os.path.abspath("../DenseNet/model")
sys.path.append(densenet_path)
from DenseNet import DenseNet
from UNet import DecoderBlock


## Define DenseNet Encoder with UNet Decoder : No bottleneck layer
class Hybrid_DenseNet_UNet(nn.Module):
    """
    Combines a pre-trained DenseNet encoder with a U-Net style decoder for image segmentation tasks. No bottleneck layer
    
    Attributes:
        encoder (nn.Module): The DenseNet feature extractor used as the encoder.
        conv1 (nn.Conv2d): First convolutional layer of the DenseNet.
        bn1 (nn.BatchNorm2d): First BatchNorm layer of the DenseNet.
        dec4, dec3, dec2, dec1 (DecoderBlock): Decoder blocks to upsample the features with skip connections.
        final_conv (nn.Conv2d): 1x1 convolution to reduce channels to the number of output classes.
        upsample_final (nn.Upsample): Upsampling layer to ensure the final output matches the desired size.
    """
    def __init__(self, densenet, out_channels=1, output_size=(256, 256)):
        """
        Initializes the Hybrid DenseNet-UNet model by configuring the encoder and defining the U-Net decoder layers.

        Args:
            densenet (nn.Module): Pre-trained DenseNet model to use as the encoder.
            out_channels (int): Number of output channels for the final segmentation map (default=1).
            output_size (tuple): Desired output spatial dimensions (default=(256, 256)).
        """
        super(Hybrid_DenseNet_UNet, self).__init__()

        ## Use the pre-trained DenseNet encoder 
        self.encoder = densenet.features
        self.conv1 = densenet.conv1  # DenseNet's first conv layer
        self.bn1 = densenet.bn1      # DenseNet's first BatchNorm layer
        for param in self.encoder.parameters(): # Freeze DenseNet encoder parameters
            param.requires_grad = False 

        ## Define the decoder as per the original U-Net structure
        self.dec4 = DecoderBlock(1024, 512)  # 1024 channels to 512
        self.dec3 = DecoderBlock(512, 256)   # 512 channels to 256
        self.dec2 = DecoderBlock(256, 128)   # 256 channels to 128
        self.dec1 = DecoderBlock(128, 64)    # 128 channels to 64

        ## Final output layer (1x1 convolution to reduce channels to the number of output classes)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        ## Add an upsample layer to ensure the final output size matches the input
        self.upsample_final = nn.Upsample(size=output_size, mode='bilinear', align_corners=False)

        
    def forward(self, x):
        """
        Defines the forward pass through the Hybrid DenseNet-UNet model, including encoder, decoder, and final upsampling.

        Args:
            x (torch.Tensor): Input tensor, typically an image with shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: The segmentation output with dimensions (batch_size, out_channels, output_size).
        """

        ## Encoder pass through DenseNet's initial layers
        x = F.relu(self.bn1(self.conv1(x)))

        ## Pass through DenseNet's encoder blocks     
        enc1 = self.encoder[1](self.encoder[0](x))  # DenseBlock1 + Transition1 ; 64 to 128
        enc2 = self.encoder[3](self.encoder[2](enc1))  # DenseBlock2 + Transition2 ; 128 to 256
        enc3 = self.encoder[5](self.encoder[4](enc2))  # DenseBlock3 + Transition3 ; 256 to 512
        enc4 = self.encoder[6](enc3)  # DenseBlock4 (no Transition4) ; 512 to 1024

        ## Decoder with skip connections
        dec4 = self.dec4(enc4, enc3)
        dec3 = self.dec3(dec4, enc2)
        dec2 = self.dec2(dec3, enc1)
        dec1 = self.dec1(dec2, x)

        ## Final convolution and upsampling
        out = self.final_conv(dec1)
        out = self.upsample_final(out)  # Ensure output size is 256x256

        return out



