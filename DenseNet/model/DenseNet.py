import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module): # Dimensionality reduction purpose -> Reduce computation
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False) # Reduces the number of input channels (dimensionality reduction).
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False) # Processes the feature maps with a small receptive field.

    def forward(self, x):
    	out = self.conv1(F.relu(self.bn1(x)))
    	out = self.conv2(F.relu(self.bn2(out)))
    	out = torch.cat([x, out], 1)

    	return out

    
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels) 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) # 1x1 Conv; Reduces the num. of channels.
        self.pool = nn.AvgPool2d(2) # 2x2 average pooling; Reduces the spatial dimensions by half.

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

    
class DenseBlock(nn.Module):
    def __init__(self, n_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers) # Combines the bottleneck layers into a sequential container.
    def forward(self, x):
    	# print(f"DenseBlock input shape: {x.shape}")  # Check shape before entering dense blocks
    	out = self.block(x)
    	# print(f"DenseBlock output shape: {out.shape}")  # Check output shape after dense blocks
    	return out


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_layers=[6, 12, 24, 16], num_classes=10, dropout_prob=0.3):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.dropout_prob = dropout_prob  # Dropout probability

        num_init_features = 2 * growth_rate
        self.conv1 = nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)  # Adjusted to 1 input channel
        self.bn1 = nn.BatchNorm2d(num_init_features)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = nn.Sequential() # Sequentially stack the dense blocks and transition layers that form the core part of the DenseNet architecture.
        num_features = num_init_features
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_layers) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.bn2 = nn.BatchNorm2d(num_features)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1)) # Reduces each 7x7 feature map to a 1x1 feature map by averaging the values. This adapts to any input size, making it very flexible.
        self.dropout = nn.Dropout(p=dropout_prob)  
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.relu(self.bn1(x)))
        x = self.features(x)
        x = self.pool2(F.relu(self.bn2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x







