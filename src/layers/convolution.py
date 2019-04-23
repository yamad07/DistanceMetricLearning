import torch
import torch.nn as nn

def convolution(input_dim, output_dim):
    return nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
