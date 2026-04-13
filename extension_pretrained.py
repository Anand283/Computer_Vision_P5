"""
Name: 1. Varun Reddy Patlolla
      2. Anand Pinnamaneni

Date: April 7th, 2026

Description: Extension - Analyze first convolutional layers of a pre-trained ResNet18 network
             and compare with our trained MNIST CNN filters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


# MNIST CNN class (same as MNIST.py)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


def visualize_resnet_filters():
    """Load pre-trained ResNet18 and visualize its first conv layer filters."""
    # Load pre-trained ResNet18
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.eval()

    # Print model structure (first few layers)
    print("ResNet18 First Layer:")
    print(resnet.conv1)
    print(f"  Weight shape: {resnet.conv1.weight.shape}")
    print(f"  → {resnet.conv1.weight.shape[0]} filters, "
          f"{resnet.conv1.weight.shape[1]} input channels, "
          f"{resnet.conv1.weight.shape[2]}x{resnet.conv1.weight.shape[3]} size")

    # Get first layer weights: shape [64, 3, 7, 7]
    weights = resnet.conv1.weight.data.clone()

    # Visualize first 16 filters (4x4 grid)
    # For RGB filters, we show each channel separately OR average across channels
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle("ResNet18 - First Conv Layer Filters (first 32 of 64)", fontsize=14)

    with torch.no_grad():
        for i in range(32):
            row = i // 8
            col = i % 8
            # Average across RGB channels for visualization
            filt = weights[i].mean(dim=0).cpu().numpy()
            axes[row, col].imshow(filt, cmap='viridis')
            axes[row, col].set_title(f"F{i}", fontsize=8)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/resnet18_conv1_filters.png", bbox_inches="tight", dpi=300)
    print("\nSaved: results/resnet18_conv1_filters.png")
    plt.close()

    # Also show some filters with their RGB channels separated
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    fig.suptitle("ResNet18 - First 8 Filters (R, G, B channels)", fontsize=14)
    channel_names = ['Red', 'Green', 'Blue']

    with torch.no_grad():
        for ch in range(3):
            for i in range(8):
                filt = weights[i, ch].cpu().numpy()
                axes[ch, i].imshow(filt, cmap='gray')
                if i == 0:
                    axes[ch, i].set_ylabel(channel_names[ch], fontsize=10)
                axes[ch, i].set_title(f"F{i}" if ch == 0 else "", fontsize=8)
                axes[ch, i].set_xticks([])
                axes[ch, i].set_yticks([])

    plt.tight_layout()
    plt.savefig("results/resnet18_rgb_channels.png", bbox_inches="tight", dpi=300)
    print("Saved: results/resnet18_rgb_channels.png")
    plt.close()

    return weights


def visualize_mnist_filters():
    """Load our trained MNIST CNN and visualize its first conv layer filters."""
    model = Net()
    model.load_state_dict(torch.load("mnist_model_weights.pth", map_location="cpu"))
    model.eval()

    print("\nMNIST CNN First Layer:")
    print(model.conv1)
    print(f"  Weight shape: {model.conv1.weight.shape}")

    weights = model.conv1.weight.data.clone()

    # Visualize all 10 filters (similar to Task 2)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("MNIST CNN - First Conv Layer Filters (10 filters, 5x5)", fontsize=14)

    with torch.no_grad():
        for i in range(10):
            row = i // 5
            col = i % 5
            filt = weights[i, 0].cpu().numpy()
            axes[row, col].imshow(filt, cmap='gray')
            axes[row, col].set_title(f"Filter {i}", fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.tight_layout()
    plt.savefig("results/mnist_conv1_filters.png", bbox_inches="tight", dpi=300)
    print("Saved: results/mnist_conv1_filters.png")
    plt.close()

    return weights


def compare_filter_stats(mnist_weights, resnet_weights):
    """Print comparison statistics between MNIST and ResNet filters."""
    print("\n" + "=" * 60)
    print("Filter Comparison Statistics")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'MNIST CNN':<20} {'ResNet18':<20}")
    print("-" * 70)
    print(f"{'Number of filters':<30} {mnist_weights.shape[0]:<20} {resnet_weights.shape[0]:<20}")
    print(f"{'Input channels':<30} {mnist_weights.shape[1]:<20} {resnet_weights.shape[1]:<20}")
    print(f"{'Filter size':<30} {f'{mnist_weights.shape[2]}x{mnist_weights.shape[3]}':<20} "
          f"{f'{resnet_weights.shape[2]}x{resnet_weights.shape[3]}':<20}")
    print(f"{'Total parameters':<30} {mnist_weights.numel():<20} {resnet_weights.numel():<20}")
    print(f"{'Weight range':<30} "
          f"[{mnist_weights.min():.4f}, {mnist_weights.max():.4f}]   "
          f"[{resnet_weights.min():.4f}, {resnet_weights.max():.4f}]")
    print(f"{'Weight std':<30} {mnist_weights.std():.4f}{'':<16} {resnet_weights.std():.4f}")


# Main function
def main(argv):
    print("=" * 60)
    print("Extension: Pre-trained Network Layer Analysis")
    print("=" * 60)

    # 1. Visualize ResNet18 filters
    resnet_weights = visualize_resnet_filters()

    # 2. Visualize MNIST CNN filters
    mnist_weights = visualize_mnist_filters()

    # 3. Compare statistics
    compare_filter_stats(mnist_weights, resnet_weights)

    print("\nDone! All visualizations saved to results/")


if __name__ == "__main__":
    main(sys.argv)
