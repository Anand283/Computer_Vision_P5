"""
Name: 1. Varun Reddy Patlolla
      2. Anand Pinnamaneni

Date: April 8th, 2026

Description: This file is to extract the text filters out of the convolution layer one, 
             it uses matplotlib funcition to plot the images on the window.

"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size = 5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear( 320 ,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

    
# loading the minst trained data model 
def main(argv):
    model = Net()
    model.load_state_dict(torch.load("mnist_model_weights.pth"))
    model.eval() 

    # the filters used in layer one
    with torch.no_grad():
        weights = model.conv1.weight.detach().numpy()
        
    print("\nAnalyzing First Layer (conv1)")
    print(f"Shape of the first layer weights: {weights.shape}")
    for i in range(10):
        print(f"\nWeights of Filter {i}:")
        print(weights[i, 0])
    
    fig1 = plt.figure(figsize=(10, 8))
    for i in range(10):
        filter_2d = weights[i, 0] 
        
        ax = fig1.add_subplot(3, 4, i + 1)
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        plt.imshow(filter_2d, cmap='gray')
        ax.set_title(f"Filter {i}")
    
    fig1.suptitle("First Layer Filters (10 5x5 filters)", fontsize=16)
    


    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    
    image_tensor, label = train_data[0] # Index 0 is '5'
    image_cv = image_tensor.squeeze().numpy()

    # Side by side results
    fig2 = plt.figure(figsize=(8, 10))
    
    for i in range(10):
        filter_2d = weights[i, 0] 
        filtered_image = cv2.filter2D(image_cv, -1, filter_2d)
        
        filter_pos = (i // 2) * 4 + (i % 2) * 2 + 1
        
        ax1 = fig2.add_subplot(5, 4, filter_pos)
        ax1.set_xticks([]) 
        ax1.set_yticks([]) 
        ax1.imshow(filter_2d, cmap='gray')
        
        ax2 = fig2.add_subplot(5, 4, filter_pos + 1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow(filtered_image, cmap='gray')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    print("Launching both windows!")
    plt.show()

if __name__ == "__main__":
    main(sys.argv)