"""
Name: 1. Varun Reddy Patlolla
      2. Anand Pinnamaneni

Date: April 6th, 2026

Description: This files us used to automate the tests , changing three hyper paramaters that are activation,epoch and batch size.


"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import csv

device = "cpu"
print(f"Using {device} device")

class Net(nn.Module):
    def __init__(self, activation_name='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size = 5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50) 
        self.fc2 = nn.Linear(50, 10)  
        self.activation_name = activation_name

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
        
        if self.activation_name == 'relu':
            x = F.relu(x)
        elif self.activation_name == 'tanh':
            x = F.tanh(x)
        elif self.activation_name == 'sigmoid':
            x = torch.sigmoid(x)
            
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

def preprocessDataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transform
    )

    figure = plt.figure(figsize = (8,8))
    cols, rows = 3,2

    for i in range(6):
        img, label = test_data[i]
        figure.add_subplot(rows, cols, i+1)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap = "gray")

    figure.suptitle("First 6 images from test set", fontsize=16)
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/task1A.png", bbox_inches = "tight", dpi=300)

    return training_data, test_data


def train_loop(epoch, training_dataloader, model, loss_fn, optimizer, batch_size, train_losses, train_counter):
    model.train()
    size = len(training_dataloader.dataset)
    for batch, (X,y) in enumerate(training_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            train_losses.append(loss)
            train_counter.append((batch*batch_size) + ((epoch-1)*len(training_dataloader.dataset)))

def test_loop(test_dataloader, model, loss_fn, test_losses):
    model.eval()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    test_losses.append(test_loss)
    correct /= size
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return test_loss 

def run_experiment(training_data, test_data, activation_name, epochs, batch_size):
    model = Net(activation_name=activation_name).to(device)
    learning_rate = 1e-2
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    test_losses = []
    test_loop(test_dataloader, model, loss_fn, test_losses) 
    
    for t in range(epochs):
        train_losses = []
        train_counter = []
        train_loop(t+1, train_dataloader, model, loss_fn, optimizer, batch_size, train_losses, train_counter)
        final_loss = test_loop(test_dataloader, model, loss_fn, test_losses)
        
    return final_loss

if __name__ == "__main__":
    print("Loading Fashion MNIST...")
    training_data, test_data = preprocessDataset()

    experiments = [
        # Baseline
        {"activation": "relu", "epochs": 5, "batch_size": 64},
        
        # Dimension 1: Activation Function
        {"activation": "tanh", "epochs": 5, "batch_size": 64},
        {"activation": "sigmoid", "epochs": 5, "batch_size": 64},
        
        # Dimension 2: Epochs
        {"activation": "relu", "epochs": 10, "batch_size": 64},
        {"activation": "relu", "epochs": 15, "batch_size": 64},
        {"activation": "relu", "epochs": 20, "batch_size": 64},
        {"activation": "relu", "epochs": 25, "batch_size": 64},


        
        # Dimension 3: Batch Size
        {"activation": "relu", "epochs": 5, "batch_size": 32},
        {"activation": "relu", "epochs": 5, "batch_size": 128},
    ]

    results = []

    for i, exp in enumerate(experiments):
        print(f"\n--- Running Experiment {i+1}/{len(experiments)} ---")
        print(f"Activation: {exp['activation']}, Epochs: {exp['epochs']}, Batch Size: {exp['batch_size']}")
        
        final_loss = run_experiment(
            training_data, 
            test_data, 
            activation_name=exp['activation'], 
            epochs=exp['epochs'],
            batch_size=exp['batch_size']
        )
        
        exp["final_test_loss"] = final_loss
        results.append(exp)

    csv_file = "results/experiment_results.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["activation", "epochs", "batch_size", "final_test_loss"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAll experiments complete! Results saved to {csv_file}")