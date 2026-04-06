"""
Name: 1. Varun Reddy Patlolla
      2. Anand Pinnamaneni

Date: April 6th, 2026

Description: 


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


#Get Device for training
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#========== NeuralNetwork Class ==========
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

"""
Brief Description: 

"""
def preprocessDataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    # Plotting Data from Test Set
    figure = plt.figure(figsize = (8,8))
    cols, rows = 3,2

    for i in range(6):
        img, label = test_data[i]

        figure.add_subplot(rows, cols, i+1)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap = "gray")

    figure.suptitle("First 6 images from test set", fontsize=16)
    plt.savefig("results/task1A.png", bbox_inches = "tight", dpi=300)
    plt.show()

    return training_data, test_data


def train_loop(epoch,training_dataloader, model, loss_fn, optimizer,batch_size, train_losses, train_counter):

    size = len(training_dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    
    
    for batch, (X,y) in enumerate(training_dataloader):

        # X = set of 64 inpu timage data 
        # Y = set of 64 corresponding targets

        # Move data to device
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred,y)

        # Backpropagation -- Compute the gradient of loss with respect to parameters and send it back
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ----- Print after 100 batches -------
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            train_losses.append(loss)
            train_counter.append(
               (batch*64) + ((epoch-1)*len(training_dataloader.dataset)))
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(test_dataloader, model, loss_fn, test_losses):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in test_dataloader:

            # Move data to the GPU/CPU
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    test_losses.append(test_loss)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def Training_Plot(train_losses, train_counter, test_losses, test_counter):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    
    #save and show the plot
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_performance.png", bbox_inches="tight", dpi=300)
    plt.show()

def main(argv):
    training_data, test_data = preprocessDataset()

    #initialize the model
    model = Net().to(device)

    #----- HyperParameters of NN -----
    learning_rate = 1e-2
    batch_size = 64
    epochs = 5

    # Loss function
    loss_fn = nn.NLLLoss()

    #Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # ----- Dataloaders -----
    train_dataloader = DataLoader(training_data,batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

    # Lists to store data forplotting
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_dataloader.dataset) for i in range(epochs + 1)]

    # ----- Training Loop ------
    test_loop(test_dataloader, model, loss_fn, test_losses) #calling just to create initial dot for plotting
    for t in range(epochs):
        print(f"Epoch {t+1}\n----------")
        train_loop(t+1, train_dataloader,model, loss_fn, optimizer,batch_size, train_losses, train_counter)
        test_loop(test_dataloader, model, loss_fn, test_losses)
    
    print("Completed!")

    # Save the model
    """
    state_dict is a python dictionary. Pytorch models store the learned parameters in 
    internal state dictionary: state_dict
    """

    torch.save(model.state_dict(), "mnist_model_weights.pth")
    print("Saved PyTorch Model State to mnist_model_weights.pth")

    # plot

    Training_Plot(train_losses, train_counter, test_losses, test_counter)


if __name__ == "__main__":
    main(sys.argv)

