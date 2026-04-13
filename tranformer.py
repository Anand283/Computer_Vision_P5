"""
Name: 1. Varun Reddy Patlolla
      2. Anand Pinnamaneni

Date: April 6th, 2026

Description: Transformer-based model for MNIST digit recognition


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


#========== Transformer Network Class ==========
class NetTransformer(nn.Module):
    def __init__(self, patch_size=7, embed_dim=64, num_heads=4, num_layers=2, num_classes=10):
        super().__init__()
        
        # Parameters
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (28 // patch_size) ** 2  # For 28x28 image with 7x7 patches = 16 patches
        
        # Linear layer to convert flattened patches to embeddings
        # Each patch is patch_size x patch_size = 49 pixels (for 7x7)
        self.patch_embedding = nn.Linear(patch_size * patch_size, embed_dim)
        
        # Positional embeddings - learnable parameters
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True  # Important: input shape will be (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc1 = nn.Linear(embed_dim, 50)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        batch_size = x.shape[0]
        
        # Step 1: Split image into patches
        # Unfold creates patches: (batch, channels, num_patches_h, num_patches_w, patch_h, patch_w)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # Reshape to: (batch, num_patches, patch_size*patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size)
        
        # Step 2: Convert patches to token embeddings
        # patches shape: (batch, num_patches, patch_size^2)
        # After embedding: (batch, num_patches, embed_dim)
        token_embeddings = self.patch_embedding(patches)
        
        # Step 3: Add positional embeddings
        token_embeddings = token_embeddings + self.position_embedding
        
        # Step 4: Pass through transformer encoder
        # Output shape: (batch, num_patches, embed_dim)
        encoded = self.transformer_encoder(token_embeddings)
        
        # Step 5: Generate single representation by averaging all patches
        # Shape: (batch, embed_dim)
        pooled = encoded.mean(dim=1)
        
        # Step 6: Classification layers
        x = self.fc1(pooled)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        
        return x


#========== CNN Network Class (original) ==========
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
Brief Description: Preprocess and visualize MNIST dataset
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
    model.train()
    
    for batch, (X,y) in enumerate(training_dataloader):

        # Move data to device
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred,y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print after 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            train_losses.append(loss)
            train_counter.append(
               (batch*64) + ((epoch-1)*len(training_dataloader.dataset)))
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def Training_Plot(train_losses, train_counter, test_losses, test_counter):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_performance_transformer.png", bbox_inches="tight", dpi=300)
    plt.show()


def main(argv):
    training_data, test_data = preprocessDataset()

    # Initialize the TRANSFORMER model (change this line to switch between models)
    model = NetTransformer(patch_size=7, embed_dim=64, num_heads=4, num_layers=2).to(device)
    # model = Net().to(device)  # Use this for CNN
    
    print(model)  # Print model architecture

    # HyperParameters
    learning_rate = 1e-3  # Transformers often need lower learning rate
    batch_size = 64
    epochs = 10  # May need more epochs for transformer

    # Loss function
    loss_fn = nn.NLLLoss()

    # Optimizer - Adam works better for transformers
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Dataloaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Lists to store data for plotting
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_dataloader.dataset) for i in range(epochs + 1)]

    # Training Loop
    test_loop(test_dataloader, model, loss_fn, test_losses)
    for t in range(epochs):
        print(f"Epoch {t+1}\n----------")
        train_loop(t+1, train_dataloader, model, loss_fn, optimizer, batch_size, train_losses, train_counter)
        test_loop(test_dataloader, model, loss_fn, test_losses)
    
    print("Completed!")

    # Save the model
    torch.save(model.state_dict(), "mnist_transformer_weights.pth")
    print("Saved PyTorch Model State to mnist_transformer_weights.pth")

    # Plot
    Training_Plot(train_losses, train_counter, test_losses, test_counter)


if __name__ == "__main__":
    main(sys.argv)