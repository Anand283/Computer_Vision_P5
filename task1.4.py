import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# 1. We MUST define the class structure again so PyTorch knows what to load
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main():
    # Load data (same transform as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    # 2. Load the model
    model = Net()
    model.load_state_dict(torch.load("mnist_model_weights.pth"))
    model.eval() # CRITICAL: Sets dropout to evaluation mode

    # 3. Process the first 10 examples
    print(f"{'Idx':<4} | {'Network Outputs (0-9)':<50} | {'Max Idx':<8} | {'Label':<5}")
    print("-" * 80)

    fig = plt.figure(figsize=(10, 10))
    
    with torch.no_grad(): # Disable gradient calculation for speed
        for i in range(10):
            img, label = test_data[i]
            
            # Prepare image for network (Add batch dimension: [1, 1, 28, 28])
            output = model(img.unsqueeze(0))
            
            # Format the 10 outputs to 2 decimal places
            formatted_output = [f"{val:.2f}" for val in output[0].tolist()]
            
            # Get prediction (Index of max value)
            prediction = output.data.max(1, keepdim=True)[1].item()

            # Print table row
            print(f"{i:<4} | {str(formatted_output):<50} | {prediction:<8} | {label:<5}")

            # Plot first 9 images
            if i < 9:
                plt.subplot(3, 3, i + 1)
                plt.tight_layout()
                plt.imshow(img.squeeze(), cmap='gray', interpolation='none')
                plt.title(f"Prediction: {prediction}")
                plt.xticks([])
                plt.yticks([])

    # 1. Ensure the results directory exists
    os.makedirs("results", exist_ok=True)
    
    # 2. Save the figure
    plt.savefig("results/test_predictions_grid.png", bbox_inches="tight", dpi=300)
    print("\n[INFO] Plot saved to results/test_predictions_grid.png")

    plt.show()

if __name__ == "__main__":
    main()