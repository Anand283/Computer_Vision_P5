"""
Name: 1. Varun Reddy Patlolla
      2. Anand Pinnamaneni

Date: April 8th, 2026

Description: This file is to train the same model used for the MINST data set on Greek letters
             alph,beta and gamma.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

# Same model as task 1
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


# transfor used on the image provided in assignment
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

def main(argv):
    model = Net()
    model.load_state_dict(torch.load("mnist_model_weights.pth"))

    for param in model.parameters():
        param.requires_grad = False

    # We remove the old nn.Linear(50, 10) and replace it with 3 outputs (alpha, beta, gamma)
    # By default, new layers have requires_grad=True, so ONLY this layer will learn!
    model.fc2 = nn.Linear(50, 3)
    
    print("\nModified Model Structure")
    print(model) 

    training_set_path = "greek_train/greek_train" 
    
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             GreekTransform(),
                                             torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                         ])),
        batch_size=5,
        shuffle=True
    )

    optimizer = optim.SGD(model.fc2.parameters(), lr=0.01, momentum=0.5)
    loss_fn = nn.NLLLoss()
    
    epochs = 100
    train_losses = []

    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(greek_train):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            total += target.size(0)
            
        avg_loss = epoch_loss / len(greek_train)
        train_losses.append(avg_loss)
        accuracy = 100. * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.0f}%")

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, color='blue', marker='o')
    plt.title('Training Loss on Greek Letters')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.xticks(range(1, epochs + 1))
    plt.grid(True)
    plt.show()

    torch.save(model.state_dict(), "greek_model_weights.pth")
    print("Saved Greek Model State to greek_model_weights.pth")

if __name__ == "__main__":
    main(sys.argv)