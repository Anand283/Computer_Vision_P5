"""
Name: 1. Varun Reddy Patlolla
      2. Anand Pinnamaneni

Date: April 8th, 2026

Description: This file is to verify the greek letters alpha,beta and gamma function,
             by giving it data which is outside of test data.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

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

class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

def analyze_greek_image(image_path):
    model = Net()
    
    model.fc2 = nn.Linear(50, 3)
    
    try:
        model.load_state_dict(torch.load("greek_model_weights.pth", weights_only=True))
    except FileNotFoundError:
        print("Error: Could not find 'greek_model_weights.pth'. Did you save it in Task 3?")
        return

    model.eval()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        GreekTransform(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    input_tensor = transform(img)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
        
        prediction_idx = output.argmax(dim=1, keepdim=True).item()
        
        probabilities = torch.exp(output)[0]

   
    greek_classes = ["Alpha", "Beta", "Gamma"]
    final_answer = greek_classes[prediction_idx]

    print(f"FINAL PREDICTION: The symbol is: {final_answer}!")
    
    for idx, name in enumerate(greek_classes):
        print(f"{name:>6}: {probabilities[idx].item() * 100:>6.2f}%")

if __name__ == "__main__":
    target_image = "folder_handwritten_ABG/gamma3.png"
    analyze_greek_image(target_image)