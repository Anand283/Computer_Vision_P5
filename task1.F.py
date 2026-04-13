import torch
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

# 1. Define the exact same Net class
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
    # 2. Setup the exact same transformations used in training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Ensure it's 1-channel
        transforms.Resize((28, 28)),                # Match MNIST size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))         # Match MNIST intensity scaling
    ])

    # 3. Load the trained model
    model = Net()
    model.load_state_dict(torch.load("mnist_model_weights.pth"))
    model.eval()

    image_folder = "test_images"
    image_files = sorted(os.listdir(image_folder)) # Sorts them for the grid

    plt.figure(figsize=(12, 5))
    
    print(f"{'Filename':<15} | {'Prediction':<10}")
    print("-" * 30)

    for i, filename in enumerate(image_files):
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # 4. Load and Preprocess the image
        img_path = os.path.join(image_folder, filename)
        raw_img = Image.open(img_path)
        img_t = transform(raw_img)
        
        # Add batch dimension [1, 1, 28, 28]
        img_batch = img_t.unsqueeze(0)

        # 5. Inference
        with torch.no_grad():
            output = model(img_batch)
            prediction = output.data.max(1, keepdim=True)[1].item()

        print(f"{filename:<15} | {prediction:<10}")

        # 6. Plotting
        plt.subplot(2, 5, i + 1)
        plt.tight_layout()
        # Squeeze out the batch/channel dims for plotting
        plt.imshow(img_t.squeeze(), cmap='gray', interpolation='none')
        plt.title(f"Pred: {prediction}")
        plt.xticks([])
        plt.yticks([])

    # Save the custom results
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/custom_digit_test.png", bbox_inches="tight", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()