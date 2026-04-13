"""
Name: 1. Varun Reddy Patlolla
      2. Anand Pinnamaneni

Date: April 7th, 2026

Description: Extension - Demo of live video digit recognition using saved digit images.
             Processes all images from test_images/ folder through the same preprocessing
             pipeline as the live webcam app and displays predictions with confidence.
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
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
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def preprocess_image(image_path):
    """
    Preprocess an image using the same pipeline as the live webcam app:
    1. Read the image
    2. Convert to grayscale
    3. Gaussian blur
    4. Otsu's thresholding (auto-inverts to white-on-black)
    5. Resize to 28x28
    6. Normalize (mean=0.5, std=0.5)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [ERROR] Could not read: {image_path}")
        return None, None, None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's thresholding (automatically inverts: white digit on black bg)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize to 28x28
    resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)

    # Convert to tensor and normalize
    tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5

    return tensor, resized, img


def main(argv):
    # Load model
    model = Net()
    model.load_state_dict(torch.load("mnist_model_weights.pth", map_location="cpu"))
    model.eval()
    print("Model loaded successfully!\n")

    # Get all digit images
    image_dir = "test_images"
    digit_names = ["zero", "one", "two", "three", "four",
                   "five", "six", "seven", "eight", "nine"]

    results = []

    for i, name in enumerate(digit_names):
        path = os.path.join(image_dir, f"{name}.jpeg")
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue

        tensor, processed, original = preprocess_image(path)
        if tensor is None:
            continue

        # Run inference
        with torch.no_grad():
            output = model(tensor)
            probs = torch.exp(output)
            prediction = output.argmax(1).item()
            confidence = probs[0][prediction].item()

        correct = "✓" if prediction == i else "✗"
        print(f"  Digit {i} ({name}.jpeg): Predicted={prediction} "
              f"Confidence={confidence:.1%} {correct}")

        results.append({
            "true_label": i,
            "name": name,
            "prediction": prediction,
            "confidence": confidence,
            "probs": probs[0].numpy(),
            "processed": processed,
            "original": cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        })

    # Create a visualization grid
    fig, axes = plt.subplots(3, 10, figsize=(20, 7))
    fig.suptitle("Live Video Extension Demo - Digit Recognition Results",
                 fontsize=16, fontweight="bold")

    for idx, r in enumerate(results):
        # Row 1: Original image
        axes[0, idx].imshow(r["original"])
        color = "green" if r["prediction"] == r["true_label"] else "red"
        axes[0, idx].set_title(f"True: {r['true_label']}", fontsize=10)
        axes[0, idx].set_xticks([])
        axes[0, idx].set_yticks([])

        # Row 2: Preprocessed 28x28 input
        axes[1, idx].imshow(r["processed"], cmap="gray")
        axes[1, idx].set_title(f"Pred: {r['prediction']} ({r['confidence']:.0%})",
                               fontsize=9, color=color, fontweight="bold")
        axes[1, idx].set_xticks([])
        axes[1, idx].set_yticks([])

        # Row 3: Probability bar chart
        colors = ["green" if j == r["prediction"] else "gray" for j in range(10)]
        axes[2, idx].bar(range(10), r["probs"], color=colors, width=0.8)
        axes[2, idx].set_ylim(0, 1)
        axes[2, idx].set_xticks(range(10))
        axes[2, idx].set_xticklabels(range(10), fontsize=6)
        if idx == 0:
            axes[2, idx].set_ylabel("Probability", fontsize=8)
        else:
            axes[2, idx].set_yticks([])

    # Label rows
    axes[0, 0].set_ylabel("Original", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("28×28 Input", fontsize=10, fontweight="bold")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/live_video_demo.png", bbox_inches="tight", dpi=300)
    print(f"\nSaved: results/live_video_demo.png")
    plt.close()

    # Print summary
    correct_count = sum(1 for r in results if r["prediction"] == r["true_label"])
    print(f"\nAccuracy on custom digits: {correct_count}/{len(results)} "
          f"({100 * correct_count / len(results):.0f}%)")


if __name__ == "__main__":
    main(sys.argv)
