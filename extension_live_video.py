"""
Name: 1. Varun Reddy Patlolla
      2. Anand Pinnamaneni

Date: April 7th, 2026

Description: Extension - Live video digit recognition using the trained MNIST CNN.
             Supports two modes:
             1. Live webcam mode (default): real-time digit classification
             2. Image mode (--images <folder>): process saved images with same UI

             Usage:
               python3 extension_live_video.py                  # webcam mode
               python3 extension_live_video.py --images test_images  # image mode
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import glob


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


def preprocess_roi(roi):
    """
    Preprocess a region of interest (ROI) to match MNIST format.
    """
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)

    tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5

    return tensor, resized


def draw_prediction_bar(frame, output_probs, prediction, x_start, y_start):
    """Draw a bar chart of class probabilities on the frame."""
    bar_width = 20
    max_height = 100

    for i in range(10):
        prob = output_probs[i]
        bar_height = int(prob * max_height)

        x = x_start + i * (bar_width + 5)
        y = y_start

        color = (0, 255, 0) if i == prediction else (150, 150, 150)
        cv2.rectangle(frame, (x, y - bar_height), (x + bar_width, y), color, -1)
        cv2.putText(frame, str(i), (x + 3, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_bounding_box(frame, roi_x, roi_y, roi_size, prediction, confidence):
    """
    Draw a prominent bounding box around the ROI with prediction label.
    Color changes based on confidence: green (high), yellow (medium), red (low).
    """
    if confidence > 0.7:
        color = (0, 255, 0)       # Green
    elif confidence > 0.4:
        color = (0, 255, 255)     # Yellow
    else:
        color = (0, 0, 255)       # Red

    # Draw thick bounding box
    cv2.rectangle(frame, (roi_x, roi_y),
                  (roi_x + roi_size, roi_y + roi_size), color, 3)

    # Corner accents
    corner_len = 25
    thickness = 4
    corners = [
        # Top-left
        ((roi_x, roi_y), (roi_x + corner_len, roi_y)),
        ((roi_x, roi_y), (roi_x, roi_y + corner_len)),
        # Top-right
        ((roi_x + roi_size, roi_y), (roi_x + roi_size - corner_len, roi_y)),
        ((roi_x + roi_size, roi_y), (roi_x + roi_size, roi_y + corner_len)),
        # Bottom-left
        ((roi_x, roi_y + roi_size), (roi_x + corner_len, roi_y + roi_size)),
        ((roi_x, roi_y + roi_size), (roi_x, roi_y + roi_size - corner_len)),
        # Bottom-right
        ((roi_x + roi_size, roi_y + roi_size), (roi_x + roi_size - corner_len, roi_y + roi_size)),
        ((roi_x + roi_size, roi_y + roi_size), (roi_x + roi_size, roi_y + roi_size - corner_len)),
    ]
    for pt1, pt2 in corners:
        cv2.line(frame, pt1, pt2, color, thickness)

    # Label with background
    label = f"Digit: {prediction}  ({confidence:.0%})"
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    label_y = roi_y - 15
    cv2.rectangle(frame, (roi_x, label_y - text_h - 10),
                  (roi_x + text_w + 10, label_y + 5), color, -1)
    cv2.putText(frame, label, (roi_x + 5, label_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)


def process_single_image(model, image_path, output_dir):
    """Process a single image with the same UI as live video and save the result."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [ERROR] Could not read: {image_path}")
        return None

    # Create a canvas (simulate webcam frame)
    canvas_h, canvas_w = 480, 640
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Resize image to fit the ROI area
    roi_size = 200
    roi_x = (canvas_w - roi_size) // 2
    roi_y = (canvas_h - roi_size) // 2

    # Place the image in the center ROI
    img_resized = cv2.resize(img, (roi_size, roi_size))
    canvas[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size] = img_resized

    # Preprocess the ROI
    roi = canvas[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
    tensor, processed = preprocess_roi(roi)

    # Run inference
    with torch.no_grad():
        output = model(tensor)
        probs = torch.exp(output)
        prediction = output.argmax(1).item()
        confidence = probs[0][prediction].item()

    # Draw bounding box with prediction
    draw_bounding_box(canvas, roi_x, roi_y, roi_size, prediction, confidence)

    # Draw probability bars
    draw_prediction_bar(canvas, probs[0].numpy(), prediction,
                        roi_x + roi_size + 20, roi_y + roi_size)

    # Show preprocessed 28x28 image
    processed_display = cv2.resize(processed, (100, 100),
                                   interpolation=cv2.INTER_NEAREST)
    canvas[10:110, canvas_w - 110:canvas_w - 10] = \
        cv2.cvtColor(processed_display, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canvas, (canvas_w - 112, 8), (canvas_w - 8, 112), (255, 255, 255), 1)
    cv2.putText(canvas, "28x28 input", (canvas_w - 120, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Title
    cv2.putText(canvas, "Live Digit Recognition", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Confidence legend
    cv2.putText(canvas, "High >70%", (10, canvas_h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(canvas, "Med 40-70%", (10, canvas_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(canvas, "Low <40%", (10, canvas_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Save result
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"live_{basename}.png")
    cv2.imwrite(output_path, canvas)

    return prediction, confidence, output_path


def run_image_mode(model, image_folder, output_dir):
    """Process all images from a folder with the live video UI."""
    print(f"\n=== Image Mode: Processing images from {image_folder}/ ===\n")

    # Find all images
    extensions = ["*.jpeg", "*.jpg", "*.png", "*.bmp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
    image_paths.sort()

    if not image_paths:
        print(f"No images found in {image_folder}/")
        return

    print(f"Found {len(image_paths)} images\n")

    digit_names = ["zero", "one", "two", "three", "four",
                   "five", "six", "seven", "eight", "nine"]

    results = []
    for path in image_paths:
        result = process_single_image(model, path, output_dir)
        if result:
            pred, conf, out_path = result
            basename = os.path.basename(path)
            # Try to determine true label from filename
            true_label = "?"
            for i, name in enumerate(digit_names):
                if name in basename.lower():
                    true_label = str(i)
                    break
            correct = "✓" if str(pred) == true_label else "✗"
            print(f"  {basename:15s} -> Predicted: {pred}  Confidence: {conf:.1%}  "
                  f"(True: {true_label}) {correct}  Saved: {out_path}")
            results.append((path, pred, conf, out_path))

    # Create a combined grid of all results
    if results:
        n = len(results)
        cols = min(5, n)
        rows = (n + cols - 1) // cols
        cell_w, cell_h = 640, 480
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for idx, (_, _, _, out_path) in enumerate(results):
            r, c = idx // cols, idx % cols
            cell_img = cv2.imread(out_path)
            if cell_img is not None:
                grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell_img

        grid_path = os.path.join(output_dir, "live_video_grid.png")
        cv2.imwrite(grid_path, grid)
        print(f"\nSaved combined grid: {grid_path}")

    print(f"\nAll results saved to {output_dir}/")


def run_webcam_mode(model, output_dir):
    """Run real-time digit recognition from webcam."""
    # Try multiple video device indices to find available webcam
    cap = None
    for idx in range(5):
        test_cap = cv2.VideoCapture(idx)
        if test_cap.isOpened():
            ret, _ = test_cap.read()
            if ret:
                cap = test_cap
                print(f"Opened webcam at index {idx}")
                break
            test_cap.release()
        else:
            test_cap.release()

    if cap is None:
        print("Error: Could not open webcam on any device index (0-4)")
        print("Try: python3 extension_live_video.py --images test_images")
        return

    print("\n=== Live Digit Recognition ===")
    print("Instructions:")
    print("  - Hold a handwritten digit in front of the camera")
    print("  - Position it within the bounding box")
    print(f"  - Press 's' to save a screenshot to {output_dir}/")
    print("  - Press 'q' to quit")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_size = 200
    roi_x = (frame_width - roi_size) // 2
    roi_y = (frame_height - roi_size) // 2

    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
        tensor, processed = preprocess_roi(roi)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.exp(output)
            prediction = output.argmax(1).item()
            confidence = probs[0][prediction].item()

        draw_bounding_box(frame, roi_x, roi_y, roi_size, prediction, confidence)

        draw_prediction_bar(frame, probs[0].numpy(), prediction,
                            roi_x + roi_size + 20, roi_y + roi_size)

        processed_display = cv2.resize(processed, (100, 100),
                                       interpolation=cv2.INTER_NEAREST)
        frame[10:110, frame_width - 110:frame_width - 10] = \
            cv2.cvtColor(processed_display, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(frame, (frame_width - 112, 8),
                      (frame_width - 8, 112), (255, 255, 255), 1)
        cv2.putText(frame, "28x28 input", (frame_width - 120, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(frame, "Live Digit Recognition", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, "High >70%", (10, frame_height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, "Med 40-70%", (10, frame_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame, "Low <40%", (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imshow("Live Digit Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_count += 1
            filename = os.path.join(output_dir, f"live_digit_{screenshot_count}.png")
            cv2.imwrite(filename, frame)
            print(f"Saved screenshot: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! {screenshot_count} screenshots saved to {output_dir}/")


def main(argv):
    output_dir = "results/live_video"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = Net()
    model.load_state_dict(torch.load("mnist_model_weights.pth", map_location="cpu"))
    model.eval()
    print("Model loaded successfully!")

    # Check for --images flag
    if "--images" in argv:
        idx = argv.index("--images")
        if idx + 1 < len(argv):
            image_folder = argv[idx + 1]
        else:
            image_folder = "test_images"
        run_image_mode(model, image_folder, output_dir)
    else:
        run_webcam_mode(model, output_dir)


if __name__ == "__main__":
    main(sys.argv)
