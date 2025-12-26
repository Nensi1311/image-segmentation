# Image Segmentation with U²-Net

A Python-based image segmentation tool using the U²-Net (U Square Net) deep learning model for human segmentation. This project processes images and generates binary masks highlighting the segmented subjects.

## Features

- **Human Segmentation**: Automatically segments humans from images using pre-trained U²-Net model
- **Batch Processing**: Process multiple images in a folder at once
- **Configurable Threshold**: Adjustable segmentation threshold for fine-tuning results
- **GPU Support**: Automatic GPU detection for faster processing (CUDA)
- **Organized Output**: Automatically saves masks to a dedicated output folder

## Requirements

- Python 3.7+
- PyTorch (CPU or CUDA-enabled)
- OpenCV (cv2)
- Pillow (PIL)
- NumPy

## Installation

1. **Clone or download this repository**

2. **Install required dependencies**:
```bash
pip install torch torchvision opencv-python pillow numpy
```

3. **Download the pre-trained model weights**:
   - Download the weights file from: [Google Drive](https://drive.google.com/file/d/1Wk21u8Jce0gS4irTPclDkq_Lln8dGxRw/view?usp=sharing)
   - Place the downloaded `u2net_human_seg.pth` file in the `weights/` directory

## Project Structure

```
image-segmentation/
├── images/              # Input images directory
│   ├── 37.jpg
│   ├── 40.jpg
│   └── 42.jpg
├── mask/                # Output masks directory (auto-created)
│   ├── 37.jpg
│   ├── 40.jpg
│   └── 42.jpg
├── model/
│   └── u2net.py         # U²-Net model architecture
├── weights/
│   ├── u2net_human_seg.pth  # Pre-trained model weights (download required)
│   └── weights.txt      # Download link for weights
├── run.py               # Main execution script
└── README.md
```

## Usage

1. **Place your images** in the `images/` folder (supports JPG, PNG, JPEG, BMP, WEBP formats)

2. **Run the segmentation script**:
```bash
python run.py
```

3. **Find the generated masks** in the `mask/` folder

## Configuration

You can modify the following parameters in `run.py`:

- `INPUT_DIR`: Input images directory (default: `"images"`)
- `OUTPUT_DIR`: Output masks directory (default: `"mask"`)
- `MODEL_PATH`: Path to the model weights file (default: `"weights/u2net_human_seg.pth"`)
- `IMAGE_SIZE`: Input image size for the model (default: `320`)
- `THRESHOLD`: Segmentation threshold (default: `0.80`)
  - Lower values (e.g., 0.5): More lenient segmentation, includes more pixels
  - Higher values (e.g., 0.9): Stricter segmentation, includes fewer pixels

## Model Architecture

This project uses the **U²-Net** (U Square Net) architecture, which features:
- Nested U-structure encoder-decoder architecture
- Residual U-blocks (RSU) for feature extraction
- Multi-scale feature fusion
- Side outputs for deep supervision

The model is specifically trained for human segmentation tasks.

## References

- U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection
- Original U²-Net repository: [https://github.com/xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)
