# Depth Estimation from a Single RGB Image using PyTorch

This project demonstrates a deep learning-based approach to estimate scene depth from a single RGB image using a U-Net-like architecture with a DenseNet169 encoder. The model is trained and evaluated using the NYU Depth V2 dataset. This work serves as a complete pipeline—from data preprocessing and custom dataset loading to model training, visualization, and evaluation—making it ideal for academic learning and practical use in computer vision tasks.

---

## Project Overview

Depth estimation is a fundamental problem in computer vision with applications in autonomous driving, robotics, augmented reality, and 3D scene reconstruction. This project addresses the monocular depth estimation problem using a convolutional neural network (CNN) based encoder-decoder architecture.

**Key Features:**
- Encoder: DenseNet169 backbone (pretrained)
- Decoder: Custom U-Net-style upsampling blocks with skip connections
- Mixed Precision Training using PyTorch AMP
- Custom loss function with RMSE and Absolute Relative Error
- Training and inference on the NYU Depth V2 dataset
- Visualization of predictions vs ground truth
- GPU support with automatic CUDA/CPU selection

---

## Model Architecture
The model consists of:

Encoder: Pretrained DenseNet169 to extract deep image features

Decoder: Upsampling layers with skip connections to reconstruct the depth map

Output: A single-channel depth image with the same resolution as the input image

Mixed-precision training is used via torch.cuda.amp for memory efficiency and speedup.

## Evaluation Metrics
Root Mean Square Error (RMSE)

Absolute Relative Error (AbsRel)

δ Accuracy Thresholds (δ < 1.25, 1.25², 1.25³)

These metrics are computed on the validation set to assess prediction quality.

## Achievements
Full pipeline implementation for monocular depth estimation using CNNs

DenseNet encoder used for rich feature extraction

Mixed precision training with PyTorch AMP for improved performance

Visualizations and evaluation incorporated directly into the notebook

Demonstrated reliable predictions even on a small subset of NYU Depth V2

## Future Work
Train on full-resolution images and full dataset

Try different backbones (ResNet, EfficientNet, etc.)

Add edge-preserving post-processing

Port the solution to a script-based or web app interface
