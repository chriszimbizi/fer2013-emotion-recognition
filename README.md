# Facial Emotion Recognition using Convolutional Neural Networks

## Project Overview

This project implements a deep learning model for facial emotion recognition using the FER2013 dataset, demonstrating advanced techniques in computer vision and machine learning.

## Project Objectives

- Develop a robust Convolutional Neural Network (CNN) for classifying facial expressions
- Explore data augmentation techniques to improve model generalization
- Implement advanced deep learning strategies for image classification

## Dataset Details

### FER2013 Dataset

- Source: [Kaggle FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- Image Specifications:
  - 48x48 pixel grayscale images
  - Faces automatically registered and centered

### Emotion Categories

- 0: Angry
- 1: Fear
- 2: Happy
- 3: Neutral
- 4: Sad
- 5: Surprise

## Methodology

### 1. Data Preprocessing and Augmentation

#### Why Data Augmentation?

- **Reduce Overfitting**: By introducing variations in training data
- **Increase Training Examples**: Each transformation adds new learning samples
- **Improve Model Robustness**: Helps model generalize to real-world variations

Augmentation Techniques:

- Rotation (±20 degrees)
- Width/Height Shifts (20%)
- Horizontal Flipping
- Zoom Range (20%)
- Rescaling to normalize pixel values

### 2. CNN Architecture Design

#### Network Structure Rationale

- **Convolutional Blocks**: Progressively extract spatial hierarchical features
  - Initial blocks: Extract basic features
  - Intermediate blocks: Capture complex patterns
  - Advanced blocks: Learn high-level representations

Key Architectural Choices:

- Batch Normalization: Stabilize learning
- Spatial Dropout: Prevent feature co-adaptation
- Multiple Dense Layers: Combine and reduce dimensionality

### 3. Training Strategy

- **Optimizer**: Adam (adaptive learning rate)
- **Loss Function**: Categorical Cross-Entropy
- **Early Stopping**: Prevent overfitting by monitoring validation loss

## Performance Metrics

### Model Performance

- **Test Accuracy**: 65%
- Per-Class Performance Highlights:
  - **Highest Accuracy**: Happy (86%)
  - **Challenging Classes**: Fear, Sad

## Installation and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/facial-emotion-recognition.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
jupyter notebook src/main.ipynb
```
