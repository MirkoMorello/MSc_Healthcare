# EchoGuardian: AI-Powered Breast Cancer Segmentation and Classification

> **Course:** Signal and Imaging Acquisition and Modelling in Healthcare

## Overview

EchoGuardian is an AI-powered diagnostic tool designed to assist radiologists in the early detection of breast cancer. This project tackles two critical tasks: the automatic segmentation of lesions from ultrasound images and their subsequent classification as benign or malignant. The primary goal is to create a reliable pipeline that improves diagnostic accuracy and efficiency, with a strong focus on achieving high sensitivity for malignant cases.

This project was developed in collaboration with [@andypalmi](https://github.com/andypalmi) and [@andreaborghesi00](https://github.com/andreaborghesi00)

## Core Pipeline

The project follows a two-stage analysis pipeline:

### 1. Lesion Segmentation
The first stage focuses on accurately identifying and outlining the boundaries of breast lesions in ultrasound images.

*   **Models:** We implemented and evaluated state-of-the-art semantic segmentation architectures, including:
    *   **DeepLabV3+** (with ResNet34, ResNet50, and Xception65 backbones)
    *   **UNet++** (with a ResNet34 backbone)
*   **Data & Augmentation:** The models were trained on a dataset of 647 ultrasound images with corresponding pixel-wise masks. To improve model robustness and generalization, we applied a series of data augmentations during training.

### 2. Lesion Classification
Once a lesion is segmented, the second stage classifies it as benign or malignant.

*   **Feature Extraction:** We extracted a comprehensive set of **101 radiomic features** from each segmented lesion mask. These features capture quantitative information about the lesion's shape, texture, and intensity.
*   **Classifiers:** The extracted features were used to train and evaluate three different machine learning models:
    *   Support Vector Machine (SVM)
    *   Random Forest
    *   Feed-Forward Neural Network (FFN)
*   **Evaluation:** Performance was measured using F1-score, sensitivity, specificity, and accuracy, with a primary objective of maximizing sensitivity (recall) for malignant lesions.

## Technologies Used

*   **Deep Learning:** Python, PyTorch, MONAI
*   **Machine Learning:** Scikit-learn
*   **Data Handling:** Pandas, NumPy, OpenCV
