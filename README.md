# Emotion-Based Music Recommender
This project uses Convolutional Neural Network (CNN) models to detect emotions from facial images and recommend music based on the detected emotions.

## Project Overview

The system detects a user's emotional state from their facial expressions and recommends music accordingly. Two CNN architectures are utilized: a simple CNN and a more advanced ResNet50V2.

## Datasets

### Emotion Detection Dataset
- **Description**: Comprises 35,685 grayscale facial images categorized into seven emotions: happiness, neutral, sadness, anger, surprise, disgust, and fear.
- **Source**: [Kaggle](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data)
- **Structure**: Each image is 48x48 pixels, stored as flattened arrays in the CSV file, and labeled with one of the seven emotions.

### Spotify Music Dataset
- **Description**: Contains metadata for various songs, including features related to mood and emotion.
- **Source**: [Kaggle](https://www.kaggle.com/datasets/musicblogger/spotify-music-data-to-identify-the-moods/data)
- **Structure**: Each entry includes track name, artist, album, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, time signature, and popularity.

## Model Development

### Simple CNN Model
- **Architecture**: 
  - Multiple convolutional layers with ReLU activation to extract features.
  - Max-pooling layers to reduce spatial dimensions.
  - Fully connected layers for classification.
  - Softmax output layer to provide probabilities for each emotion category.
- **Training**: Trained on the emotion detection dataset with data augmentation techniques to improve robustness.

### ResNet50V2 Model
- **Architecture**: 
  - Multiple residual blocks with convolutional layers.
  - Batch normalization to stabilize and accelerate training.
  - Global average pooling followed by fully connected layers for classification.
  - Softmax output layer to provide probabilities for each emotion category.
- **Training**: Trained on the emotion detection dataset with early stopping to prevent overfitting.

## Features

- **Emotion Detection**: Uses CNN models to detect emotions from facial images.
- **Music Recommendation**: Recommends five songs based on the detected emotion using the Spotify dataset.
- **Model Comparison**: Compares the performance of the simple CNN and ResNet50V2 models.
- **Real-time Detection**: Integrates OpenCV for real-time emotion detection from uploaded images.
- **Interactive Layer**: Allows users to upload any image of a person's face and receive song recommendations.

## How to Use

### Prerequisites

- Python 3.x
- Required libraries: pandas, numpy, matplotlib, seaborn, tensorflow, keras, cv2 (OpenCV), requests

