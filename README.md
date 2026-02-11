# Captcha Slayer

A custom CNN-based CAPTCHA solver for image-based CAPTCHAs, built from scratch using Keras.

## Overview

This project implements a convolutional neural network to automatically solve image-based CAPTCHAs. I designed and trained a simple custom CNN architecture.

## Technical Approach

**Model Architecture:**
- Custom CNN designed specifically for CAPTCHA character recognition
- Built using Keras/TensorFlow
- Trained on synthetically generated CAPTCHA dataset (captcha library)
- [View the full data table here](/CaptchaSlayer-base.csv)
- [View the google sheets here (calculations)](https://docs.google.com/spreadsheets/d/e/2PACX-1vTp6OgSoT5ML6a_sfI2lCpMmkueIl8r5ZKvutVT_T6XzKxZeOjPm004m7fAQVqYKJ8xGRBJ1bPmZeWM/pubhtml)

**Data Pipeline:**
- CAPTCHA generation for creating training data
- Image preprocessing (grayscale conversion) to improve model performance

**Key Features:**
- Basic end-to-end solution from data generation to trained model
- Benchmarking results available in `/results` folder
- Simple, lightweight architecture focused on understanding functionality

## Tech Stack

- **Framework:** Keras/TensorFlow
- **Image Processing:** OpenCV (grayscale preprocessing)
- **Dataset Generation:** [captcha](https://pypi.org/project/captcha/)

## Project Structure
```
captcha-slayer/
├── results/          # Benchmark images and performance metrics
├── [model files]     # Trained model weights
├── [training code]   # Dataset generation and training pipeline
└── README.md
```

## Results

Performance benchmarks and accuracy metrics can be found in the `/results` folder, including visual demonstrations of the model's predictions.

## What I Learned

- Designing CNN architectures from documentation
- Building basic data pipelines for ML training
- Simple image preprocessing techniques for improving model accuracy

## Future Improvements

- Expand support for different CAPTCHA types
- Optimize model architecture for better accuracy
- Clean up dependencies and document them

## Installation

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the environment
# On Windows: .\venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt
```

## Results

<details>
<summary>Click to view performance metrics and examples</summary>

![Benchmark Results](Results/results.png)
![Training_Graph](Results/training_graph.png)
---

*Note: This project is for educational purposes only.*
