# SecureVision - AI Surveillance System

Welcome to the SecureVision project! This repository contains two crucial files: one for the Python code responsible for accessing the machine learning (ML) model and another for the ML model itself. SecureVision is designed to use your webcam or camera to detect and classify objects for enhanced surveillance.

## Files

### 1. Python Access Code
- **File Name:** `RGB.py`
- **Description:** This Python script serves as the interface for accessing the ML model. It seamlessly integrates with your camera to capture footage, processes it using the ML model, and provides real-time detection and classification results.

### 2. Machine Learning Model
- **File Name:** `rgb_detect.pt`
- **Description:** This file contains the trained ML model, specifically tailored for the SecureVision project. The model has been trained to recognize various objects and classify them accurately. The Python access code utilizes this model for real-time detection during surveillance.

## Usage

1. **Ensure you have all the required dependencies installed.** You can find them listed in the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

2. **Run the `RGB.py` script to initiate the SecureVision AI surveillance system.**

    ```bash
    python RGB.py
    ```

3. **The system will access your webcam or camera, detect objects, and display the results in real-time.**

## Dependencies

- OpenCV
- Ultralytics

## Contribution Guidelines

Feel free to contribute to the SecureVision project by submitting bug reports, feature requests, or pull requests. Please adhere to the established coding conventions and documentation guidelines.

Thank you for choosing SecureVision! If you have any questions or encounter issues, don't hesitate to reach out through the GitHub issues page.
