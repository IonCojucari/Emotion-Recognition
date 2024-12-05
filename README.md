I'll review the uploaded notebook and files to ensure the README reflects their functionalities accurately. Let me examine the content of the `Emotion_Recognition.ipynb` notebook.

Based on the review of the uploaded files, here's the updated **README.md** with more details from the notebook and scripts:

---

# Real-Time Multimodal Emotion Recognition

This project leverages deep learning models for real-time emotion recognition using both image and text modalities. It includes webcam-based emotion detection and supports multimodal integration.

---

## Overview

The project includes:
1. **Real-Time Image Emotion Recognition**: Detect emotions from webcam-captured facial expressions using `real_time_image.py`.
2. **Text Emotion Recognition**: Classify emotions from text data using a trained model.
3. **Multimodal Emotion Recognition**: Combine image and text data for more accurate emotion classification (via the notebook).

---

## Features

### 1. Real-Time Image Emotion Recognition
- **File**: `real_time_image.py`
- Detects faces using MTCNN and classifies emotions into one of seven categories: `Angry`, `Disgust`, `Sad`, `Joy`, `Neutral`, `Scared`, `Surprised`.
- Works out of the box with a webcam.

### 2. Text Emotion Recognition
- **File**: `Emotion_Recognition.ipynb`
- Processes textual data (speech-to-text integration is a future goal) and classifies emotions.
- Supports evaluation of text-only emotion models.

### 3. Multimodal Emotion Recognition
- **File**: `Emotion_Recognition.ipynb`
- Integrates image and text modalities for combined emotion classification.
- Models are trained and evaluated on datasets combining facial expressions and text data.

### 4. Preprocessing (Optional)
- **File**: `data_process.py`
- Scripts to preprocess datasets, extract faces, and standardize images. Not required for real-time usage.

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/emotion-recognition.git](https://github.com/IonCojucari/Emotion-Recognition.git
   cd emotion-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the pre-trained models are in the `Trained Models/` directory.

---

## Running the Project

### Real-Time Image Emotion Recognition
1. Start the webcam-based emotion recognition system:
   ```bash
   python real_time_image.py
   ```
   - Detected emotions are displayed on the webcam feed.
   - Press `q` to exit.

### Multimodal Model Evaluation
1. Open the `Emotion_Recognition.ipynb` notebook.
2. Follow the steps to evaluate image, text, and multimodal emotion models.

---

## Project Files

- **`real_time_image.py`**: Implements webcam-based emotion recognition.
- **`data_process.py`**: Scripts for optional dataset preprocessing.
- **`Emotion_Recognition.ipynb`**: Notebook for training and evaluating text and multimodal models.
- **`Trained Models/`**: Contains pre-trained models.

---

## Notes

- Webcam-based emotion recognition (`real_time_image.py`) works directly with pre-trained models.
- Multimodal models provide enhanced accuracy by integrating text and image data, but require preprocessed datasets and appropriate configuration in the notebook.

---

## Applications

- Real-time sentiment analysis
- Multimodal emotion-based user interaction
- Use in mental health monitoring and adaptive systems

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

Feel free to contribute or raise issues if you have questions or suggestions!
