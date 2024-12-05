# Emotion-Recognition

This project focuses on multimodal emotion recognition in videos, leveraging transformer-based models and attention mechanisms to identify emotions from facial expressions.

## Overview

The dataset includes video clips that are preprocessed to extract facial features. Using a transformer-based deep learning approach, this project aims to classify emotions from the extracted data.

## Features

1. **Data Preprocessing**: 
   - Extracts the middle frame from each video in the dataset.
   - Detects and extracts the largest face from the middle frame using MTCNN (Multi-task Cascaded Convolutional Networks).
   - Resizes the extracted face to a standard dimension for uniformity.

2. **Database**:
   - Preprocessed images and faces are already provided in the GitHub repository, so it is **not necessary to compile the `data_process.py` file** unless reprocessing is required.

3. **Deep Learning Model**:
   - Implements a transformer-based architecture with attention mechanisms to analyze and classify the emotion of detected faces.

## Files

- `data_process.py`: Script to preprocess videos by extracting middle frames and faces. It uses:
  - `process_videos`: Extracts middle frames from video files.
  - `process_images_to_faces`: Extracts faces from the middle frame.
- `Emotion_recognition.ipynb`: Jupyter Notebook implementing the deep learning model for emotion recognition.

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/emotion-recognition.git
   cd emotion-recognition
   ```

2. (Optional) Run data preprocessing if needed:
   ```bash
   python data_process.py
   ```

3. Train the model:
   Open `Emotion_recognition.ipynb` and follow the steps to train and evaluate the model.

## Notes

- The preprocessed database of faces is included in the repository under the `Data/` folder.
- Use the `data_process.py` script only if you want to preprocess a new dataset or modify the current preprocessing steps.

## Results

The model achieves accurate classification of emotions by utilizing state-of-the-art transformer-based architectures, making it suitable for applications like sentiment analysis, human-computer interaction, and more.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

Feel free to contribute or raise issues if you have questions or suggestions!

