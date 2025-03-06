# CAPTCHA Attack and Recognition Project

This project demonstrates a complete workflow for building, training, and deploying a fixed-length CAPTCHA recognition model using an attention-based encoder-decoder architecture in PyTorch. The system is designed for academic research and includes:

- **Data Preparation and Caching:**  
  Efficiently loads a large dataset of small CAPTCHA images, applies optional preprocessing, and uses data augmentation (each image is augmented to generate 5 variants) to increase diversity.

- **Model Architecture:**  
  Uses a CNN backbone to extract features from CAPTCHA images, a bidirectional GRU to model sequential information, and an attention decoder that uses learnable queries to generate fixed-length outputs (5 characters). A classifier then maps these outputs to character probabilities via a log softmax.

- **Training Pipeline:**  
  The model is trained using cross-entropy loss computed independently for each character position. Early stopping and checkpointing are used based on validation loss to avoid overfitting and ensure the best model is saved.

- **Inference Module:**  
  Provides functions for single-image, batch, and real-time inference (via HTTP requests) with beam search decoding for robust prediction.

> **Important:** This project is intended strictly for academic research purposes. Do not use this code against any systems without proper authorization.

## Project Structure

```plaintext
CAPTCHA-ATTACK/
├── models.py                    # Model definitions: AttentionOCR, AttentionDecoder, etc.
├── captcha_generator.py         # Generate captcha images for test
├── DL_CRNN_with_decoder_aug.py  # Latest training script
├── real_time_inference.py       # Inference script for single/batch/real-time prediction using HTTP
└── requirements.txt             # Python package requirements
```
## Installation
1. Clone the repository:

```bash
git clone https://github.com/yourusername/CAPTCHA-ATTACK.git
cd CAPTCHA-ATTACK
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```
## Data Preparation

Pull the dataset from Kaggle by using
```python
import kagglehub

# 下载最新版本数据集（注意：下载目录结构可能为：path/images/*.png）
path = kagglehub.dataset_download("parsasam/captcha-dataset")
print("Path to dataset files:", path)
```

## Modeling

For the pre-train model, I used (CRNN.PyTorch by @meijieru)[https://github.com/meijieru/crnn.pytorch]
