
# EfficientNet-B0 Plant Disease Classification

This project implements a deep learning pipeline using PyTorch and EfficientNet-B0 for classifying plant diseases from the PlantVillage dataset. It includes automatic dataset splitting, data augmentation, balanced sampling, model training with early stopping, and final evaluation.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Configuration](#configuration)  
6. [Training and Evaluation](#training-and-evaluation)  
7. [Results](#results)  
8. [License](#license)  

---

## Project Overview

The project uses the EfficientNet-B0 architecture pretrained on ImageNet to classify 15 classes of plant diseases. It handles class imbalance with a weighted random sampler and employs various data augmentation techniques to improve generalization. Early stopping based on validation accuracy is implemented to avoid overfitting.

---

## Dataset

- Source: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)  
- The dataset is automatically split into training (70%), validation (15%), and test (15%) sets by the script.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/efficientnet-plant-disease.git
   cd efficientnet-plant-disease
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision tqdm
   ```

---

## Usage

Place your PlantVillage dataset in the folder specified by `SOURCE_DIR` in the script. Then run the main script:

```bash
python train_efficientnet.py
```

The script will:
- Automatically split the dataset if not already split.
- Train the model with data augmentation and balanced sampling.
- Save the best model based on validation accuracy.
- Evaluate the final model on the test set.

---

## Configuration

| Parameter        | Description                                   | Default Value                        |
|------------------|-----------------------------------------------|------------------------------------|
| `SOURCE_DIR`     | Path to original PlantVillage dataset         | `"C:/Users/thoms/OneDrive/Desktop/Capstone/PlantVillage"` |
| `DEST_DIR`       | Path to save split dataset                     | `"C:/Users/thoms/PycharmProjects/efficientnet/lastsplit"` |
| `SPLIT_RATIOS`   | Dataset split ratios (train, val, test)        | `(0.7, 0.15, 0.15)`                |
| `IMAGE_SIZE`     | Size to resize/crop images                      | `300`                              |
| `BATCH_SIZE`     | Number of samples per batch                      | `32`                               |
| `EPOCHS`         | Maximum number of training epochs                | `25`                               |
| `LEARNING_RATE`  | Adam optimizer learning rate                     | `5e-5`                             |
| `PATIENCE`       | Early stopping patience (epochs without improvement) | `3`                                |
| `MODEL_SAVE_PATH`| Filepath to save best model weights             | `"efficientnet_b0_best1.pth"`      |
| `DEVICE`         | Training device (CPU or CUDA GPU)                 | Auto-detected                      |

---

## Training and Evaluation

- Uses data augmentation including random crops, flips, rotations, color jitter, and random erasing during training.
- WeightedRandomSampler is used to address class imbalance.
- CrossEntropyLoss and Adam optimizer.
- Early stopping based on validation accuracy to prevent overfitting.
- After training, the best model is loaded for final evaluation on the test set.

---

## Results

- **Classes Detected:** 15  
- **Training Accuracy:** ~98.95%  
- **Validation Accuracy:** ~99.64% (best)  
- **Final Test Accuracy:** ~68%  

> Note: The test accuracy reflects the final evaluation metric for model performance on unseen data.

---

## License

This project is licensed under the MIT License.

---

*For any questions or contributions, feel free to open an issue or pull request.*
