# Advanced Out-of-Distribution Detection for Multi-Class Classification

## Description

This project explores and implements advanced methods for Out-of-Distribution (OOD) detection in the context of a multi-class image classification task. The primary goal is to build a reliable system that can not only classify in-distribution (ID) images accurately but also identify when it is presented with an unknown, OOD input.

The core problem, known as the "OOD Challenge," occurs when a machine learning model, trained on a specific data distribution (e.g., food images), encounters an input from a different distribution (e.g., images of digits or textures). Standard models often produce highly confident but completely incorrect predictions for these OOD inputs. This project tackles this issue by training a model on the **Food-101** dataset and evaluating its ability to detect OOD samples from a diverse set of 8 other datasets.

We implement and compare a baseline model against a model enhanced with **Energy-based fine-tuning**. We evaluate both models using two distinct post-hoc scoring functions: the **Energy score** and the **CORES score**.

## Key Features

- A from-scratch PyTorch implementation of the **MobileNetV3** architecture, chosen for its excellent balance of performance and efficiency.
- **Energy-based OOD detection**, implements the energy score, a robust method for OOD detection derived from model logits, which avoids the overconfidence issues of traditional softmax scores.
- **CORES score implementation**, implements the CORES (Convolutional Response-based Score) method, which leverages internal kernel response patterns for OOD detection without requiring fine-tuning.
- **Energy fine-tuning** leverages auxiliary OOD data to explicitly shape the model's "energy surface," improving the separation between ID and OOD samples.
- **Comprehensive evaluation**. The models are benchmarked against 8 different OOD datasets, categorized as "near-OOD" (e.g., Flowers102, DTD) and "far-OOD" (e.g., SVHN, FashionMNIST), using standard metrics like **AUROC**, **AUPR**, and **FPR@95TPR**.
- **Hyperparameter optimization** utilizes **Optuna** to systematically search for the optimal hyperparameters (learning rate, energy margins) for the fine-tuning process.

## Libraries Used

  - [PyTorch](https://pytorch.org/) & [TorchVision](https://pytorch.org/vision/stable/index.html): For model creation, training, and data handling.
  - [Scikit-learn](https://scikit-learn.org/stable/): For calculating evaluation metrics (AUROC, AUPR, etc.).
  - [Pandas](https://pandas.pydata.org/): For organizing and displaying results.
  - [NumPy](https://numpy.org/): For numerical operations.
  - [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/): For data visualization and plotting score distributions.
  - [Optuna](https://optuna.org/): For hyperparameter optimization.
  - [Tqdm](https://github.com/tqdm/tqdm): For progress bars during training and evaluation.


## Project Structure

├── data/ # Directory where datasets will be automatically downloaded

├── models/ # Directory where trained model weights are saved

├── plots/ # Directory where evaluation plots are saved

├── ood_detection_vision.ipynb # Main Jupyter notebook with all code

└── README.md # This file



## Usage

All the code for training, fine-tuning, and evaluation is contained within the `ood_detection_vision.ipynb` Jupyter/Colab notebook. The cells are organized sequentially to guide you through the entire workflow.

1.  **Setup and data loading**: Running the initial cells will set up the configuration and automatically download the **Food-101** dataset and the required OOD datasets from `torchvision.datasets`.

2.  **Baseline model training**:
    - Execute the "Training" cells to train the standard MobileNetV3 model using cross-entropy loss.
    - The best model weights will be saved to `models/mobilenetv3_standard.pth`. If this file already exists, the training step will be skipped, and the saved weights will be loaded.

3.  **Energy-based fine-tuning**:
    - Run the "Energy-based fine-tuning" cells. This process loads the baseline model's weights and fine-tunes the network head using a combined cross-entropy and energy-based loss objective.
    - The best fine-tuned model is saved to `models/mobilenetv3_energy.pth`.

4.  **Evaluation**:
    - The final "Results" cells are used for comprehensive evaluation.
    - You can evaluate either the baseline or the fine-tuned model by uncommenting the desired model path.
    - The code will generate:
      - A summary table in the console with the model's ID accuracy and OOD detection performance (AUROC, AUPR, FPR@95TPR) across all 8 OOD datasets.
      - Distribution plots for each OOD dataset, saved as `.png` files in the `plots/` directory.

## Results & Findings

Our experiments show that energy-based fine-tuning is a highly effective technique for improving both in-distribution classification and out-of-distribution detection.

- **Improved classification accuracy**: Energy-based fine-tuning significantly boosted the model's core classification accuracy on the Food-101 test set, improving it from **67.56%** (baseline) to **76.00%**.

- **Enhanced OOD Detection**: The fine-tuned model demonstrated a consistent and significant reduction in the false positive rate (FPR@95TPR) across all evaluated OOD datasets, especially for challenging, near-distribution datasets.

| OOD Dataset   | Baseline FPR@95TPR | Fine-Tuned FPR@95TPR | Improvement |
|---------------|--------------------|----------------------|-------------|
| DTD (Textures)| 25.32%             | 18.40%               | **-6.92%**  |
| Flowers 102   | 50.97%             | 35.92%               | **-15.05%** |
| OxfordIIITPet | 40.64%             | 34.29%               | **-6.35%**  |

- **CORES vs. Energy score**:
  - The **CORES score** is a powerful alternative that excels "out-of-the-box" on the baseline model, especially on texture-based datasets like DTD, achieving a low **7.39%** FPR@95TPR without any fine-tuning.
  - However, **energy-based fine-tuning** is crucial for improving performance on datasets that are visually similar to the in-distribution data (like Flowers102 and OxfordIIITPet), where CORES struggles more.

In conclusion, a systematic approach combining a solid backbone like MobileNetV3, targeted fine-tuning, and robust scoring functions is key to building reliable models that know what they don't know.

## Acknowledgments

- This project was developed by **Federico Tranzocchi** and **Gianmarco Corsi**.
- The methods implemented are based on the foundational research in the following papers:
  - Liu, W., et al. (2020). *Energy-based Out-of-distribution Detection.*
  - Tang, K., et al. (2024). *CORES: Convolutional Response-based Score for OOD Detection.*
  - Sharifi, S., et al. (2024). *Gradient-Regularized Out-of-Distribution Detection.*
