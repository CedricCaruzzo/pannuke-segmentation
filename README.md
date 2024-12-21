# UNet for Histopathology Image Segmentation


![app demonstration](https://github.com/user-attachments/assets/2e430457-0c06-461a-ae21-e436d1c9a887)

This repository contains a project focused on segmentation in histopathology using a modified UNet architecture. The primary dataset used is the PanNuke dataset. The project includes both a training pipeline and a web application for easy exploration and inference on the dataset.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Web Application](#web-application)
- [Future Plans](#future-plans)
- [References](#references)
- [License](#license)

## Introduction

This project implements a segmentation pipeline for histopathological images using a modified version of the UNet architecture, inspired by the original paper:

> Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18 (pp. 234-241). Springer International Publishing.

The project leverages the PanNuke dataset for nuclei segmentation, which is a diverse histological dataset spanning multiple cancer types.

## Features

### Modified UNet Architecture:
- Replaced transpose convolutions with bilinear interpolation to avoid checkerboard artifacts
- Added batch normalization for improved training stability
- Applied interpolation for odd input dimensions instead of cropping

### Training Pipeline:
Complete pipeline for training and evaluation.

### Web Application:
A user-friendly interface for visualizing and performing inference on the dataset.

### Planned Future Extensions:
- Generalizing to class segmentation and instance segmentation
- Implementation of alternative architectures (e.g., transformers)
- Incorporation of pretrained models for transfer learning

## Dataset

This project uses the PanNuke dataset:

> Gamper, J., Alemi Koohbanani, N., Benet, K., Khuram, A., & Rajpoot, N. (2019). Pannuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification. In Digital Pathology: 15th European Congress, ECDP 2019, Warwick, UK, April 10–13, 2019, Proceedings 15 (pp. 11-19). Springer International Publishing.

Key Details:
- Over 7,000 patches of size 256x256
- Includes images, masks (6-channel instance-wise masks for nuclei types), and tissue type annotations
- Unified nuclei categorization schema (e.g., Neoplastic, Inflammatory, Connective/Soft Tissue Cells, Dead Cells, Epithelial, Background)

## Model Architecture

This project uses a modified version of the UNet architecture. Key differences from the original implementation include:

- Bilinear Interpolation: Replacing transpose convolution layers to prevent checkerboard artifacts
- Batch Normalization: Improving gradient flow and model convergence
- Dynamic Input Handling: Adjustments to handle odd input dimensions without cropping

The architecture remains highly interpretable and effective for biomedical image segmentation tasks.

## Installation

Clone the repository:
```bash
git clone https://github.com/your-username/unet-histopathology.git
cd unet-histopathology
```

Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
pip install -r requirements.yaml
```

Download the PanNuke dataset (link in the Dataset section) and place it in the `data/` directory.

## Usage

### Web Application

Launch the web application to explore the dataset and perform inference interactively:

```bash
streamlit run app.py
```

Open the URL displayed in the terminal to access the app.

## Future Plans

This project will receive future updates, including:
- Improved training and hyperparameter tuning
- Extending the pipeline to support:
  - Class segmentation
  - Instance segmentation
- Implementing alternative architectures and pretrained models
- Expanding the web application's functionality

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In MICCAI 2015: Proceedings, Part III (pp. 234-241). Springer.

2. Gamper, J., Alemi Koohbanani, N., Benet, K., Khuram, A., & Rajpoot, N. (2019). Pannuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification. In ECDP 2019: Proceedings (pp. 11-19). Springer.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
