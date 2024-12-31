# UNet for Histopathology Image Segmentation

![app demonstration Update](https://github.com/user-attachments/assets/f454dbf8-4cdf-4a47-96fa-e81bd7c9f3ae)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Application](#web-application)
- [Future Plans](#future-plans)
- [References](#references)
- [License](#license)

## Introduction

This repository serves as a playground for exploring the PanNuke dataset, offering an interactive web application that allows users to visualize segmentation results with diverse deep learning models and architectures. Whether you're a beginner eager to explore and experiment with segmentation or an experienced practitioner looking to dive into the PyTorch implementation of various segmentation and computer vision models, this project provides an easy-to-use training pipeline for training models from scratch.

## Features

### Models
- UNet (fully implemented Pytorch)
- UNet++ (fully implemented Pytorch)
- ResNet/UNet (fully implemented Pytorch - scratch)
- ResNet/Unet (Pretrained ResNet + Implemented UNet decoder Pytorch)

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

[dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)

[preprocess](https://github.com/Mr-TalhaIlyas/Prerpcessing-PanNuke-Nuclei-Instance-Segmentation-Dataset/blob/master/scripts/process_pannuke.py)

## Installation

Clone the repository:
```bash
git clone https://github.com/CedricCaruzzo/pannuke-segmentation/.git
cd pannuke-segmentation
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
- Extending the pipeline to support:
  - Class segmentation
  - Instance segmentation
- Implementing alternative architectures and pretrained models
- Expanding the web application's functionality

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In MICCAI 2015: Proceedings, Part III (pp. 234-241). Springer.

2. Gamper, J., Alemi Koohbanani, N., Benet, K., Khuram, A., & Rajpoot, N. (2019). Pannuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification. In ECDP 2019: Proceedings (pp. 11-19). Springer.

3. Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2018). Unet++: A nested u-net architecture for medical image segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support: 4th International Workshop, DLMIA 2018, and 8th International Workshop, ML-CDS 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 20, 2018, Proceedings 4 (pp. 3-11). Springer International Publishing.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
