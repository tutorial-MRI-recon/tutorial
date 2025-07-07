## Overview

This project implements an unrolled supervised deep neural network for MRI reconstruction using a U-Net-based architecture and data consistency (DC) operation implemented via a conjugate gradient (CG) method. The model is trained in a supervised setting using a subset of T2-weighted brain MR images from the fastMRI dataset. 

Training is performed using `supervised_train.py`, while inference is handled through the `test.ipynb` notebook using a pre-trained model.

## Features

- Implements an unrolled supervised deep network using PyTorch.
- Handles multi-coil data
- Provides an example workflow for training.
- Provides an example for performing inference on a test sample using a pre-trained model `unet.pth` with CG layer `cg.pth`.


## Requirements

Install dependencies using:
```bash
conda env create -f environment.yml
```
## Usage

### Training

To train the model, run:

```bash
python supervised_train.py
```

The saved model was trained using 5 iterations for 100 epochs and 10 CG steps. Uniform undersampling masks were used for training. 

### Dataset Format

Your custom dataset class must return a dictionary with the following keys and shapes:

| Keys             | Shape               | Description                                   |
|-----------------|---------------------|-----------------------------------------------|
| `combined_us`   | `(2, H, W)`      | Coil combined zero-filled reconstruction (2 corresponds to real and imaginary channels, H and W are height and width)|
| `combined_full` | `(2, H, W)`      | Fully sampled ground truth image              |
| `us_coil`       | `(2, C, H, W)`  | Multi-coil zero-filled reconstruction (C corresponds to the  number of coils)         |
| `sensitivity`   | `(2, C, H, W)`  | Coil sensitivity maps                         |
| `mask`          | `(H, W)`         | Binary undersampling mask                     |

> **Note:** Replace the `MRIReconstructionDataset` class with your implementation that returns data in this format.

---

## Inference

To perform inference using a pre-trained model:

Open the `test.ipynb` notebook in Jupyter, and run each cell. Again, replace the `MRIReconstructionDataset` class with your implementation that returns data in the correct format.


