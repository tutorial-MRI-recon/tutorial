## MRI reconstruction tutorial

This repository contains code, data references, and documentation for the **MRI Reconstruction Tutorial** accompanying the review paper:
**"A Tutorial on MRI Reconstruction: From Modern Methods to Clinical Implications"**.

---

## Overview

This tutorial is designed as a hands-on companion to the review paper, guiding readers from basic MRI physics principles to advanced reconstruction methods. It covers both traditional and modern approaches, including:

- Basic MRI signal acquisition and k-space
- Sensitivity encoding (SENSE)
- Compressed sensing (CS)
- Low-rank methods
- Self-supervised DL
- Supervised DL

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Tutorial Structure](#tutorial-structure)
3. [Installation](#installation)
4. [Data](#data)
5. [Acknowledgments](#Acknowledgments)

---

## Getting Started

This tutorial is intended for researchers, students, and practitioners interested in understanding and implementing various MRI reconstruction techniques.

You can run the code locally or in a Jupyter environment. Most examples are built mainly using SigPy, TensorFlow, PyTorch, and ZS-SSL.

---

## Tutorial Structure

Each section in the tutorial builds progressively on previous concepts. The notebook-based tutorials are located in each directory:

| Section | Methodology | Folder/Notebook |
|--------|-------------|-----------------|
| 1. MRI Physics | Basic MRI physics, including k-space, Fourier transform, coil sensitivity | `basic_sense_cs/script_basic_sense_cs.py` |
| 2. SENSE | Parallel imaging with coil sensitivity maps | `basic_sense_cs/script_basic_sense_cs.py` |
| 3. Compressed Sensing | CS with L1-regularized methods | `basic_sense_cs/script_basic_sense_cs.py` |
| 4. Low-Rank Methods | low-rank | `low_rank/low_rank.py` |
| 5. Supervised DL | Unrolled model using fully-sampled data | `supervised_dl/supervised_train.py` |
| 6. Self-Supervised DL | Zero-shot self-supervised learning with single k-space data | `scan_specific_dl/zs_ssl_train.py` |

---

## Installation

We recommend using a virtual environment. Install dependencies using .yaml file:

```bash
git clone https://github.com/yourusername/mri-reconstruction-tutorial.git
cd mri-reconstruction-tutorial/path_to_each_method
conda env create -f environment.yaml
conda activate name_of_env
```

---

## Data
To replicate the results, please download the raw [fastMRI](https://fastmri.med.nyu.edu/) data used in the tutorial from [here](https://www.dropbox.com/scl/fi/9ag9tj6oeci1x6uxaq6s4/file_brain_AXT2_205_2050058.h5?rlkey=xmr8o0xlzffjxsg604yup7vvw&dl=0).

---

## Acknowledgments
We thank the developers and contributors of:

[fastMRI](https://github.com/facebookresearch/fastMRI)
[SigPy](https://github.com/mikgroup/sigpy)
[TensorFlow](https://github.com/tensorflow/tensorflow)
[PyTorch](https://github.com/pytorch/pytorch)
[ZS-SSL](https://github.com/byaman14/ZS-SSL)
