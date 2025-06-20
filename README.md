- this readme file accompanies the python script, script_sigpy_sense_wavelet_R4_uniform.py, as part of the TBME image reconstruction tutorial

- environment.yaml file should allow for creating the environment to reproduce the results

- to replicate the results, please download the raw Fast MRI data used in the tutorial from:
https://www.dropbox.com/scl/fi/9ag9tj6oeci1x6uxaq6s4/file_brain_AXT2_205_2050058.h5?rlkey=xmr8o0xlzffjxsg604yup7vvw&dl=0

## Overview
This script demonstrates the use of the SigPy library for performing SENSE (Sensitivity Encoding) 
reconstruction as well compressed sensing alongside a simple gradient descent method for comparison.

## Features
- Implements SENSE reconstruction using SigPy.
- Handles multi-coil MRI data.
- Provides an example workflow for reconstructing undersampled k-space data.
- Includes compressed sensing reconstruction.
- Compares the results with a simple gradient descent method.

## Requirements
- detailed in the yaml file

## Usage
- Ensure you have the required libraries installed.
- Run the script in an environment with access to the required libraries
  -> having installed jupyter will allow for running each cell individually

