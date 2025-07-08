## Overview
This provides NumPy-based implementations of MRI reconstruction methods using structured low-rank modeling.

- **P-LORAKS**: Low-rank modeling for calibrationless parallel imaging reconstruction using multi-coil k-space data.
- **SENSE-LORAKS**: Integration of SENSE with low-rank k-space modeling for enhanced reconstruction quality.

Both scripts reconstruct high-quality images from undersampled multi-coil k-space data based on support constraints. Phase constraints can optionally be incorporated via Virtual Conjugate Coils (VCC) to enhance robustness.


## Features
- P-LORAKS: Calibrationless parallel imaging without ACS lines or sensitivity maps. 
- SENSE-LORAKS: Combines SENSE with low-rank matrix modeling. 


## Requirements
- detailed in the yaml file
- or, install dependencies using
```bash
pip install numpy scipy matplotlib
```

## Usage
- P-LORAKS (Calibrationless Reconstruction)
```bash
python p_loraks.py
```

#### Required Inputs (`dataset.mat`)
| Name    | Shape              | Description                              |
|---------|--------------------|------------------------------------------|
| `kData` | `(nx, ny, nCoils)` | Undersampled multi-coil k-space data     |
| `kMask` | `(nx, ny)` or `(nx, ny, 1)` or `(nx, ny, nCoils)` | Binary sampling mask         |

<br>

- SENSE-LORAKS (SENSE + Low-Rank Modeling)
```bash
python sense_loraks.py
```
#### Required Inputs (`dataset.mat`)

| Name        | Shape              | Description                              |
|-------------|--------------------|------------------------------------------|
| `kData`     | `(nx, ny, nCoils)` | Undersampled multi-coil k-space data     |
| `kMask`     | `(nx, ny)` or `(nx, ny, 1)` or `(nx, ny, nCoils)` | Binary sampling mask         |
| `coil_sens` | `(nx, ny, nCoils)` | Coil sensitivity maps (complex-valued)   |

<br>

#### Hyperparameters
Both scripts define the following key hyperparameters

| Parameter   | Description |
|-------------|-------------|
| `VCC`       | Enables Virtual Conjugate Coils to apply phase constraints |
| `R`         | Radius of circular k-space kernel |
| `lambda_`   | Regularization weight for the low-rank prior (stronger regularization = higher value) |
| `tol`       | Convergence tolerance for relative change in iterations |
| `max_iter`  | Maximum number of iterations allowed for reconstruction |
| `r_C`       | Truncated rank used in low-rank approximation (number of singular values retained) |
> Note: SENSE-LORAKS uses different default values compared to P-LORAKS.


## Reference
- Haldar JP, Zhuo J.  
  **P-LORAKS: Low-rank modeling of local k-space neighborhoods with parallel imaging data**  
  *Magnetic Resonance in Medicine*. 2016;75(4):1499–1514.  
  [https://doi.org/10.1002/mrm.25668](https://doi.org/10.1002/mrm.25717)

- Kim TH, Setsompop K, Haldar JP.  
  **LORAKS makes better SENSE: Phase‐constrained partial Fourier SENSE reconstruction without phase calibration**  
  *Magnetic Resonance in Medicine*. 2017;77(3):1021–1035.  
  [https://doi.org/10.1002/mrm.26288](https://doi.org/10.1002/mrm.26182)
 

