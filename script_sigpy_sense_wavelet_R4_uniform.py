# %% load k-space data, display rsos combination across coils for all slices

from matplotlib import pyplot as plt
import scipy.io as sio
import LIBRARY as lib

import numpy as np
import sigpy as sp

import sigpy.mri as mri
import sigpy.mri.app as app

import h5py

file_name = 'fastmri_data/file_brain_AXT2_205_2050058.h5' 

hf = h5py.File(file_name)

volume_kspace = hf['kspace'][()]

# all slices:
kspace_all = np.transpose(volume_kspace, (2, 3, 1, 0))
img_all = lib.ifft2call(kspace_all, 0, 1)

# flip the image upside down  
img_all = np.flipud(img_all)  

# Determine the size of the 0th dimension -> remove 2x readout over-sampling
dim0_size = img_all.shape[0]

# Calculate the start and end indices for cropping
start_idx = (dim0_size - 320) // 2
end_idx = start_idx + 320

# Crop the middle 320 points in the 0th dimension
img_cropped = img_all[start_idx:end_idx, :, :, :]

img_cropped_rsos = lib.rsos(img_cropped, 2)

lib.mosaic(img_cropped_rsos[:,:,0:8], 2, 4, 1, [0,5e-4], 'image rsos 058', 0, 0)

# Save the current figure as a .png file
#plt.savefig('image_rsos_058.png', dpi=300, bbox_inches='tight')


# %% select a slice from the cropped image

slice_index = 1  # Select the 1st slice (0-based index)

img = img_cropped[:,:,:,slice_index]
img_rsos = lib.rsos(img, 2)

kspace = lib.fft2call(img, 0, 1)
kspace_rsos = lib.rsos(kspace, 2)

lib.mosaic(img_rsos, 1, 1, 2, [0,5e-4], 'image slice rsos', 0, 0)
lib.mosaic(kspace_rsos, 1, 1, 3, [0,5e-4], 'kspace slice rsos', 0, 0)


#%% Compute sensitivity maps using ESPIRiT

# transpose so that coils are in the 0th dimension
kspace_calib = np.transpose(kspace, (2, 0, 1))

calib_region = 24   # Calibration region size
crop_size = 0.8     # threshold for cropping the sensitivity maps
device = sp.Device(-1)  # Use CPU 

sensitivity_maps = app.EspiritCalib(kspace_calib, calib_width=calib_region, device=device, crop=crop_size).run()

print("Sensitivity maps shape:", sensitivity_maps.shape)  # [num_coils, height, width]

# switch coils back to the last dimension
sensitivity_maps = np.transpose(sensitivity_maps, (1, 2, 0))

# Display the sensitivity maps      
lib.mosaic(np.abs(sensitivity_maps), 4, 5, 3, [0,1], 'coil sensitivities', 0, 1)


# %% create sampling pattern: uniform 

accel = 4   # acceleration factor
img_shape = sensitivity_maps[:,:,0].shape

msk = np.zeros(img_shape, dtype=float)  # Initialize with the correct shape
msk[:, 0::accel] = 1   

# include the center of k-space
center_region = calib_region  # Size of the center region to keep
msk[:, img_shape[1]//2 - center_region//2:img_shape[1]//2 + center_region//2] = 1

msk = msk[..., np.newaxis]  # Add dimensions to match kspace

lib.mosaic(msk[:,:,0], 1, 1, 4, [0,1], 'sampling mask', 0, 1)

accl_factor = np.prod(msk.shape)/ np.sum(msk) 
print('actual acceleration factor: ', accl_factor)


# %% R=1 sense coil combination: ground truth

img_combo = np.sum(img * np.conj(sensitivity_maps),2)

lib.mosaic(np.abs(img_combo), 2, 3, 2, [0,3e-4], 'R=1 combo w/ sense maps', 0, 1)


# %% zero fill reconstruction

kspace_accl = kspace * msk  # undersample k-space data

img_zf = lib.ifft2call(kspace_accl, 0, 1)

img_zf_rsos = lib.rsos(img_zf, 2)

# coil combine with estimated sensitivities
img_zf_combo = np.sum(img_zf * np.conj(sensitivity_maps), 2)

lib.mosaic(np.abs(img_zf_combo), 1, 1, 1, [0,5e-4], 'coil combine: zero fill', 0, 0)

rmse_zf = lib.nrmse(img_zf_combo, img_combo)

print('zero fill RMSE: ', rmse_zf, ' %')

lib.mosaic(np.abs(img_zf_combo - img_combo), 1, 1, 5, [0, 5e-4], 'zero fill error', 0, 0)


# %% perform sense reconstruction: sigpy

lambda_reg = 0

# Initialize a list to store the reconstructed images
sense_reconstructions = []

sens = np.transpose(sensitivity_maps, (2, 0, 1))  # Transpose so that coils are in the first dimension

# Transpose the k-space data to have coils in the first dimension
kspace_use = np.transpose(kspace_accl, (2, 0, 1))

# Perform SENSE reconstruction for the current echo
sense_recon = app.SenseRecon(
    y = kspace_use,  # Undersampled k-space data for the slice
    mps = sens,  # Coil sensitivity maps
    lamda = lambda_reg,  # Regularization parameter
    device = sp.Device(-1),  # Use CPU (-1) or GPU (e.g., 0 for the first GPU)
    show_pbar = True  # progress bar for each echo
)

# Run the reconstruction and store the result
img_sense = sense_recon.run()
sense_reconstructions.append(img_sense)

# Convert the list to a NumPy array for easier handling
sense_reconstructions = np.stack(sense_reconstructions, axis=2)

# Display the reconstructed echoes
lib.mosaic(np.abs(sense_reconstructions), 1, 1, 4, [0, 5e-4], 'SENSE Reconstruction: sigpy', 0, 0)


# compute rmse from sigpy sense recon
rmse_sigpy = lib.nrmse(sense_reconstructions[:,:,0], img_combo)

print('sigpy SENSE RMSE: ', rmse_sigpy, ' %')

lib.mosaic(2*np.abs(sense_reconstructions[:,:,0] - img_combo), 1, 1, 4, [0, 5e-4], '2x SENSE error: sigpy', 0, 0   )


# %% simple gradient descent sense recon for comparison

num_iters = 100 # number of iterations for gradient descent
step_size = 2   # gradient descent step size

cs = np.conj(sensitivity_maps)  # Conjugate of the sensitivity maps

x = img_zf_combo    # Initialize x with the zero-filled reconstruction
x = x[..., np.newaxis]

for j in range(num_iters):
    # Display iteration number every 10 iterations
    if (j + 1) % 10 == 0:
        print(f"Completed iteration {j + 1}/{num_iters}")

    # Ax-b
    res = msk * lib.fft2call(x * sensitivity_maps, 0, 1) - kspace_accl

    # x = x - step_size * A'(Ax-b)
    tmp = np.sum(cs * lib.ifft2call(res, 0, 1), 2)
    x = x - step_size * tmp[..., np.newaxis]

img_recon = x[:,:,0]

# Display the reconstructed image
lib.mosaic(np.abs(img_recon), 1, 1, 4, [0, 5e-4], 'SENSE: Gradient Descent', 0, 0)

 
# compute nrmse for gradient descent
rmse_grad = lib.nrmse(img_recon, img_combo)

print('gradient descent RMSE: ', rmse_grad, ' %')

lib.mosaic(2*np.abs(img_recon - img_combo), 1, 1, 5, [0, 5e-4], '2x gradient descent error', 0, 0)


# %% wavelet regularized reconstruction: sigpy 

# Parameters for L1 wavelet reconstruction
lambda_wavelet = 2e-6  # Regularization parameter for wavelet sparsity

    
# Perform L1 Wavelet reconstruction
l1_wavelet_recon = app.L1WaveletRecon(
    y = kspace_use,  # Undersampled k-space data
    mps = sens,  # Coil sensitivity maps
    lamda = lambda_wavelet,  # Regularization parameter
    device = sp.Device(-1),  # Use CPU (-1) or GPU (e.g., 0 for the first GPU)
    show_pbar = True  # Show progress bar
)

# Run the reconstruction and store the result
img_l1_wavelet = l1_wavelet_recon.run()

non_zero_mask = np.abs(sensitivity_maps[:,:,0]) > 0

# apply the non-zero mask to the reconstructed image
l1_wavelet_reconstructions = img_l1_wavelet * non_zero_mask

# Display the reconstructed echoes
lib.mosaic(np.abs(l1_wavelet_reconstructions), 1, 1, 4, [0, 5e-4], 'L1 Wavelet Recon', 0, 0)


# compute rmse from sigpy wavelet recon
rmse_wavelet = lib.nrmse(l1_wavelet_reconstructions, img_combo)

print('sigpy wavelet RMSE: ', rmse_wavelet, ' %')

lib.mosaic(2*np.abs(l1_wavelet_reconstructions - img_combo), 1, 1, 5, [0, 3e-4], '2x wavelet error', 0, 0)
