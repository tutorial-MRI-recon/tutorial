# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#%% imagesc def
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def imagesc( img, fig_num, clim, title = '' ):
      
    fig = plt.figure(fig_num)
    fig.patch.set_facecolor('black')

    plt.imshow( img )
    plt.gray()
    plt.clim(clim)
    plt.suptitle(title, color='white', fontsize=48)


 
def mosaic(img, num_row, num_col, fig_num, clim, title = '', use_transpose = False, use_flipud = False):
    
    fig = plt.figure(fig_num)
    fig.patch.set_facecolor('black')

    if img.ndim < 3:
        img_res = img
        plt.imshow(img_res)
        plt.gray()        
        plt.clim(clim)

    else:
        
        if img.shape[2] != (num_row * num_col):
            print('sizes do not match')    
        else:   
            
            if use_transpose:
                for slc in range(0, img.shape[2]):
                    img[:,:,slc] = np.transpose(img[:,:,slc])
            
            if use_flipud:
                img = np.flipud(img)
                
            
            img_res = np.zeros((img.shape[0]*num_row, img.shape[1]*num_col))
            
            idx = 0
            
            for r in range(0, num_row):
                for c in range(0, num_col):
                    img_res[r*img.shape[0] : (r+1)*img.shape[0], c*img.shape[1] : (c+1)*img.shape[1]] = img[:,:,idx]
                    idx = idx + 1
                
      
                
        plt.imshow(img_res)
        plt.gray()        
        plt.clim(clim)

        
    plt.suptitle(title, color='white', fontsize=48)
    
    
    

def imagesc3d( img, fig_num, scl, title = '' ):

    pos = np.divide( img.shape, 2. ).astype('int')
    
    plt.figure(fig_num)
    plt.subplot(1,3,1)
    imagesc(np.transpose(np.abs(img[:,:,pos[2]])), fig_num, scl)

    plt.subplot(1,3,2)
    imagesc(np.flip(np.transpose(np.abs(img[:,pos[1],:])), 0), fig_num, scl)

    plt.subplot(1,3,3)
    imagesc(np.flip(np.transpose(np.abs(img[pos[0],:,:])), 0), fig_num, scl, title)
    
    
    
    
def imagesc3d0( img, fig_num, scl, title = '' ):

    pos = np.divide( img.shape, 2. ).astype('int')
    
    plt.figure(fig_num)
    plt.subplot(1,3,1)
    imagesc(np.transpose(np.abs(img[:,:,pos[2]])), fig_num, scl)

    plt.subplot(1,3,2)
    imagesc(np.abs(img[:,pos[1],:]), fig_num, scl)

    plt.subplot(1,3,3)
    imagesc(np.abs(img[pos[0],:,:]), fig_num, scl, title)
    
    
    
def imagesc3d1( img, fig_num, scl, title = '', pos = [0,0,0] ):
  
    plt.figure(fig_num)
    plt.subplot(1,3,1)
    imagesc(np.transpose(np.abs(img[:,:,pos[2]])), fig_num, scl)

    plt.subplot(1,3,2)
    imagesc(np.abs(img[:,pos[1],:]), fig_num, scl)

    plt.subplot(1,3,3)
    imagesc(np.abs(img[pos[0],:,:]), fig_num, scl, title)
    
    
def imagesc3d2( img, fig_num, scl, title = '', pos = [0,0,0] ):
  
    plt.figure(fig_num)
    plt.subplot(1,3,1)
    imagesc(np.transpose(img[:,:,pos[2]]), fig_num, scl)

    plt.subplot(1,3,2)
    imagesc(img[:,pos[1],:], fig_num, scl)

    plt.subplot(1,3,3)
    imagesc(img[pos[0],:,:], fig_num, scl, title)
    
    

  
def imagesc3d3( img, fig_num, scl, title = '', pos = [0,0,0], rot=[0,0,0] ):
  
    import scipy.ndimage.interpolation as sni

    plt.figure(fig_num)
    plt.subplot(1,3,1)
    imagesc( sni.rotate(img[:,:,pos[2]], rot[0], axes=(0,1), reshape=True, order=5), fig_num, scl)

    plt.subplot(1,3,2)
    imagesc( sni.rotate(img[:,pos[1],:], rot[1], axes=(0,1), reshape=True, order=5), fig_num, scl)

    plt.subplot(1,3,3)
    imagesc( sni.rotate(img[pos[0],:,:], rot[1], axes=(0,1), reshape=True, order=5), fig_num, scl)    


    
    
def rsos(x, dim):
    return np.sqrt(np.sum(np.square(np.abs(x)), dim))

    

def ifft2call(data, dim1, dim2):
    shifted_rows = np.fft.ifftshift(data, axes=dim1)
    shifted_rows = np.fft.ifft(shifted_rows, axis=dim1)
    shifted_rows = np.fft.fftshift(shifted_rows, axes=dim1)


    shifted_rows = np.fft.ifftshift(shifted_rows, axes=dim2)
    shifted_rows = np.fft.ifft(shifted_rows, axis=dim2)
    shifted_rows = np.fft.fftshift(shifted_rows, axes=dim2)

    shifted_rows = shifted_rows * np.sqrt(shifted_rows.shape[dim1] * shifted_rows.shape[dim2])

    return shifted_rows



def fft2call(data, dim1, dim2):
    shifted_rows = np.fft.ifftshift(data, axes=dim1)
    shifted_rows = np.fft.fft(shifted_rows, axis=dim1)
    shifted_rows = np.fft.fftshift(shifted_rows, axes=dim1)


    shifted_rows = np.fft.ifftshift(shifted_rows, axes=dim2)
    shifted_rows = np.fft.fft(shifted_rows, axis=dim2)
    shifted_rows = np.fft.fftshift(shifted_rows, axes=dim2)

    shifted_rows = shifted_rows / np.sqrt(shifted_rows.shape[dim1] * shifted_rows.shape[dim2])

    return shifted_rows


def nrmse(img1, img2):
    # Compute the RMSE
    rmse = np.sqrt(np.mean(np.abs(img1 - img2) ** 2))
    
    # Compute the mean of img2
    mean_img2 = np.sqrt(np.mean(np.abs(img2) ** 2))
    
    # Compute the NRMSE
    nrmse_value = 100 * rmse / mean_img2
    
    return nrmse_value