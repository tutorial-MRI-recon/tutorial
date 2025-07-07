import torch
import torch.nn as nn
# fft2c and ifft2c
def fftshift2d(x):
    return torch.fft.fftshift(x, dim=(-2, -1))

def ifftshift2d(x):
    return torch.fft.ifftshift(x, dim=(-2, -1))

def fft2c(x):
    return fftshift2d(torch.fft.fft2(ifftshift2d(x), norm='ortho'))

def ifft2c(x):
    return fftshift2d(torch.fft.ifft2(ifftshift2d(x), norm='ortho'))

def complex_to_magnitude(img):
    """
    Convert complex image (B, 2, H, W) to magnitude (B, 1, H, W)
    """
    return torch.sqrt(img[:, 0]**2 + img[:, 1]**2).unsqueeze(1)

# === Projection and DC ===
def project_to_coils(combined, sensitivity):
    combined_cplx = torch.view_as_complex(combined.permute(0, 2, 3, 1).contiguous())
    sens_cplx = torch.view_as_complex(sensitivity.permute(0, 2, 3, 4, 1).contiguous())
    projected = combined_cplx.unsqueeze(1) * sens_cplx
    return torch.view_as_real(projected).permute(0,  4, 1, 2, 3)

def combine_from_coils(coil_img, sensitivity):
    coil_cplx = torch.view_as_complex(coil_img.permute(0, 2, 3, 4, 1).contiguous())
    sens_cplx = torch.view_as_complex(sensitivity.permute(0, 2, 3, 4, 1).contiguous())
    numerator = torch.sum(coil_cplx * torch.conj(sens_cplx), dim=1)
    denominator = torch.sum(sens_cplx.abs()**2, dim=1) + 1e-8  
    combined = numerator / denominator  
    
    return torch.view_as_real(combined).permute(0, 3, 1, 2)

def apply_data_consistency(pred_coil, us_coil, mask):
    pred_k = fft2c(torch.view_as_complex(pred_coil.permute(0, 2, 3, 4, 1).contiguous()))
    us_k = fft2c(torch.view_as_complex(us_coil.permute(0, 2, 3, 4, 1).contiguous()))
    # Apply mask in the k-space
    pred_k = pred_k * (1 - mask) + us_k * mask
    img = ifft2c(pred_k)
    return torch.view_as_real(img).permute(0,  4, 1, 2, 3).contiguous()

def A_mult(x_img, sens_maps, mask):
    """
    Apply A to image
    """
    x_coil = project_to_coils(x_img, sens_maps)  # (B, 2, C, H, W)
    x_coil_cplx = torch.view_as_complex(x_coil.permute(0, 2, 3, 4, 1).contiguous())
    kspace = fft2c(x_coil_cplx)
    return kspace * mask  # Apply sampling mask

def A_adj_mult(kspace_data, sens_maps, mask):
    """
    Apply A^H to kspace
    """
    kspace = kspace_data * mask
    img = ifft2c(kspace)
    img_real = torch.view_as_real(img).permute(0, 4, 1, 2, 3).contiguous()
    combined = combine_from_coils(img_real, sens_maps)  # (B, 2, H, W)
    return combined

def cg_data_consistency(x_prior,us_coil, sens_maps, mask, lam=1e-3, max_iter=5, tol=1e-6):
    """
    Solves: (A^H A + λI)x = A^H y + λ x_prior where x_prior is DNN reccsontruction
    """
    us_kspace = fft2c(torch.view_as_complex(us_coil.permute(0, 2, 3, 4, 1).contiguous()))
    b = A_adj_mult(us_kspace, sens_maps, mask) + lam * x_prior
    x = x_prior.clone()
    r = b - A_adj_mult(A_mult(x, sens_maps, mask), sens_maps, mask) - lam * x
    p = r.clone()
    rsold = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)

    for _ in range(max_iter):
        Ap = A_adj_mult(A_mult(p, sens_maps, mask), sens_maps, mask) + lam * p
        alpha = rsold / (torch.sum(p * Ap, dim=(1, 2, 3), keepdim=True) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)
        if torch.sqrt(rsnew).mean() < tol:
            break
        p = r + (rsnew / (rsold + 1e-8)) * p
        rsold = rsnew
    return x

def estimate_object_mask(sens_maps, threshold=0.0001):
    """
    Estimate object mask from sensitivity maps by computing RSS over coils.
    sens_maps: (B, C, 2, H, W)
    Returns: mask of shape (B, 1, H, W)
    """
    # Convert sens_maps to complex and compute RSS
    sens_cplx = torch.view_as_complex(sens_maps.permute(0, 2, 3, 4, 1).contiguous())  # (B, C, H, W)
    rss = torch.sqrt(torch.sum(torch.abs(sens_cplx) ** 2, dim=1, keepdim=True))  # (B, 1, H, W)
    object_mask = (rss > threshold).float()
    return object_mask

class CGDataConsistency(nn.Module):
    def __init__(self, lam_init=0.05, max_iter=5, tol=1e-6):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(lam_init, dtype=torch.float32))  # Learnable λ
        self.max_iter = max_iter
        self.tol = tol

    def forward(self, x_prior, us_coil, sens_maps, mask):
        """
        Solves: (A^H A + λI)x = A^H y + λ x_prior where x_prior is DNN reccsontruction
        """
        us_kspace = fft2c(torch.view_as_complex(us_coil.permute(0, 2, 3, 4, 1).contiguous()))
        b = A_adj_mult(us_kspace, sens_maps, mask) + self.lam * x_prior
        x = x_prior.clone()
        r = b - A_adj_mult(A_mult(x, sens_maps, mask), sens_maps, mask) - self.lam * x
        p = r.clone()
        rsold = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)

        for _ in range(self.max_iter):
            Ap = A_adj_mult(A_mult(p, sens_maps, mask), sens_maps, mask) + self.lam * p
            alpha = rsold / (torch.sum(p * Ap, dim=(1, 2, 3), keepdim=True) + 1e-8)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.sum(r * r, dim=(1, 2, 3), keepdim=True)
            if torch.sqrt(rsnew).mean() < self.tol:
                break
            p = r + (rsnew / (rsold + 1e-8)) * p
            rsold = rsnew
        return x