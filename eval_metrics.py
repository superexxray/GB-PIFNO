import torch
import numpy as np

def relative_l2_error(pred, target):
    """
    Compute relative L2 error over spatial dimensions.
    Assumes shape (Batch, Channels, H, W).
    """
    # pred, target: (B, C, H, W)
    diff_norms = torch.norm(pred - target, p=2, dim=(2, 3))
    target_norms = torch.norm(target, p=2, dim=(2, 3))
    return torch.mean(diff_norms / target_norms)

def fourier_gradients(u):
    """
    Computes exact spatial derivatives via FFT.
    Returns du/dx, du/dy, and Laplacian del^2 u.
    Input u: (B, C, H, W)
    """
    B, C, H, W = u.shape
    u_h = torch.fft.fftn(u, dim=[2, 3])
    
    # Wavenumbers
    k_x = torch.fft.fftfreq(W, 1.0/W, device=u.device)
    k_y = torch.fft.fftfreq(H, 1.0/H, device=u.device)
    
    k_x = k_x.view(1, 1, 1, W)
    k_y = k_y.view(1, 1, H, 1)
    
    # 1st derivatives: ik_x * u_h
    du_dx_h = 1j * 2 * torch.pi * k_x * u_h
    du_dy_h = 1j * 2 * torch.pi * k_y * u_h
    
    du_dx = torch.fft.ifftn(du_dx_h, dim=[2, 3]).real
    du_dy = torch.fft.ifftn(du_dy_h, dim=[2, 3]).real
    
    # Laplacian: -(k_x^2 + k_y^2) * u_h
    laplacian_h = -((2 * torch.pi * k_x)**2 + (2 * torch.pi * k_y)**2) * u_h
    lap_u = torch.fft.ifftn(laplacian_h, dim=[2, 3]).real
    
    return du_dx, du_dy, lap_u

def h1_relative_error(pred, target):
    """
    Sobolev H1 relative error.
    Combines L2 error of values and gradients.
    """
    # L2 of value
    val_diff = pred - target
    val_diff_norm = torch.sum(val_diff**2, dim=(2,3))
    val_targ_norm = torch.sum(target**2, dim=(2,3))
    
    # L2 of gradient
    pred_dx, pred_dy, _ = fourier_gradients(pred)
    target_dx, target_dy, _ = fourier_gradients(target)
    
    grad_diff_norm = torch.sum((pred_dx - target_dx)**2 + (pred_dy - target_dy)**2, dim=(2,3))
    grad_targ_norm = torch.sum(target_dx**2 + target_dy**2, dim=(2,3))
    
    h1_err = torch.mean(torch.sqrt((val_diff_norm + grad_diff_norm) / (val_targ_norm + grad_targ_norm)))
    return h1_err

def enstrophy(omega):
    """
    Enstrophy = 0.5 * integral(omega^2 dx)
    Input omega: (B, C, H, W)
    Returns: (B, C) enstrophy per sample per timestep
    """
    return 0.5 * torch.mean(omega**2, dim=(2, 3)) # assuming dx=1/H, dy=1/W for mean

def radial_energy_spectrum(omega):
    """
    1D angle-averaged energy spectrum E(k).
    Since omega = del x u, E(k) ~ |u|^2 ~ |omega|^2 / k^2.
    """
    B, C, H, W = omega.shape
    omega_h = torch.fft.fftn(omega, dim=[2, 3])
    
    # Power spectrum of vorticity
    power = torch.abs(omega_h)**2 / (H * W)**2
    
    k_x = torch.fft.fftfreq(W, 1.0/W, device=omega.device)
    k_y = torch.fft.fftfreq(H, 1.0/H, device=omega.device)
    kx_grid, ky_grid = torch.meshgrid(k_x, k_y, indexing='ij')
    kx_grid = kx_grid.T
    ky_grid = ky_grid.T
    
    k_sq = kx_grid**2 + ky_grid**2
    k_radial = torch.sqrt(k_sq)
    k_radial = torch.round(k_radial).to(torch.int)
    
    # Energy spectrum E(k) = (1/k^2) * VorticitySpectrum(k)
    # Avoid division by zero at k=0
    k_sq[k_sq == 0] = 1e-8
    energy_2d = power / (k_sq.unsqueeze(0).unsqueeze(0))
    
    max_k = int(torch.max(k_radial).item())
    spectrum = torch.zeros((B, C, max_k + 1), device=omega.device)
    
    # Accumulate radially
    for i in range(max_k + 1):
        mask = (k_radial == i).unsqueeze(0).unsqueeze(0)
        spectrum[:, :, i] = torch.sum(energy_2d * mask, dim=(2, 3))
        
    return spectrum
