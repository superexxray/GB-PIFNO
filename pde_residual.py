import torch
from eval_metrics import fourier_gradients

def compute_pde_residual(omega_pred, dt=1.0, nu=1e-5):
    """
    Computes the 2D incompressible Navier-Stokes PDE residual in vorticity form.
    omega_pred: predicted vorticity across time (B, T, H, W)
    dt: time step size
    nu: kinematic viscosity
    Returns: PDE residual tensor of shape (B, T-1, H, W)
    """
    B, T, H, W = omega_pred.shape
    
    # 1. Time derivative via finite difference across frames
    # shape: (B, T-1, H, W)
    domega_dt = (omega_pred[:, 1:] - omega_pred[:, :-1]) / dt
    
    # We will compute spatial derivatives for the first T-1 frames to match
    omega_t = omega_pred[:, :-1] # (B, T-1, H, W)
    
    # Reshape for fourier_gradients: (B*(T-1), 1, H, W)
    omega_flat = omega_t.reshape(B*(T-1), 1, H, W)
    
    # 2. Compute spatial derivatives of omega
    domega_dx, domega_dy, laplacian_omega = fourier_gradients(omega_flat)
    
    # 3. Stream function & Velocity field
    # omega = -laplacian(psi) => psi_h = -omega_h / (k^2)
    omega_h = torch.fft.fftn(omega_flat, dim=[2, 3])
    
    k_x = torch.fft.fftfreq(W, 1.0/W, device=omega_pred.device)
    k_y = torch.fft.fftfreq(H, 1.0/H, device=omega_pred.device)
    k_x = k_x.view(1, 1, 1, W)
    k_y = k_y.view(1, 1, H, 1)
    k_sq = (2 * torch.pi * k_x)**2 + (2 * torch.pi * k_y)**2
    k_sq[0, 0, 0, 0] = 1e-8 # avoid division by zero at k=0
    
    psi_h = -omega_h / k_sq
    psi_h[0, 0, 0, 0] = 0 # zero out mean
    
    # u = (u_x, u_y) = (dpsi/dy, -dpsi/dx)
    dpsi_dx_h = 1j * 2 * torch.pi * k_x * psi_h
    dpsi_dy_h = 1j * 2 * torch.pi * k_y * psi_h
    
    u_x = torch.fft.ifftn(dpsi_dy_h, dim=[2, 3]).real
    u_y = -torch.fft.ifftn(dpsi_dx_h, dim=[2, 3]).real
    
    # Advection term: u * grad(omega) = u_x * domega_dx + u_y * domega_dy
    advection = u_x * domega_dx + u_y * domega_dy
    
    # 4. Forcing term f(x, y) = 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    grid_x = torch.linspace(0, 1, W, device=omega_pred.device).view(1, 1, 1, W).expand(1, 1, H, W)
    grid_y = torch.linspace(0, 1, H, device=omega_pred.device).view(1, 1, H, 1).expand(1, 1, H, W)
    forcing = 0.1 * (torch.sin(2 * torch.pi * (grid_x + grid_y)) + torch.cos(2 * torch.pi * (grid_x + grid_y)))
    
    # 5. Assemble PDE residual
    # R = domega/dt + u.grad(omega) - nu*laplacian(omega) - f
    R_flat = domega_dt.reshape(B*(T-1), 1, H, W) + advection - nu * laplacian_omega - forcing
    
    R = R_flat.reshape(B, T-1, H, W)
    return R
