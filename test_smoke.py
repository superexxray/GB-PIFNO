import torch
from dataset import NSVorticityDataset
from sparsity import generate_pixel_mask, apply_sparsity
from fno import FNO2d
from unet import UNet2d
from eval_metrics import relative_l2_error
from pde_residual import compute_pde_residual
from train_gb_pifno import get_gradient_norm
import torch.nn as nn

def run_smoke_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running smoke test on device: {device}")
    
    # 1. Dataset test
    print("\n--- Testing Dataset Loading ---")
    dataset = NSVorticityDataset("NavierStokes_V1e-5_N1200_T20.mat", split="test")
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    
    # Sparsity
    mask = generate_pixel_mask(x.shape, 0.05, device)
    x_masked = apply_sparsity(x, mask)
    print(f"Masked Input shape: {x_masked.shape}")
    
    # 2. Architecture forward passes
    print("\n--- Testing Architectures ---")
    fno = FNO2d(in_channels=11, out_channels=10).to(device)
    unet = UNet2d(in_channels=11, out_channels=10).to(device)
    
    out_fno = fno(x_masked)
    out_unet = unet(x_masked)
    print(f"FNO Output shape: {out_fno.shape}")
    print(f"UNet Output shape: {out_unet.shape}")
    
    # 3. PDE Residual test
    print("\n--- Testing PDE Residual ---")
    x_last = x[:, -1:]
    omega_pred = torch.cat([x_last, out_fno], dim=1)
    pde_res = compute_pde_residual(omega_pred, dt=1.0, nu=1e-5)
    print(f"PDE Residual shape: {pde_res.shape}")
    
    # 4. Gradient Balancing test
    print("\n--- Testing Gradient Balancing ---")
    criterion = nn.MSELoss()
    l_data = criterion(out_fno, y)
    l_pde = torch.mean(pde_res**2)
    
    # Test autograd
    g_data = get_gradient_norm(l_data, fno, create_graph=False, retain_graph=True)
    g_pde = get_gradient_norm(l_pde, fno, create_graph=False, retain_graph=True)
    print(f"G_data norm: {g_data.item():.6f}")
    print(f"G_pde norm: {g_pde.item():.6f}")
    
    print("\nAll smoke tests passed successfully! 🎉")

if __name__ == "__main__":
    run_smoke_test()
