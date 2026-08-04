import torch
import torch.nn as nn
from dataset import NSVorticityDataset
from fno import FNO2d
from unet import UNet2d
from sparsity import generate_pixel_mask, apply_sparsity
from eval_metrics import relative_l2_error, h1_relative_error

def train_data_only(model_type='fno', sparsity_level=0.05, epochs=100, batch_size=20, lr=1e-3, device='cuda'):
    # Setup dataset
    # Placeholder path for structural completeness
    mat_file = "NavierStokes_V1e-5_N1200_T20.mat"
    train_dataset = NSVorticityDataset(mat_file, split='train')
    test_dataset = NSVorticityDataset(mat_file, split='test')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model setup
    if model_type == 'fno':
        # Input channels: T_in + 1 (for mask)
        model = FNO2d(in_channels=11, out_channels=10).to(device)
    elif model_type == 'unet':
        model = UNet2d(in_channels=11, out_channels=10).to(device)
    else:
        raise ValueError("Invalid model type")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_mse = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Apply sparsity
            mask = generate_pixel_mask(x.shape, sparsity_level, device)
            x_masked = apply_sparsity(x, mask)
            
            optimizer.zero_grad()
            out = model(x_masked)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_mse += loss.item()
            
        scheduler.step()
        train_mse /= len(train_loader)
        
        # Eval
        model.eval()
        test_l2 = 0.0
        test_h1 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                mask = generate_pixel_mask(x.shape, sparsity_level, device)
                x_masked = apply_sparsity(x, mask)
                
                out = model(x_masked)
                test_l2 += relative_l2_error(out, y).item()
                test_h1 += h1_relative_error(out, y).item()
                
        test_l2 /= len(test_loader)
        test_h1 /= len(test_loader)
        
        print(f"Epoch {epoch} - Train MSE: {train_mse:.4f}, Test L2: {test_l2:.4f}, Test H1: {test_h1:.4f}")

if __name__ == '__main__':
    # train_data_only('fno', epochs=1, batch_size=2, device='cpu')
    pass
