import torch
import torch.nn as nn
from dataset import NSVorticityDataset
from fno import FNO2d
from sparsity import generate_pixel_mask, apply_sparsity
from eval_metrics import relative_l2_error, h1_relative_error
from pde_residual import compute_pde_residual

def train_static_pifno(sparsity_level=0.05, epochs=100, batch_size=20, lr=1e-3, lam=1.0, device='cuda'):
    mat_file = "NavierStokes_V1e-5_N1200_T20.mat"
    train_dataset = NSVorticityDataset(mat_file, split='train')
    test_dataset = NSVorticityDataset(mat_file, split='test')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = FNO2d(in_channels=11, out_channels=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion_data = nn.MSELoss()
    history = {'l_data': [], 'l_pde': []}

    for epoch in range(epochs):
        model.train()
        train_l_data = 0.0
        train_l_pde = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            mask = generate_pixel_mask(x.shape, sparsity_level, device)
            x_masked = apply_sparsity(x, mask)
            
            optimizer.zero_grad()
            
            # Forward pass
            out = model(x_masked) # shape: (B, 10, H, W)
            
            # Data loss
            l_data = criterion_data(out, y)
            
            # PDE loss
            # Construct full trajectory: x (last timestep) + predicted y
            # Actually, x has 10 timesteps. We take the last frame of x and concat with out.
            x_last = x[:, -1:] # (B, 1, H, W)
            omega_pred = torch.cat([x_last, out], dim=1) # (B, 11, H, W)
            
            pde_res = compute_pde_residual(omega_pred, dt=1.0, nu=1e-5) # (B, 10, H, W)
            l_pde = torch.mean(pde_res**2)
            
            # Total loss
            loss = l_data + lam * l_pde
            
            loss.backward()
            optimizer.step()
            
            train_l_data += l_data.item()
            train_l_pde += l_pde.item()
            
        scheduler.step()
        train_l_data /= len(train_loader)
        train_l_pde /= len(train_loader)
        
        # Eval
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                mask = generate_pixel_mask(x.shape, sparsity_level, device)
                x_masked = apply_sparsity(x, mask)
                out = model(x_masked)
                test_l2 += relative_l2_error(out, y).item()
                
        
        history['l_data'].append(train_l_data)
        history['l_pde'].append(train_l_pde)
        print(f"Epoch {epoch} - L_data: {train_l_data:.4f}, L_pde: {train_l_pde:.4f}, Test L2: {test_l2:.4f}")

    import json
    with open('static_history.json', 'w') as f:
        json.dump(history, f)
    print("Saved training history to static_history.json")

if __name__ == '__main__':
    train_static_pifno(epochs=5, batch_size=2, device='cuda' if torch.cuda.is_available() else 'cpu')
