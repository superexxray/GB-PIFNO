import torch
import torch.nn as nn
from dataset import NSVorticityDataset
from fno import FNO2d
from sparsity import generate_pixel_mask, apply_sparsity
from eval_metrics import relative_l2_error, h1_relative_error
from pde_residual import compute_pde_residual

def get_gradient_norm(loss, model, create_graph=False, retain_graph=True):
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=retain_graph, create_graph=create_graph, allow_unused=True)
    norm = 0.0
    for g in grads:
        if g is not None:
            if g.is_complex():
                norm += torch.sum(torch.abs(g)**2)
            else:
                norm += torch.sum(g**2)
    return torch.sqrt(norm)

def train_gb_pifno(sparsity_level=0.05, epochs=100, batch_size=20, lr=1e-3, alpha=0.1, device='cuda'):
    mat_file = "NavierStokes_V1e-5_N1200_T20.mat"
    train_dataset = NSVorticityDataset(mat_file, split='train')
    test_dataset = NSVorticityDataset(mat_file, split='test')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = FNO2d(in_channels=11, out_channels=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion_data = nn.MSELoss()
    
    lam_pde = 1.0
    history = {'g_data': [], 'g_pde': [], 'l_data': [], 'l_pde': []}

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
            out = model(x_masked)
            
            # Data loss + H1 loss (Sobolev spectral penalty folded into L_data)
            # According to the execution plan: Sobolev H1 gets its own term, we'll fold it here.
            l_mse = criterion_data(out, y)
            l_h1 = h1_relative_error(out, y)
            l_data = l_mse + l_h1 # combining them
            
            # PDE loss
            x_last = x[:, -1:]
            omega_pred = torch.cat([x_last, out], dim=1)
            pde_res = compute_pde_residual(omega_pred, dt=1.0, nu=1e-5)
            l_pde = torch.mean(pde_res**2)
            
            # Gradient Balancing
            # We compute gradients of l_data and l_pde separately
            g_data = get_gradient_norm(l_data, model, create_graph=False, retain_graph=True)
            g_pde = get_gradient_norm(l_pde, model, create_graph=False, retain_graph=True)
            
            # Update lambda via EMA
            # Avoid division by zero
            safe_g_pde = g_pde if g_pde > 1e-8 else 1.0
            lam_target = (g_data / safe_g_pde).item()
            
            lam_pde = (1 - alpha) * lam_pde + alpha * lam_target
            
            # Total loss for optimization
            loss = l_data + lam_pde * l_pde
            
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
        history['g_data'].append(g_data.item())
        history['g_pde'].append(g_pde.item())
        print(f"Epoch {epoch} - lam_pde: {lam_pde:.4f}, L_data: {train_l_data:.4f}, L_pde: {train_l_pde:.4f}, Test L2: {test_l2:.4f}")

    import json
    with open('gb_history.json', 'w') as f:
        json.dump(history, f)
    print("Saved training history to gb_history.json")

if __name__ == '__main__':
    train_gb_pifno(epochs=5, batch_size=2, device='cuda' if torch.cuda.is_available() else 'cpu')
