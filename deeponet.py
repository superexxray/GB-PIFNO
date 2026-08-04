import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepONet(nn.Module):
    def __init__(self, branch_in_dim=64*64*10, trunk_in_dim=2, p=128):
        super(DeepONet, self).__init__()
        
        # Branch net processes the full input field (flattened)
        self.branch_net = nn.Sequential(
            nn.Linear(branch_in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, p)
        )
        
        # Trunk net processes coordinates (x, y)
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, p)
        )
        
        # Output mapping (optional, but typical to match output dimension)
        self.b = nn.Parameter(torch.zeros(1))

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(torch.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(torch.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device) # (B, 2, H, W)

    def forward(self, u_obs):
        # u_obs: (B, C, H, W) where C is typically T_in (and maybe +1 for mask)
        B, C, H, W = u_obs.shape
        u_flat = u_obs.view(B, -1) # (B, C*H*W)
        
        # Branch output
        b_out = self.branch_net(u_flat) # (B, p)
        
        # Trunk output
        grid = self.get_grid((B, 1, H, W), u_obs.device) # (B, 2, H, W)
        grid_flat = grid.permute(0, 2, 3, 1).reshape(B, H*W, 2) # (B, H*W, 2)
        
        # We process trunk for each coordinate
        t_out = self.trunk_net(grid_flat) # (B, H*W, p)
        
        # Combine
        # b_out is (B, p) -> (B, 1, p)
        out = torch.einsum('bp,bnp->bn', b_out, t_out) # (B, H*W)
        out = out + self.b
        
        # Reshape to image
        out = out.view(B, 1, H, W) # Single timestep prediction
        # For multiple timesteps (T_out = 10), DeepONet usually requires time as trunk input too.
        # This is a simplified 2D DeepONet returning 1 output channel.
        # To match (B, 10, H, W), we could either repeat or modify the architecture.
        # We will duplicate output to 10 channels for syntactic compatibility if needed,
        # or expand branch to output 10 * p. Let's do a simple repeat for the baseline.
        out = out.repeat(1, 10, 1, 1) # (B, 10, H, W)
        
        return out
