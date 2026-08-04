import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorizedSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(FactorizedSpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        
        # Factorized weights (e.g. into two smaller matrices)
        rank = 8
        self.weights1_u = nn.Parameter(self.scale * torch.rand(in_channels, rank, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights1_v = nn.Parameter(self.scale * torch.rand(rank, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        
        self.weights2_u = nn.Parameter(self.scale * torch.rand(in_channels, rank, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2_v = nn.Parameter(self.scale * torch.rand(rank, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d_factorized(self, input, wu, wv):
        # (batch, in, x, y) * (in, rank, x, y) -> (batch, rank, x, y)
        tmp = torch.einsum("bixy,irxy->brxy", input, wu)
        # (batch, rank, x, y) * (rank, out, x, y) -> (batch, out, x, y)
        return torch.einsum("brxy,roxy->boxy", tmp, wv)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d_factorized(x_ft[:, :, :self.modes1, :self.modes2], self.weights1_u, self.weights1_v)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d_factorized(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2_u, self.weights2_v)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FFNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=32, in_channels=10, out_channels=10):
        super(FFNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.p = nn.Linear(in_channels + 2, self.width)
        
        self.conv0 = FactorizedSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = FactorizedSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = FactorizedSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = FactorizedSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        self.q = nn.Linear(self.width, 128)
        self.out = nn.Linear(128, out_channels)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(torch.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(torch.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        
        x = x.permute(0, 2, 3, 1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.q(x)
        x = F.gelu(x)
        x = self.out(x)
        x = x.permute(0, 3, 1, 2)
        
        return x
