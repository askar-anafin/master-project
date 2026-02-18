
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        
        # Projects input to hidden state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Conv1d for local features
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM Parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x: (Batch, Seq, D)
        batch, seq_len, d_model = x.shape
        
        x_and_res = self.in_proj(x) # (B, L, 2*D_inner)
        x_in, res = x_and_res.split(self.d_inner, dim=-1)
        
        # Conv1d expects (B, C, L)
        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[:, :, :seq_len]
        x_in = x_in.transpose(1, 2)
        
        x_in = F.silu(x_in)
        
        # SSM Implementation (Sequential Scan - Slow but Portable)
        y = self.ssm(x_in)
        
        y = y * F.silu(res)
        y = self.out_proj(y)
        return y

    def ssm(self, x):
        # x: (B, L, D_inner)
        d_in = self.d_inner
        A = -torch.exp(self.A_log.float()) # (D_inner, D_state)
        D = self.D.float() # (D_inner)
        
        delta_A_B_C = self.x_proj(x) # (B, L, dt_rank + 2*d_state)
        delta, B, C = torch.split(delta_A_B_C, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta)) # (B, L, D_inner)
        
        # Discretize
        # We need to run this sequentially per token in the sequence for the "Mamba" effect
        # This is the "Selective Scan" part
        
        # Simple Sequential Loop used for compatibility
        # For production Mamba, we'd use the CUDA kernel
        
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, d_in, self.d_state, device=x.device) # Hidden State
        y = []
        
        for t in range(seq_len):
            # Projections for this step
            dt = delta[:, t, :] # (B, D_inner)
            Bt = B[:, t, :] # (B, D_state)
            Ct = C[:, t, :] # (B, D_state)
            xt = x[:, t, :] # (B, D_inner)
            
            # Discretize A and B (Zero-Order Hold)
            # dA = exp(delta * A)
            # einsum('bd,dn->bdn', dt, A) equivalent to:
            dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0)) # (B, D_inner, D_state)
            
            # dB = (inverse(A) * (dA - I)) * delta * B  <-- approximation: delta * B
            # einsum('bd,bn->bdn', dt, Bt) equivalent to:
            dB = dt.unsqueeze(-1) * Bt.unsqueeze(1) # (B, D_inner, D_state) (Broadcasting (B, D, 1) * (B, 1, N))
            
            # State Update: h_t = dA * h_{t-1} + dB * x_t
            h = h * dA + dB * xt.unsqueeze(-1)
            
            # Output: y_t = C * h_t + D * x_t
            # einsum('bdn,bn->bd', h, Ct) equivalent to:
            # h: (B, D, N), Ct: (B, N) -> (B, N, 1) for matmul? No, h @ Ct.T?
            # We want for each b, d: sum_n h[b,d,n] * Ct[b,n]
            # h * Ct.unsqueeze(1) -> (B, D, N) * (B, 1, N) -> (B, D, N) -> sum(dim=-1) -> (B, D)
            yt = (h * Ct.unsqueeze(1)).sum(dim=-1) + D * xt
            y.append(yt)
            
        y = torch.stack(y, dim=1) # (B, L, D_inner)
        return y

class MambaECG(nn.Module):
    def __init__(self, num_classes=5, input_channels=12, d_model=128, num_layers=4, stride=4):
        super(MambaECG, self).__init__()
        
        # Project Input (B, 12, L) -> (B, L, 12) -> Embedding -> (B, L, D)
        # Downsample to reduce sequence length
        self.embedding = nn.Sequential(
            nn.Conv1d(input_channels, d_model, kernel_size=4, stride=stride, padding=0),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MambaBlock(d_model),
                RMSNorm(d_model)
            ]) for _ in range(num_layers)
        ])
        
        self.norm_f = RMSNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x: (B, L, 12) -> (B, 12, L)
        x = x.transpose(1, 2)
        
        # Embed
        x = self.embedding(x) # (B, D, L)
        x = x.transpose(1, 2) # (B, L, D) for Mamba
        
        for mamba_layer, norm in self.layers:
            x_res = x
            x = mamba_layer(norm(x))
            x = x + x_res
            
        x = self.norm_f(x)
        
        # Pooling (Mean)
        x = x.mean(dim=1) # (B, D)
        
        output = self.fc(x)
        return output
