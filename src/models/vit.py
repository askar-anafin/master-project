
import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=12, d_model=128, patch_size=50):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, L)
        x = self.proj(x) # (B, D, L_new)
        x = x.transpose(1, 2) # (B, L_new, D)
        return x

class ViT1D(nn.Module):
    def __init__(self, num_classes=5, input_channels=12, seq_len=5000, patch_size=50, d_model=128, nhead=4, num_layers=4, dim_feedforward=256):
        super(ViT1D, self).__init__()
        
        self.d_model = d_model
        
        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(input_channels, d_model, patch_size)
        num_patches = seq_len // patch_size
        
        # 2. CLS Token & Positional Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. MLP Head
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, L, 12) -> (B, 12, L)
        x = x.transpose(1, 2)
        
        # Flatten patches: (B, N, D)
        x = self.patch_embed(x)
        batch_size = x.shape[0]
        
        # Add CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Positional Embedding
        x = x + self.pos_embedding
        
        # Transformer
        x = self.transformer(x)
        
        # Take CLS Token output
        cls_out = x[:, 0]
        cls_out = self.norm(cls_out)
        
        # Classify
        out = self.fc(cls_out)
        return out
