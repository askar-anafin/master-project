import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation1D, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Input shape: (Batch, Channels, Seq)
        b, c, _ = x.size()
        # Global Average Pooling
        y = x.mean(dim=2) # (Batch, Channels)
        y = self.fc(y).view(b, c, 1) # (Batch, Channels, 1)
        return x * y

class ResNetTransformer(nn.Module):
    def __init__(self, num_classes=5, input_channels=12, seq_len=5000, 
                 d_model=256, nhead=8, num_transformer_layers=3, dim_feedforward=512):
        super(ResNetTransformer, self).__init__()
        
        # 1. 1D CNN Front-End (Extracts local patterns and downsamples)
        # Input shape: (Batch, input_channels, seq_len)
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=15, stride=2, padding=7, bias=False) # L -> L/2 (2500)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # L/2 -> L/4 (1250)
        
        # ResNet stages
        self.in_channels = 64
        self.layer1 = self._make_layer(64, 2, stride=1)    # 1250
        self.layer2 = self._make_layer(128, 2, stride=2)   # 625
        self.layer3 = self._make_layer(256, 2, stride=2)   # 313
        
        # Squeeze-and-Excitation (SE) / Channel Attention to focus on key leads/channels
        self.se_block = SqueezeExcitation1D(256)
        
        # Projection to Transformer d_model
        self.proj = nn.Conv1d(256, d_model, kernel_size=1)
        
        # 2. Transformer Encoder Back-End (Models global sequence context)
        # Feature sequence length after Layer3 is 313
        self.feature_seq_len = 313 
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.feature_seq_len + 1, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # 3. Classifier Head
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )
        
    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock1D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNetBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        # Input shape: (Batch, Seq, Leads) -> Transpose to (Batch, Leads, Seq) for Conv1d
        x = x.transpose(1, 2)
        
        # CNN Feature Extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) # Shape: (Batch, 256, 313)
        
        # Channel Attention
        x = self.se_block(x)
        
        # Project channels to d_model
        x = self.proj(x) # Shape: (Batch, d_model, 313)
        x = x.transpose(1, 2) # Shape: (Batch, 313, d_model)
        
        # Add CLS token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # Shape: (Batch, 314, d_model)
        
        # Add Positional Embedding
        x = x + self.pos_embedding
        
        # Transformer Encoder
        x = self.transformer(x) # Shape: (Batch, 314, d_model)
        
        # Extract CLS token output
        cls_out = x[:, 0]
        cls_out = self.norm(cls_out)
        
        # Classification
        out = self.fc(cls_out)
        return out
