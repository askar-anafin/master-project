import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaselineCNN1D(nn.Module):
    def __init__(self, num_classes=5, input_channels=12):
        super(BaselineCNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, distinct_channels) -> need (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x # Logits (use BCEWithLogitsLoss for multi-label)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
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

class ResNet1D(nn.Module):
    def __init__(self, num_classes=5, input_channels=12):
        super(ResNet1D, self).__init__()
        
        self.in_channels = 32
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(32, 2, stride=1) # 2 blocks
        self.layer2 = self._make_layer(64, 2, stride=2) # 2 blocks
        self.layer3 = self._make_layer(128, 2, stride=2) # 2 blocks
        self.layer4 = self._make_layer(256, 2, stride=2) # 2 blocks
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
         # x shape: (batch_size, seq_len, distinct_channels) -> need (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ECGTransformer(nn.Module):
    def __init__(self, num_classes=5, input_channels=12, d_model=128, nhead=4, num_layers=3, dim_feedforward=256):
        super(ECGTransformer, self).__init__()
        # Use Conv1d for initial embedding and downsampling (5000 -> 1000)
        self.embedding = nn.Sequential(
            nn.Conv1d(input_channels, d_model, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Channels) -> (Batch, Channels, Seq_Len)
        x = x.transpose(1, 2)
        
        # Embedding + Downsampling
        x = self.embedding(x)
        
        # x shape: (Batch, d_model, Seq_Len_Reduced)
        # Permute for Transformer (Seq_Len, Batch, d_model)
        x = x.permute(2, 0, 1)
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Permute back to (Batch, d_model, Seq_Len)
        x = x.permute(1, 2, 0)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super(InceptionBlock, self).__init__()
        
        self.use_bottleneck = False
        if in_channels > 1 and bottleneck_channels > 0:
            self.use_bottleneck = True
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            in_channels = bottleneck_channels
            
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.use_bottleneck:
            x_in = self.bottleneck(x)
        else:
            x_in = x
            
        outs = [conv(x_in) for conv in self.convs]
        
        pool_out = self.maxpool(x_in)
        pool_out = self.conv_pool(pool_out)
        outs.append(pool_out)
        
        out = torch.cat(outs, dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out

class InceptionTime(nn.Module):
    def __init__(self, num_classes=5, input_channels=12, num_blocks=3, use_residual=True):
        super(InceptionTime, self).__init__()
        
        self.use_residual = use_residual
        self.num_blocks = num_blocks
        self.in_channels = input_channels
        self.out_channels = 32 # Per branch
        
        # 4 branches * 32 = 128 channels output per block
        
        self.blocks = nn.ModuleList()
        self.residuals = nn.ModuleList()
        
        for i in range(num_blocks):
            # Inception Modules inside a residual block
            # For simplicity, we can treat each InceptionBlock as a layer
            # But standard InceptionTime creates a block of 3 InceptionModules + Residual
            
            # Simplified version: Stack InceptionBlocks directly with residuals if dims match
            self.blocks.append(InceptionBlock(
                input_channels if i == 0 else 128, 
                self.out_channels
            ))
            
            if self.use_residual:
                if i % 3 == 2:
                    res_layer = nn.Sequential(
                        nn.Conv1d(input_channels if i == 0 else 128, 128, kernel_size=1, bias=False),
                        nn.BatchNorm1d(128)
                    )
                    self.residuals.append(res_layer)
                else: 
                     # For blocks where we want residuals but channels might mismatch (like block 0)
                     if (input_channels if i==0 else 128) != 128:
                        res_layer = nn.Sequential(
                            nn.Conv1d(input_channels, 128, kernel_size=1, bias=False),
                            nn.BatchNorm1d(128)
                        )
                        self.residuals.append(res_layer)
                     else:
                        self.residuals.append(None) # Check identity later
                    
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (Batch, Seq, Chan) -> (Batch, Chan, Seq)
        x = x.transpose(1, 2)
        
        input_res = x
        
        for i in range(self.num_blocks):
            out = self.blocks[i](input_res)
            
            if self.use_residual:
                res = input_res
                # Check if we have a projection layer for this block
                if self.residuals[i] is not None:
                    res = self.residuals[i](res)
                elif input_res.shape[1] != out.shape[1]:
                     # Should have been handled by init, but safety check
                     # If mismatch and no projection, we can't add. 
                     # But our init logic should cover it.
                     pass 
                
                if res.shape[1] == out.shape[1]:
                    out = out + res
                    out = F.relu(out)
            
            input_res = out
            
        x = self.avgpool(out)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Test model input/output
    model = BaselineCNN1D()
    dummy_input = torch.randn(2, 5000, 12) # (Batch, Seq, Channels)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(output)
