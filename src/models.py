import torch
import torch.nn as nn
import torch.nn.functional as F

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

if __name__ == "__main__":
    # Test model input/output
    model = BaselineCNN1D()
    dummy_input = torch.randn(2, 5000, 12) # (Batch, Seq, Channels)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(output)
