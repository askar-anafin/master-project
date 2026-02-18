
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
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
        out = out + identity
        out = self.relu(out)
        return out

class LearnableAdjacency(nn.Module):
    def __init__(self, num_nodes):
        super(LearnableAdjacency, self).__init__()
        self.A = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.01)
        self.activation = nn.Sigmoid()

    def forward(self):
        return self.activation(self.A)
    
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight) 
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class STReGE(nn.Module):
    def __init__(self, num_classes=5, num_nodes=12, feature_dim=256):
        super(STReGE, self).__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        
        # Temporal Backbone (Shared Weights)
        # Using a simplified ResNet structure
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            ResNetBlock(32, 64, stride=2),
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 256, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Graph Components
        self.learnable_adj = LearnableAdjacency(num_nodes)
        
        # Residual Graph Convolutions
        self.gc1 = GraphConvolution(256, 256)
        self.gc2 = GraphConvolution(256, 256)
        
        self.bn_g1 = nn.BatchNorm1d(num_nodes)
        self.bn_g2 = nn.BatchNorm1d(num_nodes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        
        # x: (Batch, Seq, Nodes)
        x = x.permute(0, 2, 1).contiguous() 
        # (Batch, Nodes, Seq)
        
        # Reshape for Backbone: (Batch*Nodes, 1, Seq)
        x = x.view(batch_size * self.num_nodes, 1, -1)
        
        # Extract Temporal Features
        x_feat = self.backbone(x) # (Batch*Nodes, 256, 1)
        x_feat = x_feat.view(batch_size, self.num_nodes, -1) # (Batch, Nodes, 256)
        
        # Graph Convolutions with Residuals
        adj = self.learnable_adj()
        
        # Block 1
        res = x_feat
        out = self.gc1(x_feat, adj)
        out = self.bn_g1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = out + res # Residual
        
        # Block 2
        res = out
        out = self.gc2(out, adj)
        out = self.bn_g2(out)
        out = self.relu(out)
        out = out + res # Residual
        
        # Pooling
        out = out.mean(dim=1) # (Batch, 256)
        
        out = self.fc(out)
        return out
