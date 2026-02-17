
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models import ResNet1D

class LearnableAdjacency(nn.Module):
    def __init__(self, num_nodes=12):
        super(LearnableAdjacency, self).__init__()
        # Initialize with random small values or identity
        self.A = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.01)
        self.activation = nn.Sigmoid() # Normalize weights between 0 and 1

    def forward(self):
        # Enforce symmetry (optional, depending on assumption of undirected vs directed graph)
        # Using directed graph assumption for now as some leads might drive others
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
        # x shape: (Batch, Nodes, Features)
        # adj shape: (Nodes, Nodes) or (Batch, Nodes, Nodes) if dynamic
        
        # 1. Feature Transformation: H * W
        support = torch.matmul(x, self.weight) 
        
        # 2. Information Propagation: A * (H*W)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class STGNN(nn.Module):
    def __init__(self, num_classes=5, num_nodes=12, feature_dim=256):
        super(STGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        
        # 1. Temporal Feature Extractor
        # We use ResNet1D as a backbone. We need to strip its final fc layer.
        resnet = ResNet1D(num_classes=num_classes, input_channels=1) # 1 channel per lead
        # Remove the final Classification Head (fc) and AvgPool to utilize features
        # ResNet1D structure: ... -> layer4 -> avgpool -> fc
        # We want the output of layer4 
        self.temporal_backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool 
        )
        # Output of backbone (with avgpool) is (Batch, 256, 1) -> (Batch, 256)
        
        # 2. Graph Construction
        self.learnable_adj = LearnableAdjacency(num_nodes)
        
        # 3. Graph Convolutions
        self.gc1 = GraphConvolution(feature_dim, feature_dim)
        self.gc2 = GraphConvolution(feature_dim, feature_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # 4. Final Classification
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Channels) -> We need (Batch, Channels, Seq_Len)
        # But wait, our Dataset outputs (Batch, 5000, 12)
        
        batch_size = x.size(0)
        
        # Reshape to treat each lead independently for temporal extraction
        # Input x: (Batch, Seq, Nodes=12)
        # We process each lead as a separate sample with 1 channel
        
        # 1. Permute to (Batch, Nodes, Seq)
        x = x.permute(0, 2, 1).contiguous() 
        
        # 2. View as (Batch*Nodes, 1, Seq) - This is (N, C, L)
        x = x.view(batch_size * self.num_nodes, 1, -1)
        
        # NO Transpose needed! Sequential backbone expects (N, C, L) directly.
        # x = x.transpose(1, 2)
        
        # Extract Temporal Features
        # Temporal backbone output is (Batch*Nodes, 256)
        x_feat = self.temporal_backbone(x)
        x_feat = x_feat.view(batch_size * self.num_nodes, -1)
        
        # Reshape back to Graph format
        # (Batch, Nodes, Features)
        x_feat = x_feat.view(batch_size, self.num_nodes, -1)
        
        # Get Adjacency Matrix
        adj = self.learnable_adj()
        
        # Graph Convolutions
        h = self.gc1(x_feat, adj)
        h = self.relu(h)
        h = self.dropout(h)
        
        h = self.gc2(h, adj)
        h = self.relu(h)
        
        # Readout / Pooling (Mean over nodes)
        # (Batch, Nodes, Features) -> (Batch, Features)
        h_pool = torch.mean(h, dim=1)
        
        # Classification
        out = self.fc(h_pool)
        
        return out

if __name__ == "__main__":
    model = STGNN()
    print(model)
    dummy = torch.randn(2, 5000, 12)
    out = model(dummy)
    print(f"Output: {out.shape}")
