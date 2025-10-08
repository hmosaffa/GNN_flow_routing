"""
LSTM-GCN 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LSTMEncoder(nn.Module):
    """LSTM encoder for processing temporal features with static feature integration."""
    
    def __init__(self, dynamic_input_dim, static_input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(LSTMEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=dynamic_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.combined_layer = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_dynamic, x_static):
        lstm_out, _ = self.lstm(x_dynamic)
        lstm_out = lstm_out[:, -1, :]
        
        static_out = self.static_encoder(x_static)
        
        combined = torch.cat([lstm_out, static_out], dim=1)
        combined = self.combined_layer(combined)
        combined = self.activation(combined)
        combined = self.dropout(combined)
        
        return combined


class SimpleGCNWithStatic(nn.Module):
    """2-layer GCN module for spatial processing with static feature integration."""
    
    def __init__(self, temporal_dim, static_dim, hidden_dim, output_dim, dropout=0.5):
        super(SimpleGCNWithStatic, self).__init__()
        
        self.static_processor = nn.Linear(static_dim, static_dim // 4)
        
        combined_dim = temporal_dim + static_dim // 4
        
        self.conv1 = GCNConv(combined_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.lin = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, temporal_features, static_features, edge_index):
        if static_features.dim() == 3:
            static_features = static_features[0]
        
        processed_static = F.relu(self.static_processor(static_features))
        combined_features = torch.cat([temporal_features, processed_static], dim=1)
        
        x = self.conv1(combined_features, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.lin(x)
        return x.squeeze(-1)


class CombinedLSTMGCNWithStatic(nn.Module):
    """
    Combined LSTM-GCN model for spatiotemporal forecasting.
    
    Architecture:
    1. LSTM Encoder: Processes temporal sequences for each station
    2. Static Feature Integration: Combines temporal embeddings with static features
    3. 2-Layer GCN: Propagates information across spatial network
    
    Args:
        dynamic_input_dim (int): Number of temporal features
        static_input_dim (int): Number of static features
        lstm_hidden_dim (int): Hidden dimension for LSTM
        gnn_hidden_dim (int): Hidden dimension for GCN
        output_dim (int): Output dimension (default: 1)
        lstm_layers (int): Number of LSTM layers (default: 2)
        lstm_dropout (float): Dropout rate for LSTM (default: 0.2)
        gnn_dropout (float): Dropout rate for GCN (default: 0.2)
    """
    
    def __init__(
        self,
        dynamic_input_dim,
        static_input_dim,
        lstm_hidden_dim,
        gnn_hidden_dim,
        output_dim=1,
        lstm_layers=2,
        lstm_dropout=0.2,
        gnn_dropout=0.2
    ):
        super(CombinedLSTMGCNWithStatic, self).__init__()
        
        self.lstm_encoder = LSTMEncoder(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout
        )
        
        self.gnn_module = SimpleGCNWithStatic(
            temporal_dim=lstm_hidden_dim,
            static_dim=static_input_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=output_dim,
            dropout=gnn_dropout
        )
    
    def forward(self, dynamic_features, static_features, edge_index):
        """
        Args:
            dynamic_features: [batch_size, n_stations, seq_len, dynamic_dim]
            static_features: [batch_size, n_stations, static_dim]
            edge_index: [2, num_edges] - graph connectivity
            
        Returns:
            predictions: [batch_size, n_stations, output_dim]
        """
        batch_size = dynamic_features.size(0)
        n_stations = dynamic_features.size(1)
        
        dynamic_feat_reshaped = dynamic_features.view(batch_size * n_stations, *dynamic_features.shape[2:])
        static_feat_reshaped = static_features.view(batch_size * n_stations, -1)
        
        station_embeddings = self.lstm_encoder(dynamic_feat_reshaped, static_feat_reshaped)
        
        station_embeddings = station_embeddings.view(batch_size, n_stations, -1)
        
        if edge_index.dim() > 2:
            edge_index = edge_index[0]
        
        outputs = []
        for i in range(batch_size):
            batch_embedding = station_embeddings[i]
            gnn_output = self.gnn_module(batch_embedding, static_features, edge_index)
            outputs.append(gnn_output)
        
        outputs = torch.stack(outputs)
        
        return outputs
