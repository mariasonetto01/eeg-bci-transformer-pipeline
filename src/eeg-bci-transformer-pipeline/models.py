
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    '''
    Computes Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
    '''
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        """
        Executes the forward pass for Scaled Dot-Product Attention.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor of shape (Batch, nhead, Seq_Len, d_k).
        k : torch.Tensor
            Key tensor of shape (Batch, nhead, Seq_Len, d_k).
        v : torch.Tensor
            Value tensor of shape (Batch, nhead, Seq_Len, d_k).

        Returns
        -------
        output : torch.Tensor
            Context vectors weighted by attention, shape (Batch, nhead, Seq_Len, d_k).
        attn_weights : torch.Tensor
            Attention weight matrices, shape (Batch, nhead, Seq_Len, Seq_Len).
        """
        
        d_k = q.size(-1)

        # Compute scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply softmax and dropout to obtain attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))

        # Compute the final context vector output
        output = torch.matmul(attn_weights, v)

        return output, attn_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module projecting inputs into multiple subspaces 
    to capture diverse contextual relationships.
    """
    
    def __init__(self, d_model=128, nhead=8, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model deve essere divisibile per nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead  
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_out = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        
    def forward(self, x):  
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (Batch, Seq_Len, d_model).
        """

        # x shape: (Batch, Seq_Len, d_model)
        batch_size, seq_len, _ = x.size()

        #Linear projections for Query, Key, and Value
        q = self.w_q(x)  # Shape : (Batch, Seq_Len, d_model)
        k = self.w_k(x)  # Shape : (Batch, Seq_Len, d_model)
        v= self.w_v(x)   # Shape : (Batch, Seq_Len, d_model)

       # Split dimensions to separate heads: (Batch, Seq_Len, d_model) -> (Batch, nhead, Seq_Len, d_k)
        q = q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention across heads
        out, attn_weights = self.attention(q, k, v) # Shape : (Batch, nhead, Seq_Len, d_k)
        
        # Concatenate heads back together: (Batch, nhead, Seq_Len, d_k) -> (Batch, Seq_Len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # Shape : (Batch, Seq_Len, d_model)
        
        # Final linear output projection
        output = self.w_out(out)
        
        return output, attn_weights

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) composed of two linear transformations with a ReLU activation and dropout in between.
    """

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x): 
        """
        Input shape: (Batch, Seq_Len, d_model)
        """

        x = self.w_1(x) # Shape (batch, Seq_Len, d_ff)
        x = torch.relu(x) # Shape (batch, Seq_Len, d_ff)
        x = self.dropout(x) # Shape (batch, Seq_Len, d_ff)
        x = self.w_2(x) # Shape (batch, Seq_Len, d_model)

        return x # Shape (batch, Seq_Len, d_model)

class TransformerLayer(nn.Module):
    """
    Standard Transformer Encoder Layer combining Multi-Head Self-Attention and a Feed-Forward Network with Residual Connections and Layer Normalization.
    """

    def __init__(self, dropout, d_model=128, nhead=8, d_ff=512):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.ffn = FeedForward( d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self,x):
        """
        Input shape: (Batch, Seq_Len, d_model)
        """

        # Sub-layer 1: Self-Attention + Residual + LayerNorm
        attn_out, attn_weights = self.self_attention(x)
        x = self.norm1(x + self.drop1(attn_out))

        # Sub-layer 2: Feed-Forward + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out))  # x Shape : (Batch, Seq_Len, d_model)

        return x, attn_weights 

class SpatialCNN(nn.Module):
    """
    Spatial CNN module (s-CTrans stream):
    Extracts spatial representations across EEG channels using convolutional layers.
    """
    def __init__(self, output_channels= 64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels=output_channels, kernel_size=(1, 16), padding="same")
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 32))
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=(1, 15), padding="valid")
        self.relu = nn.ReLU()



    def forward(self, x):
        """
        Input x shape: (batch_size, num_channels, time_points)
        """
       # Add a channel dimension for Conv2d (requires 4 dimensions)
        x = x.unsqueeze(1) 

        x = self.conv1(x) # Shape (batch_size, out_channels, num_channels, time_points)
        x = self.relu(x) # Shape (batch_size, out_channels, num_channels, time_points)

        x = self.pool1(x) # Shape (batch_size, out_channels, num_channels, (time_points/32))

        x = self.conv2(x) # Shape (batch_size, out_channels, num_channels, (time_points/32)-15)
        x = self.relu(x) # Shape (batch_size, out_channels, num_channels, (time_points/32)-15))


        # Reshape/permute dimensions to map features into a structure suitable for sequential modeling
        x = x.permute(0, 2, 1, 3) # Shape: (batch_size, num_channels, out_channels, reduced_time)
        x = x.mean(dim=-1) # Average over the time dimension

        return x # Shape (batch_size, out_channels, num_channels, reduced_time=(time_points/32)-15))

class TemporalCNN(nn.Module): 
    """
    Temporal CNN module (t-CTrans stream):
    Extracts temporal representations along the time axis of EEG signals.
    """

    def __init__(self , output_channels= 64, num_channels =64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1 , out_channels=output_channels, kernel_size=(num_channels, 1), padding=0)    
        # padding=0 ensures the spatial dimension collapses to 1, allowing x.squeeze(2) to work properly.
        #(ho messo padding = 0 perchè altrimenti avrei avuto in posizione 2 64 e non 1 e il squezee 2 non avrebbe fatto niente)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 8))
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Input x shape: (batch_size, num_channels, time_points)
        """

        # Add a channel dimension for Conv2d (requires 4 dimensions)
        x = x.unsqueeze(1)
        

        x = self.conv1(x) # Shape (batch_size, 1, num_channels, time_points)
        x = self.relu(x) # Shape (batch_size, out_channels, num_channels, time_points)
        x = self.pool1(x) # Shape (batch_size, out_channels, num_channels, reduced_time= (time_points/8))

        # Squeeze the collapsed spatial dimension and permute for sequence modeling
        x = x.squeeze(2) # Shape (batch_size, out_channels, reduced_time)
        x = x.permute(0, 2, 1) # (batch_size, reduced_time, out_channels)
      

        return x  # Shape (batch_size, seq_len_t, out_channels)

class PositionalEncoding_PE(nn.Module):
    """
    Standard sinusoidal Positional Encoding for temporal sequences.
    """
    
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it is saved with model state but not updated by gradients
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x):
        """
        Input shape: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class LearnableSpatialPositionalEncoding_LPE(nn.Module):
    """
    Learnable Spatial Positional Encoding specifically designed for EEG electrode channels.
    """

    def __init__(self, num_channels, d_model):
        super().__init__()
        self.spatial_embedding = nn.Parameter(torch.randn(1, num_channels, d_model))

    def forward(self, x):
        """
        Input shape: (batch_size, num_channels, d_model)
        """

        return x + self.spatial_embedding[:, :x.size(1), :]

class F_CTrans(nn.Module):
    """
    Fusion CNN + Transformer (f-CTrans) model following the paper
    """

    def __init__(self, num_channels,d_model, nhead, d_ff, dropout, num_layers, num_classes):
        super().__init__()
        
        # Spatial Stream
        self.spatial_cnn = SpatialCNN(output_channels = d_model)
        self.spatial_lpe = LearnableSpatialPositionalEncoding_LPE(num_channels, d_model)
        self.spatial_transformer_blocks = nn.ModuleList([
            TransformerLayer(dropout, d_model, nhead, d_ff)
            for _ in range(num_layers)
        ])
        
        # Temporal Stream
        self.temporal_cnn = TemporalCNN(output_channels = d_model, num_channels=num_channels)
        self.temporal_pe = PositionalEncoding_PE(d_model, max_len=5000) 
        self.temporal_transformer_blocks = nn.ModuleList([
            TransformerLayer(dropout, d_model, nhead, d_ff)
            for _ in range(num_layers)
        ])

        # Classifier after fusion
        self.classifier = nn.Linear(d_model * 2, num_classes) # Concatenate outputs from two streams

   
    def forward(self, x):
        """
        Input x: (batch_size, num_channels, time_points)
        """

        # Spatial Stream Forward
        x_spatial = self.spatial_cnn(x) # Output: (batch_size, num_channels, model_dim)
        x_spatial = self.spatial_lpe(x_spatial)

        spatial_attention_weights = []
        for block in self.spatial_transformer_blocks:
            out = block(x_spatial)
            if isinstance(out, tuple):
                x_spatial, attn = out
                spatial_attention_weights.append(attn)
            else:
                x_spatial = out
                spatial_attention_weights.append(None)

        pooled_spatial = x_spatial.mean(dim=1)

        # Temporal Stream Forward
        x_temporal = self.temporal_cnn(x) # Output: (batch_size, time_points_after_pool, model_dim)
        x_temporal = self.temporal_pe(x_temporal)

        temporal_attention_weights = []
        for block in self.temporal_transformer_blocks:
            out = block(x_temporal)
            if isinstance(out, tuple):
                x_temporal, attn = out
                temporal_attention_weights.append(attn)
            else:
                x_temporal = out
                temporal_attention_weights.append(None)
        pooled_temporal = x_temporal.mean(dim=1)

        # Feature Fusion and Classification
        fused_output = torch.cat((pooled_spatial, pooled_temporal), dim=1)
        logits = self.classifier(fused_output)

        return logits, (spatial_attention_weights, temporal_attention_weights)

class F_CTrans_modified(nn.Module):
    """
    Modified F-CTrans architecture incorporating Mean + Max pooling, LayerNorm pre-fusion, and a deeper MLP classifier for enhanced representational capacity.
    """

    def __init__(self, num_channels, d_model, nhead, d_ff, dropout, num_layers, num_classes):
        super().__init__()

        # Spatial stream
        self.spatial_cnn = SpatialCNN(output_channels=d_model)
        self.spatial_lpe = LearnableSpatialPositionalEncoding_LPE(num_channels, d_model)
        self.spatial_transformer_blocks = nn.ModuleList([
            TransformerLayer(dropout, d_model, nhead, d_ff)
            for _ in range(num_layers)
        ])

        # Temporal stream
        self.temporal_cnn = TemporalCNN(output_channels=d_model, num_channels=num_channels)
        self.temporal_pe = PositionalEncoding_PE(d_model, max_len=5000) 
        self.temporal_transformer_blocks = nn.ModuleList([
            TransformerLayer(dropout, d_model, nhead, d_ff)
            for _ in range(num_layers)
        ])

        # Pre-fusion Layer Normalization to balance numerical scales between branches
        # Each branch produces (d_model * 2) features due to the Mean + Max Pooling combination
        self.norm_spatial = nn.LayerNorm(d_model * 2)
        self.norm_temporal = nn.LayerNorm(d_model * 2)

        # Enhanced MLP Classifier 
        # Total concatenated dimension = (d_model * 2 spatial) + (d_model * 2 temporal) = d_model * 4
        fusion_dim = d_model * 4 

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    

   
    def forward(self, x):
        """
        Input x shape: (batch_size, num_channels, time_points)
        """

        # Spatial Stream Forward
        x_spatial = self.spatial_cnn(x) # Output: (batch_size, num_channels, d_model)
        x_spatial = self.spatial_lpe(x_spatial)

        spatial_attention_weights = []
        for block in self.spatial_transformer_blocks:
            out = block(x_spatial)
            if isinstance(out, tuple):
                x_spatial, attn = out
                spatial_attention_weights.append(attn)
            else:
                x_spatial = out
                spatial_attention_weights.append(None)

        # Spatial Mean + Max Pooling combination
        mean_spatial = x_spatial.mean(dim=1)                            # Shape: (batch_size, d_model)
        max_spatial = x_spatial.max(dim=1)[0]                           # Shape: (batch_size, d_model)
        pooled_spatial = torch.cat([mean_spatial, max_spatial], dim=1)  # Shape: (batch_size, d_model * 2)
        pooled_spatial = self.norm_spatial(pooled_spatial)

        # Temporal Stream Forward
        x_temporal = self.temporal_cnn(x) # Output: (batch_size, reduced_time, d_model)
        x_temporal = self.temporal_pe(x_temporal)

        temporal_attention_weights = []
        for block in self.temporal_transformer_blocks:
            out = block(x_temporal)
            if isinstance(out, tuple):
                x_temporal, attn = out
                temporal_attention_weights.append(attn)
            else:
                x_temporal = out
                temporal_attention_weights.append(None)
                
        # Temporal Mean + Max Pooling combination
        mean_temporal = x_temporal.mean(dim=1)                             # Shape: (batch_size, d_model)
        max_temporal = x_temporal.max(dim=1)[0]                            # Shape: (batch_size, d_model)
        pooled_temporal = torch.cat([mean_temporal, max_temporal], dim=1)  # Shape: (batch_size, d_model * 2)
        pooled_temporal = self.norm_temporal(pooled_temporal)

        #  Feature Fusion and Classification
        fused_output = torch.cat((pooled_spatial, pooled_temporal), dim=1) # Shape: (batch_size, d_model * 4)
        
        # Pass through the enhanced MLP classifier
        logits = self.classifier(fused_output)

        return logits, (spatial_attention_weights, temporal_attention_weights)

