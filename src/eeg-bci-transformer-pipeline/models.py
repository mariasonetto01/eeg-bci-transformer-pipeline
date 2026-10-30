
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
        '''rr
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
        '''
        
        d_k = q.size(-1)

        # k.transpose(-2, -1) scambia le ultime due dimensioni per fare il prodotto matriciale
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        #attn_weights = torch.softmax(scores, dim=-1)

        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
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
        
    def forward(self, x):  #x che do sarebbe x_embed che ho alla fine di s-trans ad esmpio 
        # x shape: (Batch, Seq_Len, d_model)
        batch_size, seq_len, _ = x.size()

        #Linear projections
        q = self.w_q(x) # (Batch, Seq_Len, d_model)
        k = self.w_k(x) # (Batch, Seq_Len, d_model)
        v= self.w_v(x) # (Batch, Seq_Len, d_model)

        #split into heads; (Batch, Seq_Len, d_model) --> (Batch, nhead, Seq_Len, d_k)
        q = q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        # Apply attention
        out, attn_weights = self.attention(q, k, v) #shape (Batch, nhead, Seq_Len, d_k)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # shape(Batch, Seq_Len, d_model)
        
        # FASE 5: Proiezione di output finale
        # out ha forma: (Batch, Seq_Len, d_model)
        output = self.w_out(out)
        
        return output, attn_weights

class FeedForward(nn.Module):
    '''
    '''

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x): #x shape in input: (Batch, Seq_Len, d_model)

        x = self.w_1(x) # Shape (batch, Seq_Len, d_ff)
        x = torch.relu(x) # Shape (batch, Seq_Len, d_ff)
        x = self.dropout(x) # Shape (batch, Seq_Len, d_ff)
        x = self.w_2(x) # Shape (batch, Seq_Len, d_model)

        return x # Shape (batch, Seq_Len, d_model)

class TransformerLayer(nn.Module):
    '''
    '''

    def __init__(self, dropout, d_model=128, nhead=8, d_ff=512):
        super().__init__()
        #self.dropout=dropout

        self.self_attention = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.ffn = FeedForward( d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self,x): # x shape in input: (Batch, Seq_Len, d_model)

        attn_out, attn_weights = self.self_attention(x)
        x = self.norm1(x + self.drop1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out)) #Shape  (Batch, Seq_Len, d_model)

        return x, attn_weights 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialCNN(nn.Module):
    """
    Ramo s-CTrans (Spatial/Temporal CNN Module):
    1. Conv2D (1x16,SAME padding) -> Conv temporale
    2. AvgPool2D (1x32)
    3. Conv2D (1x15, VALID padding)
    """
    def __init__(self, output_channels= 64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels=output_channels, kernel_size=(1, 16), padding="same")
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 32))
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=(1, 15), padding="valid")
        self.relu = nn.ReLU()



    def forward(self, x):
        # x: (batch_size, num_channels, time_points)
       
        x = x.unsqueeze(1) # Unsqueeze to add a channel dimension for Conv2d that need 4 dimensions

        x = self.conv1(x) # Shape (batch_size, out_channels, num_channels, time_points)
        x = self.relu(x) # Shape (batch_size, out_channels, num_channels, time_points)

        x = self.pool1(x) # Shape (batch_size, out_channels, num_channels, (time_points/32))

        x = self.conv2(x) # Shape (batch_size, out_channels, num_channels, (time_points/32)-15)
        x = self.relu(x) # Shape (batch_size, out_channels, num_channels, (time_points/32)-15))




        # Faccio il seguente reshape per avere le dimensioni da poter usare con Pytorch
        x = x.permute(0, 2, 1, 3) # Shape: (batch_size, num_channels, out_channels, reduced_time)
        
        x = x.mean(dim=-1) # Shape finale: (batch_size, num_channels, out_channels) 

        return x # Shape (batch_size, out_channels, num_channels, reduced_time=(time_points/32)-15))



class TemporalCNN(nn.Module): #Da migliorare
    """
         Ramo t-CTrans (Temporal/Spatial CNN Module):
           1. Conv2D (C x 1, SAME padding) -> Conv spaziale tra tutti i canali EEG
           2. AvgPool2D (1 x 8)
           3. Trasposizione features
    """
    def __init__(self , output_channels= 64, num_channels =64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1 , out_channels=output_channels, kernel_size=(num_channels, 1), padding=0) #ho messo padding = 0 perchè altrimenti avrei avuto in posizione 2 64 e non 1 e il squuezee 2 non avrebbe fatto nietne
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 8))
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, num_channels, time_points)
       
        x = x.unsqueeze(1) # Unsqueeze to add a channel dimension for Conv2d that need 4 dimensions

        x = self.conv1(x) # Shape (batch_size, 1, num_channels, time_points)
        x = self.relu(x) # Shape (batch_size, out_channels, num_channels, time_points)

        x = self.pool1(x) # Shape (batch_size, out_channels, num_channels, reduced_time= (time_points/8))

       

        x = x.squeeze(2) # Shape (batch_size, out_channels, reduced_time)

        x = x.permute(0, 2, 1) # (batch_size, reduced_time, out_channels)
      

        return x  # Shape (batch_size, seq_len_t, out_channels)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        
        # Creiamo una matrice di zeri per le posizioni massime
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Fattore di divisione per le frequenze trigonometriche
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Applichiamo il seno alle colonne pari e il coseno alle dispari
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Salviamo 'pe' come buffer (non è un peso da addestrare, ma fa parte dello stato del modello)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x):
        # x arriva con shape: (batch_size, seq_len, d_model)
        # Sommiamo il positional encoding corrispondente alla lunghezza della sequenza attuale (x.size(1))
        x = x + self.pe[:, :x.size(1)]
        return x


class F_CTrans(nn.Module):

    def __init__(self, num_channels,d_model, nhead, d_ff, dropout, num_layers, num_classes):
        super().__init__()

        #spatial

        self.spatial_cnn = SpatialCNN(output_channels = d_model)
        self.spatial_pe = PositionalEncoding(d_model, max_len=num_channels)
        self.spatial_transformer_blocks = nn.ModuleList([
            TransformerLayer(dropout, d_model, nhead, d_ff)
            for _ in range(num_layers)
        ])

        #temporal

        self.temporal_cnn = TemporalCNN(output_channels = d_model, num_channels=num_channels)
        self.temporal_pe = PositionalEncoding(d_model, max_len=5000) 
        self.temporal_transformer_blocks = nn.ModuleList([
            TransformerLayer(dropout, d_model, nhead, d_ff)
            for _ in range(num_layers)
        ])

        # classifier 
        #input_dim = d_model * 2 
        '''
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        '''


        # Classifier after fusion
        self.classifier = nn.Linear(d_model * 2, num_classes) # Concatenate outputs from two streams

   
    def forward(self, x):
        # x: (batch_size, num_channels, time_points)

        # Spatial stream
        x_spatial = self.spatial_cnn(x) # Output: (batch_size, num_channels, model_dim)
        x_spatial = self.spatial_pe(x_spatial)

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

        # Temporal stream
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

        # Concatenate and classify
        fused_output = torch.cat((pooled_spatial, pooled_temporal), dim=1)
        logits = self.classifier(fused_output)

        return logits, (spatial_attention_weights, temporal_attention_weights)



