
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
    def __init__(self, input_channels=1 , output_channels= 64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(1, 16), padding="same")
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
    def __init__(self, input_channels=1 , output_channels= 64, num_channels =64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(num_channels, 1), padding="same")
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 8))
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, num_channels, time_points)
       
        x = x.unsqueeze(1) # Unsqueeze to add a channel dimension for Conv2d that need 4 dimensions

        x = self.conv1(x) # Shape (batch_size, 1, num_channels, time_points)
        x = self.relu(x) # Shape (batch_size, out_channels, num_channels, time_points)

        x = self.pool1(x) # Shape (batch_size, out_channels, num_channels, reduced_time= (time_points/8))

        b, c, h, w = x.shape
        x = x.permute(0, 3, 2, 1).contiguous().view(b, w * h, c) # Shape (batch_size,seq_len_t=reduced_time * num_channels, out_channels )


        return x  # Shape (batch_size, seq_len_t, out_channels)


class StreamCTrans(nn.Module):
    """
    Singolo flusso CTrans (usato sia per s-CTrans che per t-CTrans):
    CNN -> Positional Encoding -> Transformer Encoder Layer
    """
    def __init__(self, cnn_module: nn.Module, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.cnn = cnn_module
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                # [batch, seq_len, d_model]
        x = self.pos_encoder(x)         # Aggiunge informazioni posizionali
        x = self.transformer(x)        # Multi-Head Attention + FeedForward
        x = x.mean(dim=1)              # Global Pooling lungo la dimensione della sequenza
        return x

class FCTransModel(nn.Module):
    """
    Fusion-CNN + Transformer Model (f-CTrans)
    Combina in parallelo i due rami:
    - s-CTrans (Spatial stream)
    - t-CTrans (Temporal stream)
    """
    def __init__(self, num_channels: int, num_classes: int, d_model: int = 64, nhead: int = 4):
        super().__init__()
        
        # Ramo Spaziale
        s_cnn = SpatialCNN(in_channels=num_channels, out_dim=d_model)
        self.s_ctrans = StreamCTrans(cnn_module=s_cnn, d_model=d_model, nhead=nhead)
        
        # Ramo Temporale
        t_cnn = TemporalCNN(num_eeg_channels=num_channels, out_dim=d_model)
        self.t_ctrans = StreamCTrans(cnn_module=t_cnn, d_model=d_model, nhead=nhead)
        
        # Classification Head (Fully Connected Layers)
        # Concatenazione delle feature dei due rami -> 2 * d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: Tensor con forma [Batch_Size, Canali_EEG, Punti_Temporali]
        """
        # Aggiunge dimensione canale immagine (1) per la Conv2D -> [Batch, 1, Canali, Tempo]
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # Elaborazione in parallelo nei due stream
        feat_spatial = self.s_ctrans(x)   # [Batch, d_model]
        feat_temporal = self.t_ctrans(x)  # [Batch, d_model]
        
        # Fusion dei due flussi tramite concatenazione
        fusion_features = torch.cat((feat_spatial, feat_temporal), dim=1) # [Batch, d_model * 2]
        
        # Classificazione finale
        output = self.fc(fusion_features)
        return output