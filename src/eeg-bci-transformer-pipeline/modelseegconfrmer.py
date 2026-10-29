
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

class ConvolutionModule(nn.Module):

    '''
    '''
    def __init__(self, k=40, num_chan = 64):
        super().__init__()

        # Parametri articolo
        #self.conv1 = nn.Conv2d(in_channels=1, out_channels=k, kernel_size=(1, 25), stride=(1, 1))
        #self.conv2 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(num_chan, 1), stride=(1, 1))
        
        #self.bn = nn.BatchNorm2d(k)
        #self.elu = nn.ELU()

        #self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))      

        # Nuovi parametri

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=k, kernel_size=(1, 16), stride=(1, 1)) #cambiato kernel size per prendere finestre di 100 ms 
        self.conv2 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(num_chan, 1), stride=(1, 1))
        
        self.bn = nn.BatchNorm2d(k)
        self.elu = nn.ELU()

        self.pool = nn.AvgPool2d(kernel_size=(1, 48), stride=(1, 10)) #cambiato kernel size per prendere finestre di 300 ms (con il vecchio valore di 75 la finestra era di 468 ms e forse troppo ampia da smorzare il segnale nelle bande mu e beta)   

        self.droputconv = nn.Dropout(0.5)

    def forward(self,x):

        # x Shape: (batch_size, num_channels, time_points)
        
        x = x.unsqueeze(1) # Unsqueeze to add a channel dimension for Conv2d that need 4 dimensions

        x = self.conv1(x) # Shape (batch_size, k, num_channels, Ti=(time_points - kernel size (25)) +1)
        x = self.conv2(x) # Shape (batch_size, k, C=(num_channels - kernel size (66)) + 1, Ti)
        x = self.bn(x) # Shape (batch_size, k, C , Ti)
        x = self.elu(x) # Shape (batch_size, k, C, Ti) 
        x = self.pool(x) # Shape (batch_size, k, C, ((Ti - kernel size (75))/stride (75)) +1)

        return x # Shape (batch_size, k, C, ((Ti - kernel size (75))/stride (75)) +1)

class EEGConformer(nn.Module):
    '''
    '''

    def __init__(self, num_channels, time_steps, d_model, nhead, num_layers, d_ff, dropout, num_classes):
        super().__init__()
    
        self.num_channels = num_channels
        self.time_steps = time_steps
        self.d_model = d_model 
        self.nhead = nhead
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_classes = num_classes

        self.cnn = ConvolutionModule(k=40, num_chan=64)

        # Calcoliamo la lunghezza della sequenza temporale in uscita dalla CNN per dimensionare il Positional Encoding
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, time_steps)
            dummy_out = self.cnn(dummy)
            # dummy_out ha forma: (Batch, k=40, C=1, reduced_time)
            self.seq_len = dummy_out.shape[3]

        self.input_linear = nn.Linear(40, d_model)
        #self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)

        self.encoder_layers = nn.ModuleList([
                TransformerLayer(dropout=dropout, d_model=d_model, nhead=nhead, d_ff=d_ff)
                for _ in range(num_layers)
            ])

        self.classifier = nn.Sequential(
        nn.Linear(d_model, 32),       # Primo layer fully-connected che riduce la dimensionalità
        nn.ReLU(),                    # Attivazione non lineare
        nn.Dropout(dropout),          # Dropout per regolarizzare
        nn.Linear(32, num_classes)    # Secondo layer che sputa fuori i valori (logit) per le M classi
        )

    def forward(self,x) :


        

        x = self.cnn(x) # Shape (batch_size, k, C, ((Ti - kernel size (75))/stride (75)) +1)
        #print("Shape dopo CNN:", x.shape)

        x = x.squeeze(2) # Shape (batch_size, k) reduced_time, si rimuove 1 perchè non serve più
        x = x.permute(0, 2, 1) # Shape (batch_size, reduced_time, k)
        #print("Shape pronta per Transformer:", x.shape)
        
        x = self.input_linear(x)  # Shape: (batch_size, reduced_time, d_model)
        
        #x = x + self.pos_embed[:, :x.size(1), :]

        all_attn_weights = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x) # x mantiene la forma: (batch_size, reduced_time, d_model)
            all_attn_weights.append(attn_weights)
            
        x = x.mean(dim=1) # Ora la forma diventa: (batch_size, d_model)
        
        out = self.classifier(x) # Forma finale: (batch_size, num_classes)
        
        return out, all_attn_weights

import torch
import torch.nn as nn

class EEGTransformerConv1D(nn.Module):
    def __init__(self, num_channels=64, time_steps=641, d_model=32, nhead=4, num_layers=1, d_ff=128, dropout=0.4, num_classes=2):
        super().__init__()
        
        # 1. Estrazione Temporale Locale tramite Conv1D
        # Prende in input i canali EEG come canali della convoluzione (Batch, num_channels, time_steps)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=d_model, kernel_size=32, stride=4, padding=15),
            nn.BatchNorm1d(d_model),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # Calcoliamo dinamicamente la lunghezza della sequenza temporale in uscita dalla Conv1D
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, time_steps)
            dummy_out = self.temporal_conv(dummy) # Shape: (Batch, d_model, seq_len)
            self.seq_len = dummy_out.shape[2]
            
        # 2. Positional Encoding temporale addestrabile
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        
        # 3. Modulo Transformer (sfrutta i tuoi TransformerLayer esistenti)
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(dropout=dropout, d_model=d_model, nhead=nhead, d_ff=d_ff)
            for _ in range(num_layers)
        ])
        
        # 4. Classificatore Finale
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape in ingresso: (Batch, num_channels, time_steps)
        
        # A. Passaggio attraverso la Conv1D -> (Batch, d_model, seq_len)
        x = self.temporal_conv(x)
        
        # B. Riorganizziamo gli assi per il Transformer -> (Batch, seq_len, d_model)
        x = x.transpose(1, 2)
        
        # C. Aggiungiamo il Positional Encoding
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # D. Passaggio attraverso i blocchi Transformer (collezionando i pesi d'attenzione)
        all_attn_weights = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x) # Shape di x: (Batch, seq_len, d_model)
            all_attn_weights.append(attn_weights)
            
        # E. Global Average Pooling lungo la dimensione temporale (seq_len) -> (Batch, d_model)
        x = x.mean(dim=1)
        
        # F. Classificazione finale -> (Batch, num_classes)
        out = self.classifier(x)
        
        return out, all_attn_weights
