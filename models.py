import math
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
# ==============================================================================
# DEFINICIÓN DE LAS ARQUITECTURAS DE LAS REDES NEURONALES
# ==============================================================================

# ARQUITECTURA DE LA RED LSTM
class LSTM_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dense_dim=16, output_dim=1, num_layers=1, dropout=0.3):
        super(LSTM_Classifier, self).__init__()
        # PyTorch solo aplica el dropout interno de nn.LSTM si num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=lstm_dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(dense_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.act(self.fc1(out))
        logits = self.fc2(out)
        return logits


# ARQUITECTURA DE LA RED GRU
class GRU_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dense_dim=16, output_dim=1, num_layers=1, dropout=0.3):
        super(GRU_Classifier, self).__init__()
        # PyTorch solo aplica el dropout interno de nn.GRU si num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=lstm_dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(dense_dim, output_dim)

    def forward(self, x):
        # En GRU solo se devuelve out y el estado oculto h_n (no hay c_n)
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]  # Tomamos el estado del último paso temporal
        out = self.dropout(last_hidden)
        out = self.act(self.fc1(out))
        logits = self.fc2(out)
        return logits

    

# ARQUITECTURA DE LA RED TCN
class Chomp1d(nn.Module):
    """Elimina el padding derecho para mantener causalidad y evitar look ahead bias"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

        
# Arquitectura de un bloque residual de la red TCN
class TemporalBlock(nn.Module):
    """
    Residual block de una TCN.

    Dos convoluciones causales dilatadas:
        Conv -> ReLU -> Dropout
        Conv -> ReLU -> Dropout

    + conexión residual.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                            stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, 
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # Convolución 1x1 en caso de que cambie el número de canales
        self.downsample = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels!=out_channels else None

        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)

        return self.final_relu(out + residual)


class TCN_Classifier(nn.Module):
    def __init__(self, input_dim, num_channels=(32, 32, 32), kernel_size=3, dense_dim=16, output_dim=1, dropout=0.2):
        super(TCN_Classifier, self).__init__() 
        layers = []

        for i, out_channels in enumerate(num_channels):
            dilation = 2**i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            layers.append(TemporalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                            stride=1, padding=(kernel_size-1)*dilation, dilation=dilation, dropout=dropout))

        self.network = nn.Sequential(*layers)
        self.fc1 = nn.Linear(num_channels[-1], dense_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_dim, output_dim)

    def forward(self, x):
        # x en PyTorch DataLoader tiene forma: (batch_size, seq_len, num_features)
        # Conv1d espera: (batch_size, num_features, seq_len)
        x = x.transpose(1, 2)

        out = self.network(x)

        # Tomamos el último estado temporal para clasificar (causalidad)
        last_step = out[:, :, -1]
        dense_out = self.relu(self.fc1(last_step))
        logits = self.fc2(dense_out)

        return logits




# ARQUITECTURA DE LA RED ENCODER TRANSFORMER

# Definición del Positional Encoding
class PositionalEncoding(nn.Module):
    """Codificación posicional sinusoidal canónica de Vaswani et al. (2017)."""
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Dimensiones: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x tiene dimensiones: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EncoderTransformer_Classifier(nn.Module):
    """
    Encoder Transformer compacto para series temporales financieras.
    Diseñado para mantener paridad paramétrica (~15k-20k pesos) con LSTM y TCN.
    """
    def __init__(self, input_dim: int, d_model: int = 32, nhead: int = 2, 
                 num_layers: int = 1, dim_feedforward: int = 32, 
                 dense_dim: int = 16, output_dim: int = 1, dropout: float = 0.2):
        super(EncoderTransformer_Classifier, self).__init__()
        
        # Proyección lineal de entrada: (B, L, input_dim) -> (B, L, d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=120, dropout=dropout)
        
        # Capas Encoder del Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Formato: (batch_size, seq_len, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Cabezal de Clasificación (Head)
        self.fc1 = nn.Linear(d_model, dense_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out = self.input_projection(x)
        out = self.pos_encoder(out)
        
        # Modelado de dependencias temporales mediante self-attention
        out = self.transformer_encoder(out)
        
        # Para forecasting, tomamos el último estado temporal t (último paso de la ventana causal)
        last_step = out[:, -1, :]
        
        # Proyección final a logits
        dense = self.act(self.fc1(last_step))
        dense = self.dropout(dense)
        logits = self.fc2(dense)
        
        return logits

    


# Factory function para instanciar por nombre
def get_model(model_name, input_dim, **kwargs):
    name = model_name.lower()
    if name == 'lstm':
        return LSTM_Classifier(input_dim=input_dim, **kwargs)
    elif name == 'gru':
        return GRU_Classifier(input_dim=input_dim, **kwargs)
    elif name == 'tcn':
        return TCN_Classifier(input_dim=input_dim, **kwargs)
    elif name == 'encoder':
        return EncoderTransformer_Classifier(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

def count_parameters(model):
    """Calcula el número total de parámetros entrenables de un nn.Module."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))