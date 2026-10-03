import math
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

# ==============================================================================
# DEFINICIÓN DE LAS ARQUITECTURAS DE LAS REDES NEURONALES
# ==============================================================================

# ARQUITECTURA DE LA RED LSTM
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dense_dim=16, output_dim=1, num_layers=1, dropout=0.3):
        super(LSTM, self).__init__()
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
        output = self.fc2(out)
        return output


# ARQUITECTURA DE LA RED GRU
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dense_dim=16, output_dim=1, num_layers=1, dropout=0.3):
        super(GRU, self).__init__()
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
        output = self.fc2(out)
        return output

    

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
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                            stride=stride, padding=padding, dilation=dilation)
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


class TCN(nn.Module):
    def __init__(self, input_dim, num_channels=(32, 32, 32), kernel_size=3, dense_dim=16, output_dim=1, dropout=0.2):
        super(TCN, self).__init__() 
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
        output = self.fc2(dense_out)

        return output



# ==============================================================================
# ARQUITECTURA DE LA RED LSTM CON ATENCIÓN TEMPORAL
# ==============================================================================
class TemporalAdditiveAttention(nn.Module):
    """
    Mecanismo de atención aditiva / temporal (Bahdanau-style).
    Calcula una distribución de probabilidad sobre los pasos temporales {1, ..., L}
    para condensar la secuencia en un único vector de contexto ponderado.
    """
    def __init__(self, hidden_dim: int):
        super(TemporalAdditiveAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.act = nn.Tanh()

    def forward(self, rnn_outputs: torch.Tensor):
        # rnn_outputs: (batch_size, seq_len, hidden_dim)
        score = self.v(self.act(self.attn(rnn_outputs)))  # (batch_size, seq_len, 1)
        weights = torch.softmax(score, dim=1)              # (batch_size, seq_len, 1)
        context = torch.sum(weights * rnn_outputs, dim=1)  # (batch_size, hidden_dim)
        return context, weights.squeeze(-1)                # context: (B, hidden_dim), weights: (B, seq_len)


class LSTM_Attention(nn.Module):
    """
    LSTM con mecanismo de atención temporal sobre todos los pasos temporales.
    A diferencia de la LSTM clásica que solo toma out[:, -1, :], este modelo
    pondera dinámicamente qué días de la ventana pasada son más relevantes.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, dense_dim: int = 16, 
                 output_dim: int = 1, num_layers: int = 1, dropout: float = 0.2):
        super(LSTM_Attention, self).__init__()
        
        # PyTorch solo aplica dropout interno en nn.LSTM si num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=lstm_dropout, 
            bidirectional=False
        )
        
        self.attention = TemporalAdditiveAttention(hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Cabeza de predicción (idéntica a TransformerEncoder y PatchTST)
        self.use_dense_head = dense_dim is not None and dense_dim > 0
        if self.use_dense_head:
            self.fc1 = nn.Linear(hidden_dim, dense_dim)
            self.act = nn.LeakyReLU(negative_slope=0.01)
            self.fc2 = nn.Linear(dense_dim, output_dim)
        else:
            self.head = nn.Linear(hidden_dim, output_dim)
            
        # Atributo para almacenar los últimos pesos de atención calculados (interpretabilidad)
        self.attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)  # out: (batch_size, seq_len, hidden_dim)
        
        # Ponderación temporal de todos los pasos {t_1, ..., t_L}
        context, self.attn_weights = self.attention(out)  # context: (B, hidden_dim), weights: (B, L)
        
        out = self.dropout(context)
        
        # Proyección al target (clasificación / regresión)
        if self.use_dense_head:
            dense = self.act(self.fc1(out))
            dense = self.dropout(dense)
            output = self.fc2(dense)
        else:
            output = self.head(out)
            
        return output
    


# Definición del Positional Encoding para los Transformers
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



# ARQUITECTURA DE LA RED ENCODER TRANSFORMER
class TransformerEncoder(nn.Module):
    """
    Encoder Transformer compacto para series temporales financieras.
    Soporta proyección lineal directa o cabeza MLP intermedia.
    """
    def __init__(self, input_dim: int, d_model: int = 32, nhead: int = 2, 
                 num_layers: int = 1, dim_feedforward: int = 32, 
                 dense_dim: int = 16, output_dim: int = 1, dropout: float = 0.2):
        super(TransformerEncoder, self).__init__()
        
        # Proyección lineal de entrada: (B, L, input_dim) -> (B, L, d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=200, dropout=dropout)
        
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
        
        # Cabeza de Clasificación (Head)
        # Si dense_dim es None o <= 0, usamos proyección lineal directa
        self.use_dense_head = dense_dim is not None and dense_dim > 0
        if self.use_dense_head:
            self.fc1 = nn.Linear(d_model, dense_dim)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(dense_dim, output_dim)
        else:
            self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out = self.input_projection(x)
        out = self.pos_encoder(out)
        
        # Modelado de dependencias temporales mediante self-attention
        out = self.transformer_encoder(out)
        
        # Para forecasting, tomamos el último estado temporal t (último paso de la ventana causal)
        last_step = out[:, -1, :]
        
        # Proyección final a logits
        if self.use_dense_head:
            dense = self.activation(self.fc1(last_step))
            dense = self.dropout(dense)
            output = self.fc2(dense)
        else:
            output = self.head(last_step)
        
        return output





# ARQUITECTURA DE LA RED PATCH TST
class PatchTST(nn.Module):
    """
    PatchTST adaptado para clasificación/predicción en series temporales financieras.
    Implementa Patching + Channel Independence + Transformer Encoder + Flatten Head.
    """
    def __init__(self, input_dim: int, seq_len: int = 20, patch_len: int = 8, stride: int = 4, 
                 d_model: int = 32, nhead: int = 2, num_layers: int = 1, dim_feedforward: int = 32, 
                 dense_dim: int = 16, output_dim: int = 1, dropout: float = 0.2):
        super(PatchTST, self).__init__()

        if seq_len < patch_len:
            raise ValueError(
                "sequence_length debe ser >= patch_len"
            )

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # Número exacto de parches a lo largo de la dimensión temporal
        self.num_patches = ((seq_len - patch_len) // stride) + 1
        
        # Proyección y Codificación Posicional a nivel de Patches
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=500, dropout=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Formato: (patch_len, seq_len, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cabeza de predicción (Flatten Head)
        # Si dense_dim es None o <= 0, usamos proyección lineal directa
        flat_dim = input_dim * self.num_patches * d_model
        self.use_dense_head = dense_dim is not None and dense_dim > 0
        if self.use_dense_head:
            self.fc1 = nn.Linear(flat_dim, dense_dim)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(dense_dim, output_dim)
        else:
            self.head = nn.Linear(flat_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape de entrada: (B, L, C)
        B, L, C = x.shape

        if L != self.seq_len:
            raise ValueError(f"Dimensión temporal de entrada ({L}) no coincide con seq_len ({self.seq_len})")
        if C != self.input_dim:
            raise ValueError(f"Número de características de entrada ({C}) no coincide con input_dim ({self.input_dim})")

        # Channel Independence: permutamos a (B, C, L) para operar sobre el tiempo por variable
        x = x.permute(0, 2, 1)

        # Extracción de parches: (B, C, num_patches, patch_len)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Aplanamos Batch y Canales para procesar con Channel Independence: (B * C, num_patches, patch_len)
        patches = patches.contiguous().view(B * C, self.num_patches, self.patch_len)

        # Patch embedding (proyección lineal)
        out = self.patch_projection(patches)

        # Positional encoding
        out = self.pos_encoder(out)
        
        # Modelado temporal de parches con Multi-Head Self-Attention
        out = self.transformer_encoder(out)

        # Reconstrucción de canales y aplanado para la cabeza
        # (B, C, num_patches * d_model) -> (B, C * num_patches * d_model)
        out = out.view(B, C, self.num_patches, self.d_model).flatten(start_dim=1)

        # Proyección final
        if self.use_dense_head:
            dense = self.fc1(out)
            dense = self.activation(dense)
            dense = self.dropout(dense)
            output = self.fc2(dense)
        else:
            output = self.head(out)
        return output

    


# Factory function para instanciar por nombre
def get_model(model_name, input_dim, **kwargs):
    name = model_name.lower()
    if name == 'lstm':
        return LSTM(input_dim=input_dim, **kwargs)
    elif name == 'gru':
        return GRU(input_dim=input_dim, **kwargs)
    elif name == 'tcn':
        return TCN(input_dim=input_dim, **kwargs)
    elif name in ('lstm_att', 'lstm_attention', 'lstm-att'):
        return LSTM_Attention(input_dim=input_dim, **kwargs)
    elif name == 'encoder':
        return TransformerEncoder(input_dim=input_dim, **kwargs)
    elif name == 'patchtst':
            return PatchTST(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

def count_parameters(model):
    """Calcula el número total de parámetros entrenables de un nn.Module."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))