import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
# ==============================================================================
# DEFINICIÓN DE LAS ARQUITECTURAS DE LAS REDES NEURONALES
# ==============================================================================

# Arquitectura de la red LSTM
class LSTM_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dense_dim=16, output_dim=1, dropout_prob=0.3):
        super(LSTM_Classifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout_prob)
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


# Arquitectura de la red GRU
class GRU_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dense_dim=8, output_dim=1, dropout_prob=0.3):
        super(GRU_Classifier, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout_prob)
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
    def __init__(self, input_dim, num_channels=(16, 16), kernel_size=3, dense_dim=8, output_dim=1, dropout=0.3):
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

        

# Factory function para instanciar por nombre
def get_model(model_name, input_dim):
    name = model_name.lower()
    if name == 'lstm':
        return LSTM_Classifier(input_dim=input_dim)
    elif name == 'gru':
        return GRU_Classifier(input_dim=input_dim)
    elif name == 'tcn':
        return TCN_Classifier(input_dim=input_dim)
    elif name == 'transformer':
        # Aquí conectaremos Transformer_Classifier
        raise NotImplementedError("Transformer en desarrollo")
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

def count_parameters(model):
    """Calcula el número total de parámetros entrenables de un nn.Module."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
