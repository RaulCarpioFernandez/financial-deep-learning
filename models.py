import torch
import torch.nn as nn
# ==============================================================================
# DEFINICIÓN DE LAS ARQUITECTURAS DE LAS REDES NEURONALES
# ==============================================================================

# Arquitectura de la red LSTM
class LSTM_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dense_dim=8, output_dim=1, dropout_prob=0.3):
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
        last_hidden = out[:, -1, :]  # Tomamos el último paso temporal
        out = self.dropout(last_hidden)
        out = self.act(self.fc1(out))
        logits = self.fc2(out)
        return logits


# Factory function para instanciar por nombre
def get_model(model_name, input_dim):
    name = model_name.lower()
    if name == 'lstm':
        return LSTM_Classifier(input_dim=input_dim)
    elif name == 'gru':
        return GRU_Classifier(input_dim=input_dim)
    elif name == 'tcn':
        # Aquí conectaremos TCN_Classifier cuando lo implementemos
        raise NotImplementedError("TCN en desarrollo")
    elif name == 'transformer':
        # Aquí conectaremos Transformer_Classifier
        raise NotImplementedError("Transformer en desarrollo")
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")