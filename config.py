import torch

# Parámetros del Entorno
EVAL_SEEDS = [0, 2, 42, 123, 1111] # Lista de semillas para evaluar robustez estadística fuera de muestra
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TICKER = "^GSPC"  # S&P 500

# Mejor configuración obtenida del Grid Search para cada modelo
BEST_TCN_CONFIG = {
    'num_channels': (32, 32, 32),
    'kernel_size': 3,
    'dense_dim': 16,
    'dropout': 0.2
}
BEST_LSTM_CONFIG = {
    'hidden_dim': 32,
    'dense_dim': 16,
    'dropout': 0.2
}

# Hiperparámetros Temporales y Target
K = 5
BASELINE_WINDOW = 60
SEQUENCE_LENGTH = 20
START_DATE = '2000-01-01'
END_DATE = '2026-05-01'

# Configuración Purged Walk-Forward
N_SPLITS = 4
TRAIN_SIZE = 2016
TEST_SIZE = 504
VAL_SIZE = 126
WINDOW_TYPE = 'expanding'

# Parámetros del Backtest
COST_BPS = 5
RISK_AVERSION = 6.0

# Rutas del Proyecto
RESULTS_DIR = 'results'
FIGURES_DIR = f'{RESULTS_DIR}/figures'
METRICS_DIR = f'{RESULTS_DIR}/metrics'
MODELS_DIR = f'{RESULTS_DIR}/models'
DATA_DIR = 'data'

# Configuraciones para el grid-search
TCN_CONFIGS = [
    # 1. Compactas / Regularizadas (Poco riesgo de sobreajuste)
    {'num_channels': (16, 8), 'kernel_size': 3, 'dense_dim': 8, 'dropout': 0.2}, # Modelo ultraligero
    {'num_channels': (16, 16, 16), 'kernel_size': 3, 'dense_dim': 8, 'dropout': 0.2},
    {'num_channels': (32, 16, 8),   'kernel_size': 3, 'dense_dim': 8, 'dropout': 0.2},  # Pirámide descendente

    # 2. Capacidad Media (Equilibrio)
    {'num_channels': (32, 32, 32), 'kernel_size': 3, 'dense_dim': 16, 'dropout': 0.2},
    {'num_channels': (32, 32, 32), 'kernel_size': 3, 'dense_dim': 16, 'dropout': 0.3},
    {'num_channels': (64, 32, 16), 'kernel_size': 3, 'dense_dim': 16, 'dropout': 0.2},  # Pirámide descendente amplia
    {'num_channels': (32, 32, 32), 'kernel_size': 5, 'dense_dim': 16, 'dropout': 0.3},  # Mayor contexto local

    # 3. Alta Capacidad (Para capturar interacciones complejas)
    {'num_channels': (64, 64, 64), 'kernel_size': 3, 'dense_dim': 16, 'dropout': 0.2},
    {'num_channels': (64, 64, 64), 'kernel_size': 3, 'dense_dim': 32, 'dropout': 0.2},
    {'num_channels': (64, 64, 64), 'kernel_size': 3, 'dense_dim': 32, 'dropout': 0.3},
    {'num_channels': (64, 64, 64), 'kernel_size': 5, 'dense_dim': 32, 'dropout': 0.2},

    # 4. Configuración Profunda (4 bloques d=[1,2,4,8] para memoria larga)
    {'num_channels': (32, 32, 16, 16), 'kernel_size': 3, 'dense_dim': 16, 'dropout': 0.2}
]


LSTM_CONFIGS = [
    {'hidden_dim': 16, 'dense_dim': 8, 'dropout': 0.1},
    {'hidden_dim': 16, 'dense_dim': 8, 'dropout': 0.2},
    {'hidden_dim': 16, 'dense_dim': 8, 'dropout': 0.3},
    {'hidden_dim': 32, 'dense_dim': 8, 'dropout': 0.2},
    {'hidden_dim': 32, 'dense_dim': 8, 'dropout': 0.3},
    {'hidden_dim': 32, 'dense_dim': 16, 'dropout': 0.1},
    {'hidden_dim': 32, 'dense_dim': 16, 'dropout': 0.2},
    {'hidden_dim': 32, 'dense_dim': 16, 'dropout': 0.3},
    {'hidden_dim': 64, 'dense_dim': 16, 'dropout': 0.2},
    {'hidden_dim': 64, 'dense_dim': 16, 'dropout': 0.3},
    {'hidden_dim': 64, 'dense_dim': 32, 'dropout': 0.2},
    {'hidden_dim': 64, 'dense_dim': 32, 'dropout': 0.3}
]


def get_config_dict():
    """Devuelve un diccionario serializable con todos los hiperparámetros."""
    return {
        'SEED': SEED,
        'DEVICE': str(DEVICE),
        'TICKER': TICKER,
        'K': K,
        'START_DATE':START_DATE, 
        'END_DATE':END_DATE, 
        'BASELINE_WINDOW': BASELINE_WINDOW,
        'SEQUENCE_LENGTH': SEQUENCE_LENGTH,
        'N_SPLITS': N_SPLITS,
        'TEST_SIZE': TEST_SIZE,
        'VAL_SIZE': VAL_SIZE,
        'TRAIN_SIZE': TRAIN_SIZE,
        'WINDOW_TYPE': WINDOW_TYPE,
        'COST_BPS': COST_BPS,
        'RISK_AVERSION': RISK_AVERSION
    }