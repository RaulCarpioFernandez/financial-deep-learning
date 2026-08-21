import torch

# Parámetros del Entorno
SEED = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TICKER = "^GSPC"  # S&P 500

# Parámetros Temporales y Target
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


def get_config_dict():
    """Devuelve un diccionario serializable con todos los hiperparámetros."""
    return {
        'SEED': SEED,
        'DEVICE': str(DEVICE),
        'TICKER': TICKER,
        'K': K,
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