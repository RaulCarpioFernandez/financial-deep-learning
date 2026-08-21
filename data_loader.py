import os 
import time
import numpy as np
import pandas as pd
import yfinance as yf
from config import TICKER, K, BASELINE_WINDOW, DATA_DIR, START_DATE, END_DATE


# Carpeta donde se guardarán los archivos CSV en local
os.makedirs(DATA_DIR, exist_ok=True)

def get_clean_df(ticker, start=START_DATE, end=END_DATE):
    # Limpiamos el nombre del ticker para el archivo (por ejemplo, ^GSPC -> GSPC.csv)
    safe_name = ticker.replace("^", "")
    file_path = os.path.join(DATA_DIR, f"{safe_name}_{start[:4]}.csv")
    # 1. Si ya existe en disco, se carga directamente
    if os.path.exists(file_path):
        print(f"Cargando {ticker} desde caché local...")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    # 2. Si no existe, se descarga desde Yahoo Finance
    print(f"Descargando {ticker} desde Yahoo Finance por primera vez...")
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end)
    if df.empty:
        raise ValueError(f"No se pudieron descargar datos para el ticker: {ticker}")

    #Eliminamos la información de zona horaria del índice
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Aseguramos que solo sea la fecha (sin horas/minutos)
    df.index = df.index.normalize()
    
    # Guardamos en CSV para que no vuelva a pedirlo nunca más
    df.to_csv(file_path)
    
    # Pausa de cortesía de 2 segundoa para no saturar el servidor de Yahoo
    time.sleep(2)
    
    return df


FEATURE_COLS = [
    # Endógenas
    'Log_Return_1', 'Log_Return_5', 'Log_Return_20', 'Volatility_5', 'Volatility_20',
    'HighLowRange', 'OpenCloseReturn', 'VolumeToSMA20', 'VolumeChange', 'CloseToSMA20', 
    'CloseToSMA60', 'SMA5ToSMA20', 'SMA20ToSMA60', 'RSI_14',
    # Exógenas
    'VIX_Ratio_SMA20', 'VIX_Log_Return_1', 'VIX_Level_Norm', 
    'Yield_Curve_Slope', 'Yield_Slope_Change_5'
]


def load_and_preprocess_data():
    # 1. Datos e Indicadores
    # S&P 500
    instrument = get_clean_df(TICKER)[['Volume', 'Close', 'Open', 'High', 'Low']].copy()

    # Series Exógenas (como Series 1D)
    vix = get_clean_df("^VIX")['Close'].rename('VIX')
    tnx = get_clean_df("^TNX")['Close'].rename('TNX_10Y')
    irx = get_clean_df("^IRX")['Close'].rename('IRX_3M')

    # Combinación limpia
    df = instrument.join([vix, tnx, irx], how='left').ffill()

    # Variables endógenas
    # Retornos logarítmicos
    df['Log_Return_1'] = np.log(df['Close']) - np.log(df['Close'].shift(1))
    df['Log_Return_5'] = np.log(df['Close']) - np.log(df['Close'].shift(5))
    df['Log_Return_20'] = np.log(df['Close']) - np.log(df['Close'].shift(20))

    # Medias Móviles
    SMA_5 = df['Close'].rolling(window=5).mean()
    SMA_20 = df['Close'].rolling(window=20).mean()
    SMA_60 = df['Close'].rolling(window=60).mean()
    df['CloseToSMA20'] = df['Close'] / SMA_20 - 1
    df['CloseToSMA60'] = df['Close'] / SMA_60 - 1
    df['SMA5ToSMA20'] = SMA_5 / SMA_20 - 1
    df['SMA20ToSMA60'] = SMA_20 / SMA_60 - 1

    # Volatilidad
    df['Volatility_5'] = df['Log_Return_1'].rolling(window=5).std()
    df['Volatility_20'] = df['Log_Return_1'].rolling(window=20).std()

    # Ratios de precio y volumen
    df['HighLowRange'] = (df['High'] - df['Low']) / df['Close']
    df['OpenCloseReturn'] = (df['Close'] - df['Open']) / df['Open']

    df['Volume'] = df['Volume'].replace(0, np.nan).ffill().fillna(1)
    vol_sma20 = df['Volume'].rolling(window=20).mean()
    df['VolumeToSMA20'] = df['Volume'] / vol_sma20 - 1
    df['VolumeChange'] = df['Volume'].pct_change()
    
    # RSI 14
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    
    # Variables exógenas
    # Dinámica del VIX (Sentimiento / Pánico)
    vix_sma20 = df['VIX'].rolling(20).mean()
    df['VIX_Ratio_SMA20'] = df['VIX'] / vix_sma20 - 1
    df['VIX_Log_Return_1'] = np.log(df['VIX']) - np.log(df['VIX'].shift(1))
    df['VIX_Level_Norm'] = df['VIX'] / 100.0  # Normalización directa de escala

    # Curva de Tipos (Entorno Macroeconómico)
    df['Yield_Curve_Slope'] = (df['TNX_10Y'] - df['IRX_3M']) / 100.0  # Pendiente 10Y - 3M
    df['Yield_Slope_Change_5'] = df['Yield_Curve_Slope'] - df['Yield_Curve_Slope'].shift(5)
    
    # --- TARGET RELATIVO A LA MEDIANA HISTÓRICA ---
    df['FutureReturn_K'] = np.log(df['Close'].shift(-K) / df['Close'])
    df['Baseline_Return'] = df['Log_Return_5'].rolling(window=BASELINE_WINDOW).median()
    df['Excess_Return_K'] = df['FutureReturn_K'] - df['Baseline_Return']
    df['Target'] = (df['Excess_Return_K'] > 0).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Cálculo del retorno a K días de negociación
    #df['Target'] = (df['FutureReturn_K'] > 0).astype(int)

    return df



