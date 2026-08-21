import os
import json
import torch
import numpy as np
from config import *
from data_loader import load_and_preprocess_data, FEATURE_COLS
from validation import run_purged_walk_forward, evaluate_ml_performance
from backtesting import run_economic_backtest

# Fijar reproducibilidad
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(model_name='lstm'):
    # Crear jerarquía de carpetas
    for folder in [FIGURES_DIR, METRICS_DIR, MODELS_DIR, DATA_DIR]:
        os.makedirs(folder, exist_ok=True)

    # Carga de datos
    print(f"\n[INFO] Cargando y preprocesando datos...")
    df = load_and_preprocess_data()
    
    # Motor Purged Walk-Forward
    print(f"[INFO] Ejecutando Purged Walk-Forward para modelo: {model_name.upper()}...")
    df_wf_test, wf_probs, wf_reals, fold_metrics = run_purged_walk_forward(
        df=df,
        feature_cols=FEATURE_COLS,
        model_name=model_name,
        K=K,
        seq_length=SEQUENCE_LENGTH,
        n_splits=N_SPLITS,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        window_type=WINDOW_TYPE
    )

    # Evaluación de Machine Learning (devuelve diccionario de métricas)
    print(f"[INFO] Evaluando métricas de Machine Learning...")
    ml_results = evaluate_ml_performance(
        wf_reals=wf_reals, 
        wf_probs=wf_probs, 
        fold_metrics=fold_metrics,
        model_name=model_name,
        plot_curves=True,
        save_results=True
    )

    # Backtest Económico (devuelve DataFrame y diccionario financiero)
    print(f"[INFO] Ejecutando backtesting económico multi-benchmark...")
    backtest_results, financial_results = run_economic_backtest(
        df_total=df, 
        df_test=df_wf_test, 
        test_probs=wf_probs,
        cost_bps=COST_BPS, 
        risk_aversion=RISK_AVERSION,
        model_name=model_name, 
        plot_curves=True, 
        save_results=True
    )

    # Guardado Maestro del Experimento Completo en JSON
    full_experiment = {
        'model': model_name.upper(),
        'config': get_config_dict(),
        'ml_performance': ml_results,
        'financial_performance': financial_results
    }

    full_json_path = os.path.join(METRICS_DIR, f'{model_name.lower()}_full_experiment.json')
    with open(full_json_path, 'w', encoding='utf-8') as f:
        json.dump(full_experiment, f, indent=4, ensure_ascii=False)
        
    print(f"\n[INFO] Registro completo del experimento guardado en: {full_json_path}")

if __name__ == '__main__':
    main(model_name='gru')