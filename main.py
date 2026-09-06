import os
import json
import random
import torch
import numpy as np
import pandas as pd
from config import *
from data_loader import load_and_preprocess_data, FEATURE_COLS
from validation import run_purged_walk_forward, evaluate_ml_performance
from backtesting import run_economic_backtest
from sklearn.metrics import roc_auc_score, accuracy_score

def set_seed(seed=2):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)


def run_multiseed_grid_search(model_name='tcn', configs=None, seeds=EVAL_SEEDS):
    for folder in [FIGURES_DIR, METRICS_DIR, MODELS_DIR, DATA_DIR]:
        os.makedirs(folder, exist_ok=True)

    print(f"\n[INFO] Cargando y preprocesando datos...")
    df = load_and_preprocess_data()

    if configs is None:
        configs = TCN_CONFIGS if model_name.lower() == 'tcn' else LSTM_CONFIGS

    tuning_records = []
    cached_runs = {}
    csv_path = os.path.join(METRICS_DIR, f'{model_name.lower()}_multiseed_grid_search.csv')

    total_runs = len(configs) * len(seeds)
    print("\n" + "═" * 84)
    print(f"{f'GRID SEARCH MULTI-SEMILLA ({model_name.upper()})':^84}")
    print(f"{f'{len(configs)} configuraciones × {len(seeds)} semillas = {total_runs} ejecuciones totales':^84}")
    print("═" * 84)

    for c_idx, cfg in enumerate(configs):
        val_aucs = []
        test_aucs = []
        cached_runs[c_idx] = {}

        print(f"\n[{c_idx + 1}/{len(configs)}] Probando Cfg: {cfg}")

        for s in seeds:
            set_seed(s)
            
            # Ejecutar Walk-Forward
            df_wf_test, wf_probs, wf_reals, fold_metrics, trainable_params, mean_val_auc = run_purged_walk_forward(
                df=df,
                feature_cols=FEATURE_COLS,
                model_name=model_name,
                model_config=cfg,
                k=K,
                seq_length=SEQUENCE_LENGTH,
                n_splits=N_SPLITS,
                train_size=TRAIN_SIZE,
                test_size=TEST_SIZE,
                val_size=VAL_SIZE,
                window_type=WINDOW_TYPE,
                seed=s
            )

            # Cálculo directo y silencioso de métricas de test
            test_auc = float(roc_auc_score(wf_reals, wf_probs))
            
            val_aucs.append(mean_val_auc)
            test_aucs.append(test_auc)

            # Guardamos los objetos necesarios en memoria para la corrida final
            cached_runs[c_idx][s] = {
                'test_auc': test_auc,
                'df_wf_test': df_wf_test,
                'wf_probs': wf_probs,
                'wf_reals': wf_reals,
                'fold_metrics': fold_metrics,
                'trainable_params': trainable_params
            }

        mean_val = float(np.mean(val_aucs))
        std_val = float(np.std(val_aucs, ddof=1)) if len(seeds) > 1 else 0.0
        mean_test = float(np.mean(test_aucs))
        std_test = float(np.std(test_aucs, ddof=1)) if len(seeds) > 1 else 0.0

        tuning_records.append({
            'config_idx': c_idx + 1,
            'config': str(cfg),
            'trainable_params': trainable_params,
            'val_auc_mean': mean_val,
            'val_auc_std': std_val,
            'test_auc_mean': mean_test,
            'test_auc_std': std_test
        })

        # Resumen conciso de una sola línea por configuración
        print(f"  └──> Val AUC: {mean_val:.4f} ± {std_val:.4f} │ Test AUC: {mean_test:.4f} ± {std_test:.4f} │ Params: {trainable_params:,}")

        # Guardado incremental: evita perder datos si Colab se corta
        pd.DataFrame(tuning_records).to_csv(csv_path, index=False)

    # 1. Ordenar tabla final de resultados por Val AUC
    df_results = pd.DataFrame(tuning_records).sort_values(by='val_auc_mean', ascending=False)
    df_results.to_csv(csv_path, index=False)
    print(f"\n[INFO] Registro completo del barrido guardado en: {csv_path}")

    # 2. Selección de la mejor configuración (estrictamente por validación)
    best_row = df_results.iloc[0]
    best_c_idx = int(best_row['config_idx']) - 1
    best_cfg = configs[best_c_idx]

    print("\n" + "═" * 84)
    print(f"{'ARQUITECTURA GANADORA SELECCIONADA':^84}")
    print("═" * 84)
    print(f" Modelo               : {model_name.upper()}")
    print(f" Configuración Óptima : {best_cfg}")
    print(f" Parámetros           : {int(best_row['trainable_params']):,}")
    print(f" ROC-AUC Validación   : {best_row['val_auc_mean']:.4f} ± {best_row['val_auc_std']:.4f}")
    print(f" ROC-AUC Test OOS     : {best_row['test_auc_mean']:.4f} ± {best_row['test_auc_std']:.4f}")
    print("═" * 84)

    # 3. Evaluación exhaustiva (ML + Backtesting) ÚNICAMENTE para la mejor corrida
    best_runs_by_seed = cached_runs[best_c_idx]
    rep_seed = min(seeds, key=lambda s: abs(best_runs_by_seed[s]['test_auc'] - best_row['test_auc_mean']))
    rep_run = best_runs_by_seed[rep_seed]

    print(f"\n[INFO] Generando informe completo y curvas para la semilla representativa (Seed={rep_seed}, AUC={rep_run['test_auc']:.4f})...")

    ml_results = evaluate_ml_performance(
        wf_reals=rep_run['wf_reals'],
        wf_probs=rep_run['wf_probs'],
        fold_metrics=rep_run['fold_metrics'],
        model_name=model_name,
        trainable_params=rep_run['trainable_params'],
        plot_curves=True,
        save_results=True
    )

    backtest_results, financial_results = run_economic_backtest(
        df_total=df,
        df_test=rep_run['df_wf_test'],
        test_probs=rep_run['wf_probs'],
        cost_bps=COST_BPS,
        risk_aversion=RISK_AVERSION,
        model_name=model_name,
        plot_curves=True,
        save_results=True
    )

    # 4. Guardar archivo consolidado de la configuración ganadora
    full_experiment = {
        'model': model_name.upper(),
        'best_config': best_cfg,
        'val_auc_mean': best_row['val_auc_mean'],
        'val_auc_std': best_row['val_auc_std'],
        'test_auc_mean': best_row['test_auc_mean'],
        'test_auc_std': best_row['test_auc_std'],
        'representative_seed': rep_seed,
        'global_config': get_config_dict(),
        'ml_performance_rep_seed': ml_results,
        'financial_performance_rep_seed': financial_results
    }

    full_json_path = os.path.join(METRICS_DIR, f'{model_name.lower()}_best_multiseed_experiment.json')
    with open(full_json_path, 'w', encoding='utf-8') as f:
        json.dump(full_experiment, f, indent=4, ensure_ascii=False)
        
    print(f"[INFO] Experimento final guardado en: {full_json_path}\n")




def main(model_name='tcn', config=None, seed=SEED):
    for folder in [FIGURES_DIR, METRICS_DIR, MODELS_DIR, DATA_DIR]:
        os.makedirs(folder, exist_ok=True)

    set_seed(seed)
    df = load_and_preprocess_data()
    
    if config is None:
        config = BEST_TCN_CONFIG if model_name.lower() == 'tcn' else BEST_LSTM_CONFIG

    print(f"\n[INFO] Ejecución puntual: {model_name.upper()} | Seed={seed} | Config={config}")
    df_wf_test, wf_probs, wf_reals, fold_metrics, trainable_params, mean_val_auc = run_purged_walk_forward(
        df=df,
        feature_cols=FEATURE_COLS,
        model_name=model_name,
        model_config=config,
        k=K,
        seq_length=SEQUENCE_LENGTH,
        n_splits=N_SPLITS,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        window_type=WINDOW_TYPE,
        seed=seed, 
        verbose=True
    )

    ml_results = evaluate_ml_performance(
        wf_reals=wf_reals, 
        wf_probs=wf_probs, 
        fold_metrics=fold_metrics,
        model_name=model_name,
        trainable_params=trainable_params,
        plot_curves=True,
        save_results=True
    )

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

    full_experiment = {
        'model': model_name.upper(),
        'config': get_config_dict(),
        'model_config': config,
        'val_auc_mean': mean_val_auc,
        'ml_performance': ml_results,
        'financial_performance': financial_results
    }

    full_json_path = os.path.join(METRICS_DIR, f'{model_name.lower()}_single_experiment.json')
    with open(full_json_path, 'w', encoding='utf-8') as f:
        json.dump(full_experiment, f, indent=4, ensure_ascii=False)
        
    print(f"[INFO] Registro del experimento guardado en: {full_json_path}\n")


if __name__ == '__main__':
    # 1. Pipeline completo: Grid Search Multi-Semilla + Test OOS de la mejor
    run_multiseed_grid_search(model_name='tcn', configs=TCN_CONFIGS, seeds=EVAL_SEEDS)

    # 2. Ejecución rápida de una sola configuración y semilla (descomentar para pruebas)
    #main(model_name='tcn', config=BEST_TCN_CONFIG, seed=SEED)