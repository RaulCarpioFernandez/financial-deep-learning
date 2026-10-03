import os
import json
import random
import torch
import numpy as np
import pandas as pd
from config import *
from data_loader import load_and_preprocess_data, FEATURE_COLS
from validation import (run_purged_walk_forward, 
                        evaluate_classification_performance, 
                        evaluate_regression_performance, 
                        plot_test_temporal_series,
                        print_distributions)
from backtesting import run_economic_backtest
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

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

is_classification = (TARGET_TYPE == 'excess_direction')
def evaluate_model_performance(wf_reals, wf_preds, fold_metrics, model_name, trainable_params, plot_curves=True, save_results=True):
    """Enruta a clasificación o regresión de forma transparente."""
    if is_classification:
        return evaluate_classification_performance(
            wf_reals=wf_reals, wf_probs=wf_preds, fold_metrics=fold_metrics,
            model_name=model_name, trainable_params=trainable_params,
            plot_curves=plot_curves, save_results=save_results
        )
    else:
        return evaluate_regression_performance(
            wf_reals=wf_reals, wf_preds=wf_preds, fold_metrics=fold_metrics,
            model_name=model_name, trainable_params=trainable_params,
            plot_curves=plot_curves, save_results=save_results
        )



def run_multiseed_grid_search(model_name='lstm', seeds=EVAL_SEEDS):
    for folder in [FIGURES_DIR, METRICS_DIR, MODELS_DIR, DATA_DIR]:
        os.makedirs(folder, exist_ok=True)

    print(f"\n[INFO] Cargando y preprocesando datos...")
    df = load_and_preprocess_data()

    config_mapping = {
        'lstm': LSTM_CONFIGS,
        'gru': GRU_CONFIGS,
        'tcn': TCN_CONFIGS,
        'lstm_att': LSTM_ATT_CONFIGS,
        'encoder': ENCODER_CONFIGS,  
        'patchtst': PATCHTST_CONFIGS # Listo para añadir más adelante
    }
    name = model_name.lower()
    if name not in config_mapping:
        raise ValueError(f"Modelo no reconocido o sin lista de configuraciones definida: '{model_name}'. Opciones disponibles: {list(config_mapping.keys())}")

    configs = config_mapping[name]

    tuning_records = []
    cached_runs = {}
    csv_path = os.path.join(METRICS_DIR, f'{model_name.lower()}_multiseed_grid_search.csv')

    total_runs = len(configs) * len(seeds)
    print("\n" + "═" * 84)
    print(f"{f'GRID SEARCH MULTI-SEMILLA ({model_name.upper()} - {TARGET_TYPE.upper()})':^84}")
    print(f"{f'{len(configs)} configuraciones × {len(seeds)} semillas = {total_runs} ejecuciones totales':^84}")
    print("═" * 84)

    for c_idx, cfg in enumerate(configs):
        val_scores = []
        test_scores = []
        cached_runs[c_idx] = {}

        print(f"\n[{c_idx + 1}/{len(configs)}] Probando Cfg: {cfg}")

        for s in seeds:
            set_seed(s)
            # Ejecutar Walk-Forward
            df_wf_test, wf_preds, wf_reals, fold_metrics, trainable_params, mean_val_score = run_purged_walk_forward(
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

            #print(f"  --> Varianza de las predicciones: {np.var(wf_preds):.8f}")
            #print(f"  --> Rango de predicciones: [{np.min(wf_preds):.6f}, {np.max(wf_preds):.6f}]")
            #print(f"  --> Varianza del target real: {np.var(wf_reals):.8f}")

            # Cálculo de métrica de test según la naturaleza del target
            if is_classification:
                test_score = float(roc_auc_score(wf_reals, wf_preds))
            else:
                test_score = float(mean_squared_error(wf_reals, wf_preds))  # Test MSE

            val_scores.append(mean_val_score)
            test_scores.append(test_score)

            # Guardamos los objetos necesarios en memoria para la ejecución final
            cached_runs[c_idx][s] = {
                'test_score': test_score,
                'df_wf_test': df_wf_test,
                'wf_preds': wf_preds,
                'wf_reals': wf_reals,
                'fold_metrics': fold_metrics,
                'trainable_params': trainable_params
            }

        mean_val = float(np.mean(val_scores))
        std_val = float(np.std(val_scores, ddof=1)) if len(seeds) > 1 else 0.0
        mean_test = float(np.mean(test_scores))
        std_test = float(np.std(test_scores, ddof=1)) if len(seeds) > 1 else 0.0

        #print(f"  [Debug] Val MSEs por semilla : {[f'{v:.10f}' for v in val_scores]}")
        #print(f"  [Debug] Test MSEs por semilla: {[f'{t:.10f}' for t in test_scores]}")
        #print(f"  [Debug] Val Std cruda (Python): {std_val:.2e} │ Test Std cruda: {std_test:.2e}")

        tuning_records.append({
            'config_idx': c_idx + 1,
            'config': str(cfg),
            'trainable_params': trainable_params,
            'val_score_mean': mean_val,
            'val_score_std': std_val,
            'test_score_mean': mean_test,
            'test_score_std': std_test
        })

        # Resumen conciso de una sola línea por configuración
        score_name = "Val AUC" if is_classification else "Val MSE"
        test_name = "Test AUC" if is_classification else "Test MSE"
        print(f"  └──> {score_name}: {mean_val:.4f} ± {std_val:.4f} │ {test_name}: {mean_test:.4f} ± {std_test:.4f} │ Params: {trainable_params:,}")

        # Guardado incremental: evita perder datos si la ejecución se para
        pd.DataFrame(tuning_records).to_csv(csv_path, index=False, sep=';', decimal=',', float_format='%.10f')  # Fuerza notación decimal fija (ej: 0.00000002) en lugar de notación 'e')

    # Ordenar resultados: Maximizar si es AUC (False), Minimizar si es MSE (True)
    df_results = pd.DataFrame(tuning_records).sort_values(by='val_score_mean', ascending=(not is_classification))
    df_results.to_csv(csv_path, index=False)
    print(f"\n[INFO] Registro completo del barrido guardado en: {csv_path}")

    # Selección de la mejor configuración (estrictamente por validación)
    best_row = df_results.iloc[0]
    best_c_idx = int(best_row['config_idx']) - 1
    best_cfg = configs[best_c_idx]

    print("\n" + "═" * 84)
    print(f"{'ARQUITECTURA GANADORA SELECCIONADA':^84}")
    print("═" * 84)
    print(f" Modelo               : {model_name.upper()}")
    print(f" Configuración Óptima : {best_cfg}")
    print(f" Parámetros           : {int(best_row['trainable_params']):,}")
    print(f" Score Validación     : {best_row['val_score_mean']:.5f} ± {best_row['val_score_std']:.5f}")
    print("═" * 84)

    # Evaluación exhaustiva (ML + Backtesting) ÚNICAMENTE para la mejor corrida
   # Evaluación completa con la semilla representativa
    best_runs_by_seed = cached_runs[best_c_idx]
    rep_seed = min(seeds, key=lambda s: abs(best_runs_by_seed[s]['test_score'] - best_row['test_score_mean']))
    rep_run = best_runs_by_seed[rep_seed]

    ml_results = evaluate_model_performance(
        wf_reals=rep_run['wf_reals'], wf_preds=rep_run['wf_preds'],
        fold_metrics=rep_run['fold_metrics'], model_name=model_name,
        trainable_params=rep_run['trainable_params'], plot_curves=True, save_results=True
    )

    zoom_steps = 100
    plot_test_temporal_series(
        dates=rep_run['df_wf_test'].index[-zoom_steps:],
        reals=rep_run['wf_reals'][-zoom_steps:],
        preds=rep_run['wf_preds'][-zoom_steps:],
        model_name=model_name,
        save_results=True
    )
    
    backtest_results, financial_results = run_economic_backtest(
        df_total=df, df_test=rep_run['df_wf_test'], test_preds=rep_run['wf_preds'],
        cost_bps=COST_BPS, risk_aversion=RISK_AVERSION, model_name=model_name,
        plot_curves=True, save_results=True
    )

    # 4. Guardar archivo consolidado de la configuración ganadora
    full_experiment = {
        'model': model_name.upper(),
        'target': TARGET_TYPE,
        'best_config': best_cfg,
        'val_score_mean': best_row['val_score_mean'],
        'representative_seed': rep_seed,
        'global_config': get_config_dict(),
        'ml_performance': ml_results,
        'financial_performance': financial_results
    }

    full_json_path = os.path.join(METRICS_DIR, f'{model_name.lower()}_best_multiseed_experiment.json')
    with open(full_json_path, 'w', encoding='utf-8') as f:
        json.dump(full_experiment, f, indent=4, ensure_ascii=False)
        
    print(f"[INFO] Experimento final guardado en: {full_json_path}\n")




def evaluate_multiseed_experiment(model_name='lstm', seeds=EVAL_SEEDS):
    for folder in [FIGURES_DIR, METRICS_DIR, MODELS_DIR, DATA_DIR]:
        os.makedirs(folder, exist_ok=True)

    df = load_and_preprocess_data()

    config_mapping = {
        'lstm': BEST_LSTM_CONFIG,
        'gru': BEST_GRU_CONFIG,
        'tcn': BEST_TCN_CONFIG,
        'encoder': BEST_ENCODER_CONFIG,  
    }
    name = model_name.lower()
    if name not in config_mapping:
        raise ValueError(f"Modelo no reconocido o sin lista de configuraciones definida: '{model_name}'. Opciones disponibles: {list(config_mapping.keys())}")

    config = config_mapping[name]

    print("\n" + "═" * 86)
    print(f"{f'EVALUACIÓN MULTI-SEMILLA FINAL ({model_name.upper()} - {TARGET_TYPE.upper()})':^86}")
    print(f" Configuración: {config}")
    print(f" Semillas evaluadas: {seeds}")
    print("═" * 86)

    ml_records = []
    fin_records = []
    runs_data = []

    for s in seeds:
        print(f"\n▶ Ejecutando Semilla: {s}...")
        set_seed(s)

        df_wf_test, wf_preds, wf_reals, fold_metrics, trainable_params, mean_val_score = run_purged_walk_forward(
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
            seed=s,
            verbose=False
        )

        # 1. Métricas de ML (sin plots intermedios ni sobreescritura de archivos)
        ml_res = evaluate_model_performance(
            wf_reals=wf_reals,
            wf_preds=wf_preds,
            fold_metrics=fold_metrics,
            model_name=model_name,
            trainable_params=trainable_params,
            plot_curves=False,
            save_results=False
        )

        # 2. Backtest financiero (sin plots intermedios)
        bt_df, fin_res = run_economic_backtest(
            df_total=df,
            df_test=df_wf_test,
            test_preds=wf_preds,
            cost_bps=COST_BPS,
            risk_aversion=RISK_AVERSION,
            model_name=model_name,
            plot_curves=False,
            save_results=False
        )

        # Guardar métricas por semilla
        # Registro de métricas según tipo de target
        if is_classification:
            ml_row = {
                'seed': s,
                'val_score': mean_val_score,
                'test_auc': ml_res['global_auc'],
                'test_acc': ml_res['global_accuracy'],
                'test_bal_acc': ml_res['global_balanced_accuracy'],
                'test_f1': ml_res['f1_macro']
            }
        else:
            ml_row = {
                'seed': s,
                'val_score': mean_val_score,
                'rmse': ml_res['rmse'],
                'mae': ml_res['mae'],
                'r2_score': ml_res['r2_score'],
                'spearman_ic': ml_res['spearman_ic'],
                'directional_hit_rate': ml_res['directional_hit_rate']
            }
        ml_records.append(ml_row)

        fin_records.append({
            'seed': s,
            'CAGR_Strategy': fin_res['CAGR_Strategy'],
            'Ann_Return_Strategy': fin_res['Ann_Return_Strategy'], 
            'Ann_Vol_Strategy': fin_res['Ann_Vol_Strategy'],       
            'Sharpe_Strategy': fin_res['Sharpe_Strategy'],
            'MDD_Strategy': fin_res['MDD_Strategy'],
            'Skewness_Strategy': fin_res['Skewness_Strategy'],     
            'Kurtosis_Strategy': fin_res['Kurtosis_Strategy'],     
            'Delta_Util_Strategy': fin_res['Delta_Util_Strategy'],
            'BSS': fin_res['BSS'],
            'R2_OOS': fin_res['R2_OOS'],
            'Turnover_Ann': fin_res['Ann_Turnover'],
            'Total_Costs': fin_res['Total_Costs_Pct']
        })

        runs_data.append({
            'seed': s,
            'score': ml_res['global_auc'] if is_classification else ml_res['rmse'],
            'sharpe': fin_res['Sharpe_Strategy'],
            'df_wf_test': df_wf_test,
            'wf_preds': wf_preds,
            'wf_reals': wf_reals,
            'fold_metrics': fold_metrics,
            'trainable_params': trainable_params
        })

    # DataFrames consolidados
    df_ml = pd.DataFrame(ml_records)
    df_fin = pd.DataFrame(fin_records)
    df_consolidated = pd.merge(df_ml, df_fin, on='seed')

    # Guardar corridas individuales en CSV
    seeds_csv = os.path.join(METRICS_DIR, f'{model_name.lower()}_final_seeds_breakdown.csv')
    df_consolidated.to_csv(seeds_csv, index=False, sep=';', decimal=',')
    print(f"\n[INFO] Desglose por semilla guardado en: {seeds_csv}")

    # Cálculo de medias y desviaciones estándar
    summary_stats = []
    numeric_cols = [col for col in df_consolidated.columns if col != 'seed']
    print("\n" + "═" * 86)
    print(f"{'RESUMEN ESTADÍSTICO FINAL FUERA DE MUESTRA (OOS)':^86}")
    print("═" * 86)
    print(f" {'MÉTRICA':<30} │ {'MEDIA (μ)':>15} │ {'DESV. TÍPICA (σ)':>18} │ {'RANGO [MIN, MAX]':>15}")
    print("─" * 86)

    for col in numeric_cols:
        vals = df_consolidated[col].values
        mu = np.mean(vals)
        sigma = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        min_v, max_v = np.min(vals), np.max(vals)
        summary_stats.append({
            'Metric': col,
            'Mean': round(mu, 4),
            'Std': round(sigma, 4),
            'Min': round(min_v, 4),
            'Max': round(max_v, 4)
        })
        # Formateo porcentual para retornos, volatilidades, drawdowns y accuracy
        is_pct = any(k in col.lower() for k in ['cagr', 'return', 'vol', 'mdd', 'acc', 'costs'])
        fmt = ".2%" if is_pct else ".4f"
        print(f" {col:<30} │ {mu:>15{fmt}} │ {sigma:>18{fmt}} │ [{min_v:{fmt}}, {max_v:{fmt}}]")
    print("═" * 86)

    # Guardar resumen estadístico
    summary_csv = os.path.join(METRICS_DIR, f'{model_name.lower()}_final_config_multiseed_summary.csv')
    pd.DataFrame(summary_stats).to_csv(summary_csv, index=False, sep=';', decimal=',')
    print(f"[INFO] Resumen estadístico guardado en: {summary_csv}")

    # Seleccionar la semilla representativa (más cercana a la media de Sharpe) para graficar
    mean_sharpe = df_consolidated['Sharpe_Strategy'].mean()
    rep_run = min(runs_data, key=lambda r: abs(r['sharpe'] - mean_sharpe))
    rep_seed = rep_run['seed']

    print(f"\n[INFO] Generando gráficos oficiales para la semilla representativa (Seed={rep_seed}, Sharpe={rep_run['sharpe']:.3f})...")
    
    evaluate_model_performance(
        wf_reals=rep_run['wf_reals'],
        wf_preds=rep_run['wf_preds'],
        fold_metrics=rep_run['fold_metrics'],
        model_name=model_name,
        trainable_params=rep_run['trainable_params'],
        plot_curves=True,
        save_results=True
    )
    zoom_steps = 100
    plot_test_temporal_series(
        dates=rep_run['df_wf_test'].index[-zoom_steps:],
        reals=rep_run['wf_reals'][-zoom_steps:],
        preds=rep_run['wf_preds'][-zoom_steps:],
        model_name=model_name,
        save_results=True
    )

    run_economic_backtest(
        df_total=df,
        df_test=rep_run['df_wf_test'],
        test_preds=rep_run['wf_preds'],
        cost_bps=COST_BPS,
        risk_aversion=RISK_AVERSION,
        model_name=model_name,
        plot_curves=True,
        save_results=True
    )




def main(model_name='tcn', seed=SEED):
    for folder in [FIGURES_DIR, METRICS_DIR, MODELS_DIR, DATA_DIR]:
        os.makedirs(folder, exist_ok=True)

    set_seed(seed)
    df = load_and_preprocess_data()
    
    config_mapping = {
    'lstm': BEST_LSTM_CONFIG,
    'gru': BEST_GRU_CONFIG,
    'tcn': BEST_TCN_CONFIG,
    'encoder': BEST_ENCODER_CONFIG,
    }
    name = model_name.lower()
    if name not in config_mapping:
        raise ValueError(f"Modelo no reconocido o sin lista de configuraciones definida: '{model_name}'. Opciones disponibles: {list(config_mapping.keys())}")

    config = config_mapping[name]

    print(f"\n[INFO] Ejecución puntual: {model_name.upper()} | Seed={seed} | Config={config}")
    df_wf_test, wf_preds, wf_reals, fold_metrics, trainable_params, mean_val_score = run_purged_walk_forward(
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

    ml_results = evaluate_model_performance(
        wf_reals=wf_reals, 
        wf_preds=wf_preds, 
        fold_metrics=fold_metrics,
        model_name=model_name,
        trainable_params=trainable_params,
        plot_curves=True,
        save_results=True
    )

    zoom_steps = 100
    plot_test_temporal_series(
        dates=df_wf_test.index[-zoom_steps:],
        reals=wf_reals[-zoom_steps:],
        preds=wf_preds[-zoom_steps:],
        model_name=model_name,
        save_results=True
    )

    backtest_results, financial_results = run_economic_backtest(
        df_total=df, 
        df_test=df_wf_test, 
        test_preds=wf_preds,
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
        'val_auc_score': mean_val_score,
        'ml_performance': ml_results,
        'financial_performance': financial_results
    }

    full_json_path = os.path.join(METRICS_DIR, f'{model_name.lower()}_single_experiment.json')
    with open(full_json_path, 'w', encoding='utf-8') as f:
        json.dump(full_experiment, f, indent=4, ensure_ascii=False)
        
    print(f"[INFO] Registro del experimento guardado en: {full_json_path}\n")




if __name__ == '__main__':
    # PASO 1: Búsqueda de hiperparámetros (se ejecuta una vez para encontrar la mejor config)
    run_multiseed_grid_search(model_name='lstm', seeds=EVAL_SEEDS)
    run_multiseed_grid_search(model_name='gru', seeds=EVAL_SEEDS)
    run_multiseed_grid_search(model_name='tcn', seeds=EVAL_SEEDS)
    run_multiseed_grid_search(model_name='encoder', seeds=EVAL_SEEDS)
    run_multiseed_grid_search(model_name='lstm_att', seeds=EVAL_SEEDS)
    run_multiseed_grid_search(model_name='patchtst', seeds=EVAL_SEEDS)


    # PASO 2: Evaluación multiseed rigurosa de la mejor configuración (genera tablas mu ± sigma y gráficos)
    #evaluate_multiseed_experiment(model_name='lstm', seeds=EVAL_SEEDS)
    #evaluate_multiseed_experiment(model_name='gru', seeds=EVAL_SEEDS)
    #evaluate_multiseed_experiment(model_name='tcn', seeds=EVAL_SEEDS)
    #evaluate_multiseed_experiment(model_name='encoder', seeds=EVAL_SEEDS)

    # PASO 3 (Opcional): Prueba unitaria rápida
    #main(model_name='lstm', seed=SEED)