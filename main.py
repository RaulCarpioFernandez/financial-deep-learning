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


def run_multiseed_grid_search(model_name='lstm', configs=None, seeds=EVAL_SEEDS):
    for folder in [FIGURES_DIR, METRICS_DIR, MODELS_DIR, DATA_DIR]:
        os.makedirs(folder, exist_ok=True)

    print(f"\n[INFO] Cargando y preprocesando datos...")
    df = load_and_preprocess_data()

    if configs is None:
        config_mapping = {
            'lstm': LSTM_CONFIGS,
            'gru': GRU_CONFIGS,
            'tcn': TCN_CONFIGS,
            # 'transformer': TRANSFORMER_CONFIGS,  # Listo para añadir más adelante
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



def evaluate_multiseed_experiment(model_name='lstm', config=None, seeds=EVAL_SEEDS):
    for folder in [FIGURES_DIR, METRICS_DIR, MODELS_DIR, DATA_DIR]:
        os.makedirs(folder, exist_ok=True)

    df = load_and_preprocess_data()
    if config is None:
            config_mapping = {
                'lstm': BEST_LSTM_CONFIG,
                'gru': BEST_GRU_CONFIG,
                'tcn': BEST_TCN_CONFIG,
                # 'transformer': TRANSFORMER_CONFIGS,  # Listo para añadir más adelante
            }
            name = model_name.lower()
            if name not in config_mapping:
                raise ValueError(f"Modelo no reconocido o sin lista de configuraciones definida: '{model_name}'. Opciones disponibles: {list(config_mapping.keys())}")
    
            config = config_mapping[name]

    print("\n" + "═" * 86)
    print(f"{f'EVALUACIÓN MULTI-SEMILLA PARA CONFIGURACIÓN FINAL ({model_name.upper()})':^86}")
    print(f" Configuración: {config}")
    print(f" Semillas evaluadas: {seeds}")
    print("═" * 86)

    ml_records = []
    fin_records = []
    runs_data = []

    for s in seeds:
        print(f"\n▶ Ejecutando Semilla: {s}...")
        set_seed(s)

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
            seed=s,
            verbose=False
        )

        # 1. Métricas de ML (sin plots intermedios ni sobreescritura de archivos)
        ml_res = evaluate_ml_performance(
            wf_reals=wf_reals,
            wf_probs=wf_probs,
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
            test_probs=wf_probs,
            cost_bps=COST_BPS,
            risk_aversion=RISK_AVERSION,
            model_name=model_name,
            plot_curves=False,
            save_results=False
        )

        # Guardar métricas por semilla
        ml_records.append({
            'seed': s,
            'val_auc_mean': mean_val_auc,
            'test_auc_global': ml_res['global_auc'],
            'test_accuracy': ml_res['global_accuracy'],
            'test_balanced_acc': ml_res['global_balanced_accuracy'], 
            'test_f1_macro': ml_res['f1_macro']
        })

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
            'test_auc': ml_res['global_auc'],
            'sharpe': fin_res['Sharpe_Strategy'],
            'df_wf_test': df_wf_test,
            'wf_probs': wf_probs,
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

    # Seleccionar la corrida representativa (más cercana a la media de Sharpe) para graficar
    mean_sharpe = df_consolidated['Sharpe_Strategy'].mean()
    rep_run = min(runs_data, key=lambda r: abs(r['sharpe'] - mean_sharpe))
    rep_seed = rep_run['seed']

    print(f"\n[INFO] Generando gráficos oficiales para la semilla representativa (Seed={rep_seed}, Sharpe={rep_run['sharpe']:.3f})...")
    
    evaluate_ml_performance(
        wf_reals=rep_run['wf_reals'],
        wf_probs=rep_run['wf_probs'],
        fold_metrics=rep_run['fold_metrics'],
        model_name=model_name,
        trainable_params=rep_run['trainable_params'],
        plot_curves=True,
        save_results=True
    )

    run_economic_backtest(
        df_total=df,
        df_test=rep_run['df_wf_test'],
        test_probs=rep_run['wf_probs'],
        cost_bps=COST_BPS,
        risk_aversion=RISK_AVERSION,
        model_name=model_name,
        plot_curves=True,
        save_results=True
    )



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
    # PASO 1: Búsqueda de hiperparámetros (se ejecuta una vez para encontrar la mejor config)
    #run_multiseed_grid_search(model_name='gru', configs=GRU_CONFIGS, seeds=EVAL_SEEDS)

    # PASO 2: Evaluación multiseed rigurosa de la mejor configuración (genera tablas mu ± sigma y gráficos)
    evaluate_multiseed_experiment(model_name='lstm', config=BEST_LSTM_CONFIG, seeds=EVAL_SEEDS)
    evaluate_multiseed_experiment(model_name='gru', config=BEST_GRU_CONFIG, seeds=EVAL_SEEDS)
    evaluate_multiseed_experiment(model_name='tcn', config=BEST_TCN_CONFIG, seeds=EVAL_SEEDS)

    # PASO 3 (Opcional): Prueba unitaria rápida
    # main(model_name='tcn', config=BEST_TCN_CONFIG, seed=SEED)