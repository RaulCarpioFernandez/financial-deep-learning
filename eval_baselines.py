import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from config import *
from data_loader import load_and_preprocess_data, FEATURE_COLS
from validation import evaluate_regression_performance, evaluate_classification_performance, plot_test_temporal_series
from backtesting import run_economic_backtest
from baselines import *


# ==============================================================================
# MOTOR WALK-FORWARD PURGADO PARA BASELINES (TABULAR Y UNIVARIANTE)
# ==============================================================================
def run_purged_walk_forward_baselines(df, feature_cols, model_type='xgboost', 
                                      arima_order=(1, 0, 1), garch_order=(1,1), refit_interval=1, seed=SEED, verbose=True):
    """
    Ejecuta el Purged Walk-Forward idéntico al de las redes neuronales:
    - XGBoost Regressor/Classifier: Entrena con X tabular (escalado con StandardScaler sólo en Train).
    - ARIMA: Ajuste secuencial / walk-forward 1-step-ahead sobre la serie temporal.
    - Log Regression: 
    """
    X_raw = df[feature_cols].values
    y_raw = df['Target'].values
    total_len = len(df)
    is_classification = (TARGET_TYPE == 'excess_direction')

    all_test_indices = []
    all_test_preds = []
    all_test_reals = []
    fold_metrics = []

    print("\n" + "═" * 78)
    print(f"{f'PURGED WALK-FORWARD BASELINE: {model_type.upper()} ({TARGET_TYPE.upper()})':^78}")
    print(f"{f'({N_SPLITS} Pliegues │ Test: {TEST_SIZE}d │ Purge Gap: {K}d)':^78}")
    print("═" * 78)

    for fold in range(N_SPLITS):
        # Mismos límites temporales que validation.py
        test_end = total_len - (N_SPLITS - 1 - fold) * TEST_SIZE
        test_start = test_end - TEST_SIZE

        val_end = test_start - K
        val_start = val_end - VAL_SIZE

        train_end = val_start - K
        train_start = 0 if WINDOW_TYPE == 'expanding' else train_end - TRAIN_SIZE

        train_dates = f"{df.index[train_start].strftime('%Y-%m')} a {df.index[train_end].strftime('%Y-%m')}"
        test_dates = f"{df.index[test_start].strftime('%Y-%m')} a {df.index[test_end-1].strftime('%Y-%m')}"

        if verbose:
            print(f"\n▶ [Pliegue {fold + 1}/{N_SPLITS}] Train: {train_dates} │ Test: {test_dates}")

        y_train_fold = y_raw[train_start:train_end]
        y_test_fold = y_raw[test_start:test_end]

        # Escalado sin Data Leakage (ajustado únicamente sobre Train)
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_raw[train_start:train_end])
        X_test_fold = scaler.transform(X_raw[test_start:test_end])

        # ----------------------------------------------------
        # 1. LOGISTIC REGRESSION (CLASIFICACIÓN)
        # ----------------------------------------------------
        if model_type.lower() in ('logistic', 'logistic_regression'):
            if not is_classification:
                raise ValueError("Logistic Regression solo está disponible para TARGET_TYPE='excess_direction'.")
            clf = LogisticRegressionBaseline(C=0.1, random_state=seed + fold)
            test_preds_fold = clf.fit_predict_proba(X_train_fold, y_train_fold, X_test_fold)

        # ----------------------------------------------------
        # 2. XGBOOST (CLASIFICACIÓN O REGRESIÓN)
        # ----------------------------------------------------
        elif model_type.lower() in ('xgboost'):
            if is_classification:
                clf = XGBoostClassifierBaseline(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed + fold
                )
                test_preds_fold = clf.fit_predict_proba(X_train_fold, y_train_fold, X_test_fold)
            else:
                reg = XGBoostRegressorBaseline(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed + fold
                )
                test_preds_fold = reg.fit_predict(X_train_fold, y_train_fold, X_test_fold)

        # ----------------------------------------------------
        # 3. ARIMA (REGRESIÓN SOBRE RETORNOS DIARIOS)
        # ----------------------------------------------------
        elif model_type.lower() == 'arima':
            if is_classification:
                raise ValueError("ARIMA solo opera sobre series continuas (TARGET_TYPE='future_return').")
            
            arima_model = ARIMABaseline(order=arima_order, k_steps=K, refit_interval=1)
            test_preds_fold = arima_model.evaluate_walk_forward(
                daily_series=df['Log_Return_1'].values,
                test_start_idx=test_start,
                n_test_steps=len(y_test_fold)
            )

        # ----------------------------------------------------
        # 4. GARCH(p, q) (REGRESIÓN DE VOLATILIDAD DIARIA)
        # ----------------------------------------------------
        elif model_type.lower() == 'garch':
            if TARGET_TYPE != 'volatility':
                raise ValueError("GARCH solo está disponible para TARGET_TYPE='volatility'.")
            p, q = garch_order
            garch_model = GARCHBaseline(p=p, q=q, k_steps=K)
            test_preds_fold = garch_model.evaluate_walk_forward(
                returns_series=df['Log_Return_1'].values,
                test_start_idx=test_start,
                n_test_steps=len(y_test_fold),
                refit_interval=refit_interval
            )

        else:
            raise ValueError(f"Modelo baseline no reconocido: {model_type}")

        # Registro de métricas por pliegue
        if is_classification:
            fold_auc = float(roc_auc_score(y_test_fold, test_preds_fold))
            fold_acc = float(accuracy_score(y_test_fold, (test_preds_fold > 0.5).astype(int)))
            fold_metrics.append({'fold': fold + 1, 'auc': fold_auc, 'acc': fold_acc, 'samples': len(y_test_fold)})
            if verbose:
                print(f"  └─> Test Pliegue {fold + 1}: ROC-AUC = {fold_auc:.4f} │ Acc = {fold_acc*100:.2f}%")
        else:
            rmse_f = float(np.sqrt(np.mean((y_test_fold - test_preds_fold) ** 2)))
            r2_f = float(1.0 - (np.sum((y_test_fold - test_preds_fold) ** 2) / 
                                (np.sum((y_test_fold - np.mean(y_test_fold)) ** 2) + 1e-9)))
            fold_metrics.append({'fold': fold + 1, 'rmse': rmse_f, 'r2': r2_f, 'samples': len(y_test_fold)})
            if verbose:
                print(f"  └─> Test Pliegue {fold + 1}: RMSE = {rmse_f:.5f} │ R² = {r2_f*100:.2f}%")

        all_test_preds.extend(test_preds_fold)
        all_test_reals.extend(y_test_fold)
        all_test_indices.extend(list(df.index[test_start:test_end]))

    df_wf_test = df.loc[all_test_indices].copy()
    wf_preds = np.array(all_test_preds, dtype=np.float64)
    wf_reals = np.array(all_test_reals, dtype=np.int32 if is_classification else np.float64)

    return df_wf_test, wf_preds, wf_reals, fold_metrics


# ==============================================================================
# EVALUACIÓN COMPLETA (ML + BACKTESTING) DE UN BASELINE
# ==============================================================================
def evaluate_baseline(model_name='xgboost'):
    for folder in [FIGURES_DIR, METRICS_DIR]:
        os.makedirs(folder, exist_ok=True)

    df = load_and_preprocess_data()
    is_classification = (TARGET_TYPE == 'excess_direction')

    # 1. Ejecutar Purged Walk-Forward
    df_wf_test, wf_preds, wf_reals, fold_metrics = run_purged_walk_forward_baselines(
        df=df,
        feature_cols=FEATURE_COLS,
        model_type=model_name,
        seed=SEED,
        verbose=True
    )

    # 2. Evaluación Estadística de ML
    if is_classification:
        ml_results = evaluate_classification_performance(
            wf_reals=wf_reals,
            wf_probs=wf_preds,
            fold_metrics=fold_metrics,
            model_name=model_name,
            trainable_params=0,
            plot_curves=True,
            save_results=True
        )
    else:
        ml_results = evaluate_regression_performance(
            wf_reals=wf_reals,
            wf_preds=wf_preds,
            fold_metrics=fold_metrics,
            model_name=model_name,
            trainable_params=0,
            plot_curves=True,
            save_results=True
        )

    zoom_steps = 200
    plot_test_temporal_series(
        dates=df_wf_test.index[-zoom_steps:],
        reals=wf_reals[-zoom_steps:],
        preds=wf_preds[-zoom_steps:],
        model_name=model_name,
        save_results=True
    )

    # 3. Backtest Económico Híbrido
    bt_results, financial_results = run_economic_backtest(
        df_total=df,
        df_test=df_wf_test,
        test_preds=wf_preds,
        cost_bps=COST_BPS,
        risk_aversion=RISK_AVERSION,
        model_name=model_name,
        plot_curves=True,
        save_results=True
    )

    # 4. Guardado consolidado
    full_experiment = {
        'model': model_name.upper(),
        'target': TARGET_TYPE,
        'ml_performance': ml_results,
        'financial_performance': financial_results
    }
    full_json_path = os.path.join(METRICS_DIR, f'{model_name.lower()}_baseline_experiment.json')
    with open(full_json_path, 'w', encoding='utf-8') as f:
        json.dump(full_experiment, f, indent=4, ensure_ascii=False)

    print(f"\n[INFO] Experimento completo de {model_name.upper()} guardado en: {full_json_path}")
    return ml_results, financial_results


if __name__ == '__main__':
    # Para Future Return, ejecutar XGBoost (Regressor) y ARIMA
    if TARGET_TYPE == 'future_return':
        evaluate_baseline(model_name='xgboost')
        evaluate_baseline(model_name='arima')
    # Para Excess Direction, ejecutar XGBoost (Classifier) y Log Regression
    elif TARGET_TYPE == 'excess_direction':
        evaluate_baseline(model_name='xgboost')
        evaluate_baseline(model_name='logistic')
    # Para Volatility, ejecutar XGBoost (Regressor) y GARCH
    elif TARGET_TYPE == 'volatility':
        evaluate_baseline(model_name='xgboost')
        evaluate_baseline(model_name='garch')
    else:
        raise ValueError(f"TARGET_TYPE no válido: {TARGET_TYPE}")
    