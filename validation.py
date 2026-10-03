import os
import copy
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models import get_model, count_parameters
from config import TARGET_TYPE, SEED, DEVICE, EPOCHS, BATCH_SIZE, FIGURES_DIR, METRICS_DIR, MODELS_DIR
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score, mean_squared_error, mean_absolute_error, r2_score
)

# Creación de Secuencias Continuas
def create_sequences(X_data, y_data, seq_length=20):
    X, y = [], []
    max_idx = len(X_data) - seq_length + 1
    for i in range(max_idx):
        X.append(X_data[i : (i + seq_length)])
        y.append(y_data[i + seq_length - 1])
    return np.array(X), np.array(y)

# ==============================================================================
# MOTOR DE VALIDACIÓN PURGED WALK-FORWARD
# ==============================================================================
def run_purged_walk_forward(df, feature_cols, model_name='lstm', model_config=None, k=5, seq_length=20, n_splits=4, train_size=2016, test_size=504, val_size=126, window_type='expanding', seed=SEED, verbose=False):
    """
    Ejecuta Purged Walk-Forward CV con Rolling o Expanding Window
    
    Parámetros:
    -----------
    train_size : Número de sesiones fijas de entrenamiento (ej. 2016 = ~8 años, 2520 = ~10 años)
    test_size  : Sesiones por bloque de test ciego (ej. 504 = ~2 años)
    val_size   : Sesiones para Early Stopping (ej. 126 = ~6 meses)
    k          : Purging gap para evitar data leakage
    """
    if model_config is None:
        model_config = {}
        

    if window_type not in ['expanding', 'rolling']:
        raise ValueError(
            "window_type debe ser 'expanding' o 'rolling'"
        )
    X_raw = df[feature_cols].values
    y_raw = df['Target'].values
    total_len = len(df)

    # Verificación de datos mínimos necesarios
    required_len = (n_splits * test_size) + val_size + (2 * k) + (train_size if window_type == 'rolling' else 500)
    if total_len < required_len:
        raise ValueError(
            f"Longitud insuficiente ({total_len} sesiones). Se requieren al menos {required_len} "
            f"para {n_splits} splits con train_size={train_size}, val_size={val_size}, test_size={test_size}."
        )
    
    all_test_indices = []
    all_test_preds = []
    all_test_reals = []
    fold_metrics = []
    val_score_folds = []

    window_header = f"ROLLING WINDOW ({train_size}d)" if window_type == 'rolling' else "EXPANDING WINDOW"

    if verbose:
        print("\n" + "═" * 78)
        print(f"{'INICIANDO PURGED WALK-FORWARD CROSS-VALIDATION':^78}")
        print(f"{f'({window_header} │ {n_splits} Pliegues │ Val: {val_size}d │ Test: {test_size}d │ Gap: {k}d)':^78}")
        print("═" * 78)

    for fold in range(n_splits):
        # Definición de límites temporales del pliegue
        # Pliegue 0 es el más antiguo y Pliegue (n_splits-1) termina en la última fila de df
        test_end = total_len - (n_splits - 1 - fold) * test_size
        test_start = test_end - test_size

        val_end = test_start - k        # Purging Gap 2 entre Val y Test
        val_start = val_end - val_size

        train_end = val_start - k       # Purging Gap 1 entre Train y Val
        train_start = 0 if window_type == 'expanding' else train_end - train_size  # <<--- Ventana deslizante o expansiva

        train_dates = f"{df.index[train_start].strftime('%Y-%m')} a {df.index[train_end].strftime('%Y-%m')}"
        val_dates = f"{df.index[val_start].strftime('%Y-%m')} a {df.index[val_end].strftime('%Y-%m')}"
        test_dates = f"{df.index[test_start].strftime('%Y-%m')} a {df.index[test_end-1].strftime('%Y-%m')}"
        
        if verbose:
            print(f"\n▶ [Pliegue {fold + 1}/{n_splits}] Train: {train_dates} ({train_end - train_start}d) │ Val: {val_dates} ({val_size}d) │ Test: {test_dates} ({test_size}d)")

        # Escalado ajustado EXCLUSIVAMENTE con el Train de este pliegue
        scaler = StandardScaler()
        scaler.fit(X_raw[train_start:train_end])

        # Extracción de histórico continuo para secuencias sin NaNs
        X_train_hist = X_raw[train_start - (seq_length - 1) : train_end] if train_start >= (seq_length - 1) else X_raw[train_start:train_end]
        y_train_hist = y_raw[train_start - (seq_length - 1) : train_end] if train_start >= (seq_length - 1) else y_raw[train_start:train_end]

        X_val_hist = X_raw[val_start - (seq_length - 1) : val_end]
        y_val_hist = y_raw[val_start - (seq_length - 1) : val_end]

        X_test_hist = X_raw[test_start - (seq_length - 1) : test_end]
        y_test_hist = y_raw[test_start - (seq_length - 1) : test_end]

        # Escalamos
        X_train_scaled = scaler.transform(X_train_hist)
        X_val_scaled = scaler.transform(X_val_hist)
        X_test_scaled = scaler.transform(X_test_hist)

        # Generamos las secuencias
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_hist, seq_length=seq_length)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_hist, seq_length=seq_length)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_hist, seq_length=seq_length)

        # Tensores y DataLoader
        X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
        X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(1)
        X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
        y_test_t = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(1)

        g = torch.Generator()
        g.manual_seed(seed + fold)  # Semilla consistente por pliegue

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True, generator=g)

        # Modelo y Optimización del Pliegue
        model = get_model(model_name, input_dim=len(feature_cols), **model_config).to(DEVICE)
        trainable_params = count_parameters(model)

        # Configuración de Pérdida y Métrica de Early Stopping en función del Target
        is_classification = (TARGET_TYPE == 'excess_direction')

        if is_classification:
            num_pos = np.sum(y_raw[train_start:train_end] == 1)
            num_neg = np.sum(y_raw[train_start:train_end] == 0)
            pos_weight = torch.tensor([num_neg / (num_pos + 1e-9)], dtype=torch.float32).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            scheduler_mode = 'max'
            best_val_score = -float('inf')
        else:
            criterion = nn.HuberLoss(delta=1.0)
            scheduler_mode = 'min'
            best_val_score = float('inf')


        #Definimos el optimizador y el scheduler del learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.5, patience=3)

        # Entrenamiento con Early Stopping por ROC-AUC
        patience = 6
        patience_counter = 0
        best_model_weights = None

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t.to(DEVICE))
                val_loss = criterion(val_logits, y_val_t.to(DEVICE)).item()
                y_val_np = y_val_t.detach().cpu().numpy().ravel()

            if is_classification:
                val_probs = torch.sigmoid(val_logits).detach().cpu().numpy().ravel()
                val_score = roc_auc_score(y_val_np, val_probs)
                improved = val_score > best_val_score
            else:
                val_preds = val_logits.detach().cpu().numpy().ravel()
                val_score = mean_squared_error(y_val_np, val_preds)
                improved = val_score < best_val_score

            scheduler.step(val_score)

            if verbose and ((epoch + 1) % 2 == 0 or epoch == 0):
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Score: {val_score:.4f}")

            if improved:
                best_val_score = val_score
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping en época {epoch+1}")
                    break
            
        val_score_folds.append(best_val_score)
        model.load_state_dict(best_model_weights)
        
        # Guardar pesos en /results/models
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(best_model_weights, os.path.join(MODELS_DIR, f'{model_name.lower()}_fold_{fold+1}.pt'))

        # EVALUACIÓN EN TEST (fuera de la muestra)
        model.eval()
        with torch.no_grad():
            test_out = model(X_test_t.to(DEVICE))
            y_test_real = y_test_t.detach().cpu().numpy().ravel()

        if is_classification:
            test_probs = torch.sigmoid(test_out).detach().cpu().numpy().ravel()
            roc_auc = roc_auc_score(y_test_real, test_probs)

            decision_threshold = 0.5
            test_predictions = (test_probs > decision_threshold).astype(int)
            acc = accuracy_score(y_test_real, test_predictions)
            balanced_acc = balanced_accuracy_score(y_test_real, test_predictions)

            fold_metrics.append({'fold': fold + 1, 'auc': roc_auc, 'acc': acc, 'bal_acc': balanced_acc, 'samples': len(y_test_real)})
            all_test_preds.extend(test_probs)

            if verbose:
                print(f"  └─> Test Ciego Pliegue {fold + 1}: ROC-AUC = {roc_auc:.4f} │ Balanced Accuracy = {balanced_acc*100:.2f}%")
        else:
            # Regresión continua: la salida de model() son las predicciones continuas directas
            test_preds = test_out.detach().cpu().numpy().ravel()
            fold_rmse = np.sqrt(mean_squared_error(y_test_real, test_preds))
            fold_r2 = r2_score(y_test_real, test_preds)

            fold_metrics.append({'fold': fold + 1, 'rmse': fold_rmse, 'r2': fold_r2, 'samples': len(y_test_real)})
            all_test_preds.extend(test_preds)  # Almacena las predicciones continuas

            if verbose:
                print(f"  └─> Test Ciego Pliegue {fold + 1}: RMSE = {fold_rmse:.4f} │ R^2 = {fold_r2*100:.2f}%")
            
        
        all_test_indices.extend(list(df.index[test_start:test_end]))
        all_test_reals.extend(y_test_real)

    # Consolidación Global
    mean_val_score = float(np.mean(val_score_folds))
    df_wf_test = df.loc[all_test_indices].copy()
    wf_preds = np.array(all_test_preds, dtype=np.float64)
    wf_reals = np.array(all_test_reals, dtype=np.float64 if not is_classification else np.int32)

    if verbose:
        score_label = "ROC-AUC" if is_classification else "MSE"
        print(f"\nValidation {score_label} medio: {mean_val_score:.4f}")

    return df_wf_test, wf_preds, wf_reals, fold_metrics, trainable_params, mean_val_score


def evaluate_classification_performance(wf_reals, wf_probs, fold_metrics, model_name='lstm', trainable_params=0, plot_curves=True, save_results=True):
    """
    Calcula, imprime y grafica el rendimiento global fuera de muestra del modelo de ML.
    """

    if save_results:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)

    decision_threshold = 0.5 # se puede calcular como np.median(wf_probs) 
    global_predictions = (wf_probs > decision_threshold).astype(int)
    global_acc = accuracy_score(wf_reals, global_predictions)
    global_balanced_acc = balanced_accuracy_score(wf_reals, global_predictions)
    global_auc = roc_auc_score(wf_reals, wf_probs)
    mean_fold_auc = float(np.mean([m['auc'] for m in fold_metrics]))
    report = classification_report(wf_reals, global_predictions, target_names=['Baja/Lateral (0)', 'Sube (1)'], output_dict=True, zero_division=0)

    # Extracción directa y segura por nombre
    r0 = report['Baja/Lateral (0)']
    r1 = report['Sube (1)']
    r_macro = report['macro avg']

    # Informe en Consola
    print("\n" + "═" * 78)
    print(f"{'EVALUACIÓN GLOBAL DE MACHINE LEARNING (CONCATENACIÓN WALK-FORWARD)':^78}")
    print("═" * 78)
    print(f" Parámetros Entrenables      : {trainable_params:,} pesos")
    print(f" Muestras Totales Acumuladas : {len(wf_reals)} sesiones | Umbral de Decisión (Calibrado): {decision_threshold:.2f}")
    print(f" ROC-AUC Global              : {global_auc:.4f} (Promedio entre pliegues: {np.mean([m['auc'] for m in fold_metrics]):.4f})")
    print(f" Accuracy / Balanced Acc     : {global_acc*100:.2f}% / {global_balanced_acc*100:.2f}%")
    print("─" * 78)
    print(f" {'Clase':<22} │ {'Precision':>10} │ {'Recall':>10} │ {'F1-Score':>10} │ {'Soporte':>10}")
    print("─" * 78)
    print(f" {'Baja / Lateral (0)':<22} │ {r0['precision']:>10.3f} │ {r0['recall']:>10.3f} │ {r0['f1-score']:>10.3f} │ {int(r0['support']):>10d}")
    print(f" {'Sube (1)':<22} │ {r1['precision']:>10.3f} │ {r1['recall']:>10.3f} │ {r1['f1-score']:>10.3f} │ {int(r1['support']):>10d}")
    print("─" * 78)
    print(f" {'Macro Promedio':<22} │ {r_macro['precision']:>10.3f} │ {r_macro['recall']:>10.3f} │ {r_macro['f1-score']:>10.3f} │ {len(wf_reals):>10d}")
    print("═" * 78)

    if plot_curves:
        # Curvas ROC y Precision-Recall Globales 
        fpr, tpr, _ = roc_curve(wf_reals, wf_probs)
        roc_auc_curve = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(wf_reals, wf_probs)
        avg_precision = average_precision_score(wf_reals, wf_probs)
        baseline = np.mean(wf_reals) # Proporción de la clase positiva (baseline)

        # Graficar ambas curvas
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Gráfico ROC
        axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_curve:.3f})')
        axes[0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Aleatorio (AUC = 0.50)')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate (FPR)')
        axes[0].set_ylabel('True Positive Rate (TPR)')
        axes[0].set_title('Curva ROC')
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)

        # Gráfico Precision-Recall
        axes[1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        axes[1].axhline(y=baseline, color='navy', lw=1.5, linestyle='--', label=f'Baseline ({baseline:.2f})')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Curva Precision-Recall')
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        if save_results:
            fig_path = os.path.join(FIGURES_DIR, f'{model_name.lower()}_roc_pr_curves.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Gráficos de ML guardados en: {fig_path}")

        plt.tight_layout()
        plt.show()

    # Empaquetado y guardado estructurado de métricas
    ml_results = {
        'model': model_name.upper(),
        'trainable_params': trainable_params,
        'optimal_threshold': decision_threshold,
        'global_auc': global_auc,
        'mean_fold_auc': mean_fold_auc,
        'global_accuracy': global_acc,
        'global_balanced_accuracy': global_balanced_acc,
        'f1_macro': r_macro['f1-score'],
        'fold_metrics': fold_metrics,
        'classification_report': report
    }

    if save_results:
        # Guardar en JSON detallado
        json_path = os.path.join(METRICS_DIR, f'{model_name.lower()}_ml_metrics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ml_results, f, indent=4, ensure_ascii=False)
            
        # Guardar resumen tabular en CSV
        df_summary = pd.DataFrame([{
            'Model': model_name.upper(),
            'Trainable_Params': trainable_params,
            'Global_AUC': global_auc,
            'Mean_Fold_AUC': mean_fold_auc,
            'Accuracy': global_acc,
            'Balanced_Acc': global_balanced_acc,
            'Precision_Macro': r_macro['precision'],
            'Recall_Macro': r_macro['recall'],
            'F1_Macro': r_macro['f1-score']
        }])
        csv_path = os.path.join(METRICS_DIR, f'{model_name.lower()}_ml_summary.csv')
        df_summary.to_csv(csv_path, index=False)
        print(f"[INFO] Métricas de ML guardadas en: {json_path} y {csv_path}")

    return ml_results




def evaluate_regression_performance(wf_reals, wf_preds, fold_metrics, model_name='lstm', trainable_params=0, plot_curves=True, save_results=True):
    """
    Calcula, imprime y grafica el rendimiento global de regresión fuera de muestra.
    
    Métricas calculadas:
    --------------------
    - RMSE: Raíz del error cuadrático medio.
    - MAE: Error absoluto medio.
    - R² OOS: Coeficiente de determinación fuera de muestra.
    - Pearson Corr (r): Correlación lineal entre predicciones y valores reales.
    - Spearman Corr (IC - Information Coefficient): Correlación de rangos (clave en carteras cuantitativas).
    - Hit Rate direccional: Porcentaje de acierto en el signo (signo predicho vs signo real).
    """

    if save_results:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)

    wf_reals = np.array(wf_reals, dtype=np.float64)
    wf_preds = np.array(wf_preds, dtype=np.float64)
    
    # Métricas de Error y Bondad de Ajuste
    mse = mean_squared_error(wf_reals, wf_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(wf_reals, wf_preds)
    r2 = r2_score(wf_reals, wf_preds)

    # Correlaciones (Information Coefficient)
    pearson_corr, p_val_pearson = pearsonr(wf_preds, wf_reals)
    spearman_ic, p_val_spearman = spearmanr(wf_preds, wf_reals)

    # Hit Rate Direccional Implícito (Sign Match)
    directional_hit_rate = np.mean(np.sign(wf_preds) == np.sign(wf_reals))

    # Promedio entre pliegues individuales
    mean_fold_rmse = float(np.mean([m['rmse'] for m in fold_metrics])) if fold_metrics and 'rmse' in fold_metrics[0] else rmse
    mean_fold_r2 = float(np.mean([m['r2'] for m in fold_metrics])) if fold_metrics and 'r2' in fold_metrics[0] else r2

    # Informe en Consola
    print("\n" + "═" * 78)
    print(f"{'EVALUACIÓN GLOBAL DE REGRESIÓN (CONCATENACIÓN WALK-FORWARD)':^78}")
    print("═" * 78)
    print(f" Parámetros Entrenables      : {trainable_params:,} pesos")
    print(f" Muestras Totales Acumuladas : {len(wf_reals)} sesiones")
    print("─" * 78)
    print(f" {'MÉTRICA':<30} │ {'VALOR GLOBAL':>18} │ {'PROMEDIO FOLDS':>18}")
    print("─" * 78)
    print(f" {'RMSE (Root Mean Sq Error)':<30} │ {rmse:>18.5f} │ {mean_fold_rmse:>18.5f}")
    print(f" {'MAE (Mean Absolute Error)':<30} │ {mae:>18.5f} │ {'N/A':>18}")
    print(f" {'R² Fuera de Muestra (OOS)':<30} │ {r2:>18.5f} │ {mean_fold_r2:>18.5f}")
    print(f" {'Pearson Correlation (r)':<30} │ {pearson_corr:>18.4f} (p={p_val_pearson:.1e}) │ {'N/A':>18}")
    print(f" {'Spearman IC (Rank Corr)':<30} │ {spearman_ic:>18.4f} (p={p_val_spearman:.1e}) │ {'N/A':>18}")
    #print(f" {'Directional Hit Rate (Sign)':<30} │ {directional_hit_rate*100:>17.2f}% │ {'N/A':>18}")
    print("═" * 78)

    # Gráficas Diagnósticas de Regresión
    if plot_curves:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Dispersión Predicción vs Real con recta de identidad y tendencia
        axes[0].scatter(wf_reals, wf_preds, alpha=0.35, color='tab:blue', edgecolors='none', s=20)
        # Línea de 45 grados (predicción perfecta)
        min_val = min(np.min(wf_reals), np.min(wf_preds))
        max_val = max(np.max(wf_reals), np.max(wf_preds))
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Ideal (y = x)')
        
        # Ajuste lineal empírico
        if np.std(wf_preds) > 1e-8:
            m_fit, b_fit = np.polyfit(wf_reals, wf_preds, 1)
            axes[0].plot(wf_reals, m_fit * wf_reals + b_fit, 'k-', lw=1.2, label=f'Ajuste (Pendiente={m_fit:.2f})')

        axes[0].set_title(f'Real vs Predicho (Spearman IC = {spearman_ic:.3f})', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Valor Real')
        axes[0].set_ylabel('Predicción del Modelo')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, linestyle=':', alpha=0.5)

        # Histograma de Residuos (Errores e = y - ŷ)
        residuals = wf_reals - wf_preds
        axes[1].hist(residuals, bins=50, color='tab:purple', edgecolor='black', alpha=0.7, density=True)
        axes[1].axvline(0, color='red', linestyle='--', lw=1.5, label=f'Media: {np.mean(residuals):.5f}')
        axes[1].set_title(f'Distribución de Residuos (RMSE = {rmse:.5f})', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Residuo (Real - Predicho)')
        axes[1].set_ylabel('Densidad')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()

        if save_results:
            fig_path = os.path.join(FIGURES_DIR, f'{model_name.lower()}_regression_diagnostics.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Gráficos de regresión guardados en: {fig_path}")

        plt.show()

    # Empaquetado de Resultados
    ml_results = {
        'model': model_name.upper(),
        'trainable_params': trainable_params,
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2),
        'pearson_corr': float(pearson_corr),
        'spearman_ic': float(spearman_ic),
        'directional_hit_rate': float(directional_hit_rate),
        'mean_fold_rmse': mean_fold_rmse,
        'mean_fold_r2': mean_fold_r2,
        'fold_metrics': fold_metrics
    }

    if save_results:
        # Guardar JSON completo
        json_path = os.path.join(METRICS_DIR, f'{model_name.lower()}_regression_metrics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ml_results, f, indent=4, ensure_ascii=False)
            
        # Guardar CSV tabular resumen
        df_summary = pd.DataFrame([{
            'Model': model_name.upper(),
            'Trainable_Params': trainable_params,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2,
            'Pearson_Corr': pearson_corr,
            'Spearman_IC': spearman_ic,
            'Directional_Hit_Rate': directional_hit_rate
        }])
        csv_path = os.path.join(METRICS_DIR, f'{model_name.lower()}_regression_summary.csv')
        df_summary.to_csv(csv_path, index=False)
        print(f"[INFO] Métricas de regresión guardadas en: {json_path} y {csv_path}")

    return ml_results



def plot_test_temporal_series(dates, reals, preds, model_name='lstm', save_results=True):
    """
    Genera un gráfico cronológico continuo del Target Real vs la Predicción
    a lo largo de todo el periodo fuera de muestra (Test acumulado del Walk-Forward).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    is_class = (TARGET_TYPE == 'excess_direction')
    reals = np.asarray(reals)
    preds = np.asarray(preds)

    if is_class:
        # Probabilidades predichas frente a clases reales
        ax1.scatter(dates, reals, color='tab:gray', alpha=0.35, s=15, label='Etiqueta Real {0, 1}')
        ax1.plot(dates, preds, color='tab:blue', lw=1.2, label=f'Probabilidad Predicha {model_name.upper()}')
        ax1.axhline(0.5, color='red', linestyle='--', lw=1, alpha=0.7, label='Umbral de Decisión (0.5)')
        ax1.set_ylabel('Probabilidad / Clase')
        ax1.set_ylim(-0.05, 1.05)
    else:
        # Regresión continua (retorno acumulado o volatilidad)
        ax1.plot(dates, reals, color='black', lw=1.3, label='Target Real ($y_t$)', alpha=0.85)
        ax1.plot(dates, preds, color='tab:blue', lw=1.3, label=f'Predicción {model_name.upper()} ($\hat{{y}}_t$)', alpha=0.85)
        ax1.set_ylabel(f'Valor ({TARGET_TYPE})')

    ax1.set_title(f'Seguimiento Temporal Fuera de Muestra (Test Walk-Forward): {model_name.upper()}', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Panel inferior: Residuales o Divergencia
    if is_class:
        residuals = preds - reals  # Error de calibración puntual
        ax2.plot(dates, residuals, color='tab:purple', lw=1.0, label='Error de Probabilidad ($\hat{p} - y$)')
        ax2.axhline(0, color='black', linestyle=':', lw=1)
        ax2.set_ylabel('Error')
        ax2.set_ylim(-1.05, 1.05)
    else:
        residuals = reals - preds
        ax2.plot(dates, residuals, color='tab:red', lw=1.0, label='Residuo ($y_t - \hat{y}_t$)', alpha=0.75)
        ax2.axhline(0, color='black', linestyle=':', lw=1)
        ax2.set_ylabel('Residuo')

    ax2.set_xlabel('Fecha')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()

    if save_results:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig_path = os.path.join(FIGURES_DIR, f'{model_name.lower()}_temporal_tracking.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Gráfico temporal guardado en: {fig_path}")

    plt.show()



# MOSTRAMOS POR CONSOLA LA DISTRIBUCIÓN DE TARGETS Y CLASES
def print_distributions(df, k=5):
    if TARGET_TYPE == 'excess_direction':
        conteo = df['Target'].value_counts()  # Conteo absoluto de muestras por clase
        proporcion = df['Target'].value_counts(normalize=True) * 100 # Proporción porcentual
        print("\n" + "=" * 65)
        print("DISTRIBUCIÓN GLOBAL DEL TARGET (CLASIFICACIÓN)")
        print("=" * 65)
        for clase, n in conteo.items():
            print(f"Clase {clase}: {n:,} muestras ({proporcion[clase]:.2f}%)")
    else:
        print("\n" + "=" * 65)
        print(f"ESTADÍSTICAS DESCRIPTIVAS DEL TARGET ({TARGET_TYPE.upper()})")
        print("=" * 65)
        print(df['Target'].describe().to_string())