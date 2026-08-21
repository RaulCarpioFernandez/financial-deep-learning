import os
import copy
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models import get_model
from config import DEVICE, SEQUENCE_LENGTH, K, FIGURES_DIR, METRICS_DIR, MODELS_DIR
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, 
    classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score
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
def run_purged_walk_forward(df, feature_cols, model_name = 'lstm', K=5, seq_length=SEQUENCE_LENGTH, n_splits=4, train_size = 2016, test_size=504, val_size=126, window_type='expanding'):
    """
    Ejecuta Purged Walk-Forward CV con Rolling o Expanding Window
    
    Parámetros:
    -----------
    train_size : Número de sesiones fijas de entrenamiento (ej. 2016 = ~8 años, 2520 = ~10 años)
    test_size  : Sesiones por bloque de test ciego (ej. 504 = ~2 años)
    val_size   : Sesiones para Early Stopping (ej. 126 = ~6 meses)
    K          : Purging gap para evitar data leakage
    """
    if window_type not in ['expanding', 'rolling']:
        raise ValueError(
            "window_type debe ser 'expanding' o 'rolling'"
        )
    X_raw = df[feature_cols].values
    y_raw = df['Target'].values
    total_len = len(df)

    # Verificación de datos mínimos necesarios
    required_len = (n_splits * test_size) + val_size + (2 * K) + (train_size if window_type == 'rolling' else 500)
    if total_len < required_len:
        raise ValueError(
            f"Longitud insuficiente ({total_len} sesiones). Se requieren al menos {required_len} "
            f"para {n_splits} splits con train_size={train_size}, val_size={val_size}, test_size={test_size}."
        )
    
    all_test_indices = []
    all_test_probs = []
    all_test_reals = []
    fold_metrics = []

    window_header = f"ROLLING WINDOW ({train_size}d)" if window_type == 'rolling' else "EXPANDING WINDOW"

    print("\n" + "═" * 78)
    print(f"{'INICIANDO PURGED WALK-FORWARD CROSS-VALIDATION':^78}")
    print(f"{f'({window_header} │ {n_splits} Pliegues │ Val: {val_size}d │ Test: {test_size}d │ Gap: {K}d)':^78}")
    print("═" * 78)

    for fold in range(n_splits):
        # Definición de límites temporales del pliegue
        # Pliegue 0 es el más antiguo y Pliegue (n_splits-1) termina en la última fila de df
        test_end = total_len - (n_splits - 1 - fold) * test_size
        test_start = test_end - test_size

        val_end = test_start - K        # Purging Gap 2 entre Val y Test
        val_start = val_end - val_size

        train_end = val_start - K       # Purging Gap 1 entre Train y Val
        train_start = 0 if window_type == 'expanding' else train_end - train_size  # <<--- Ventana deslizante o expansiva

        train_dates = f"{df.index[train_start].strftime('%Y-%m')} a {df.index[train_end].strftime('%Y-%m')}"
        val_dates = f"{df.index[val_start].strftime('%Y-%m')} a {df.index[val_end].strftime('%Y-%m')}"
        test_dates = f"{df.index[test_start].strftime('%Y-%m')} a {df.index[test_end-1].strftime('%Y-%m')}"
        
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
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_hist, seq_length=SEQUENCE_LENGTH)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_hist, seq_length=SEQUENCE_LENGTH)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_hist, seq_length=SEQUENCE_LENGTH)

        # Tensores y DataLoader
        X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
        X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(1)
        X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
        y_test_t = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(1)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

        # Modelo y Optimización del Pliegue
        model = get_model(model_name, input_dim=len(feature_cols)).to(DEVICE)
        # Cálculo del balance en Train para calcular la pérdida ponderada
        num_pos = np.sum(y_raw[train_start:train_end] == 1)
        num_neg = np.sum(y_raw[train_start:train_end] == 0)
        pos_weight = torch.tensor([num_neg / (num_pos + 1e-9)], dtype=torch.float32).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        # Entrenamiento con Early Stopping por ROC-AUC
        EPOCHS = 40
        patience = 6
        patience_counter = 0
        #best_val_loss = float('inf')
        best_val_auc = -float('inf')
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
                # Conversión universal sin errores
                val_probs = torch.sigmoid(val_logits).detach().cpu().numpy().ravel()
                y_val_np = y_val_t.detach().cpu().numpy().ravel()
                val_auc = roc_auc_score(y_val_np, val_probs)
            scheduler.step(val_auc)

            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping en época {epoch+1}")
                    break

        model.load_state_dict(best_model_weights)

        # Guardar pesos en /results/models
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(best_model_weights, os.path.join(MODELS_DIR, f'{model_name.lower()}_fold_{fold+1}.pt'))

        # EVALUACIÓN EN TEST (fuera de la muestra)
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t.to(DEVICE))
            test_probs = torch.sigmoid(test_logits).detach().cpu().numpy().ravel()
            y_test_real = y_test_t.detach().cpu().numpy().ravel()

        threshold = 0.5
        test_predictions = (test_probs > threshold).astype(int)

        acc = accuracy_score(y_test_real, test_predictions)
        balanced_acc = balanced_accuracy_score(y_test_real, test_predictions)
        roc_auc = roc_auc_score(y_test_real, test_probs)
        
        fold_metrics.append({'fold': fold + 1, 'auc': roc_auc, 'acc': acc, 'samples': len(y_test_real)})
        print(f"  └─> Test Ciego Pliegue {fold + 1}: ROC-AUC = {roc_auc:.4f} │ Accuracy = {acc*100:.2f}%")

        all_test_indices.extend(list(df.index[test_start:test_end]))
        all_test_probs.extend(test_probs)
        all_test_reals.extend(y_test_real)

    # 8. Consolidación Global
    df_wf_test = df.loc[all_test_indices].copy()
    wf_probs = np.array(all_test_probs, dtype=np.float64)
    wf_reals = np.array(all_test_reals, dtype=np.int32)

    return df_wf_test, wf_probs, wf_reals, fold_metrics


def evaluate_ml_performance(wf_reals, wf_probs, fold_metrics, model_name='lstm', plot_curves=True, save_results=True):
    """
    Calcula, imprime y grafica el rendimiento global fuera de muestra del modelo de ML.
    """

    if save_results:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)

    optimal_threshold = np.median(wf_probs) #se puede fijar a 0.5
    global_predictions = (wf_probs > optimal_threshold).astype(int)
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
    print(f" Muestras Totales Acumuladas : {len(wf_reals)} sesiones | Umbral de Decisión (Calibrado): {optimal_threshold:.2f}")
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
            print(f"[INFO] Gráficos guardados en: {fig_path}")

        plt.tight_layout()
        plt.show()

    # Empaquetado y guardado estructurado de métricas
    ml_results = {
        'model': model_name.upper(),
        'optimal_threshold': optimal_threshold,
        'global_auc': global_auc,
        'mean_fold_auc': mean_fold_auc,
        'global_accuracy': global_acc,
        'global_balanced_accuracy': global_balanced_acc,
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
        print(f"[INFO] Métricas guardadas en: {json_path} y {csv_path}")

    return ml_results


# MOSTRAMOS POR CONSOLA LA DISTRIBUCIÓN DE TARGETS Y DE CLASES
def print_distributions(df):
    n = len(df)
    train_end = int(n * 0.70)
    val_start = train_end + K
    val_end = int(n * 0.85)
    test_start = val_end + K

    print("\n" + "=" * 60)
    print("DISTRIBUCIÓN DE TARGETS")
    print("=" * 60)

    for target_name, target_col in [
        ("Directional", "Target_Directional"),
        ("Relative", "Target_Relative")
    ]:

        print(f"\n{target_name} target:")

        for split_name, start, end in [
            ("Train", 0, train_end),
            ("Validation", val_start, val_end),
            ("Test", test_start, len(df))
        ]:
            y_split = df[target_col].iloc[start:end]

            pct_positive = y_split.mean() * 100
            pct_negative = (1 - y_split.mean()) * 100

            print(
                f"  {split_name:<12}: "
                f"Class 0 = {pct_negative:5.2f}% | "
                f"Class 1 = {pct_positive:5.2f}%"
            )

    print("\n" + "=" * 65)
    print("DISTRIBUCIÓN DE CLASES - CLASE 1")
    print("=" * 65)

    print(f"{'Target':<20} {'Train':>12} {'Validation':>12} {'Test':>12}")
    print("-" * 65)

    for target_name, target_col in [
        ("Directional", "Target_Directional"),
        ("Relative", "Target_Relative")
    ]:
        train_pct = df[target_col].iloc[:train_end].mean() * 100
        val_pct = df[target_col].iloc[val_start:val_end].mean() * 100
        test_pct = df[target_col].iloc[test_start:].mean() * 100

        print(
            f"{target_name:<20} "
            f"{train_pct:>11.2f}% "
            f"{val_pct:>11.2f}% "
            f"{test_pct:>11.2f}%"
        )

    print("=" * 65)
    print("Los porcentajes representan la proporción de clase 1.")