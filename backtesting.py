import os 
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from config import FIGURES_DIR, METRICS_DIR


# ==============================================================================
# FUNCIÓN DE EVALUACIÓN ECONÓMICA Y BACKTESTING HÍBRIDO
# ==============================================================================
def run_economic_backtest(df_total, df_test, test_probs, cost_bps=5, risk_aversion=6.0, 
                          model_name='lstm', plot_curves=True, save_results=True):
    """
    df_total: DataFrame completo (para calcular la SMA 200 histórica continua)
    df_test: DataFrame de test alineado
    test_probs: Vector 1D con las probabilidades predichas
    cost_bps: Coste por rotación (5 bps = 0.0005)
    risk_aversion: Parámetro gamma de aversión al riesgo
    """

    if save_results:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)

    bt = pd.DataFrame(index=df_test.index)
    
    # Asegurar series unidimensionales (1D) sin conflictos de MultiIndex
    price_total = df_total['Close'].squeeze()
    price_test = df_test['Close'].squeeze()
    
    bt['Price'] = price_test
    bt['Market_Return'] = bt['Price'].pct_change().shift(-1)
    bt['Prob'] = np.array(test_probs).ravel()
    
    # Filtro Macro: SMA 200
    sma_200 = price_total.rolling(200).mean().reindex(df_test.index).squeeze()
    # El mercado es alcista si el precio supera la SMA 200
    bull_market = (bt['Price'].values > sma_200.values)

    # Convicción Centrada en la Mediana Histórica de Predicciones
    conviction = 4.0 * (bt['Prob'].values - 0.5)

    # Asignación Asimétrica con Suelo Defensivo al 20%:
    # - En Bull Market: Base 90%, modulable entre 50% y 100%
    # - En Bear Market: Defensivo (entre -100% en corto y +60% en rebotes de alta convicción)
    pos_bull = np.clip(0.90 + conviction, 0.50, 1.00)
    #pos_bear = np.where(bt['Prob'].values < 0.5, -0.50, np.clip(conviction, -0.50, 0.50))
    pos_bear = np.where(bt['Prob'].values < 0.5, 0.00, np.clip(conviction, 0.00, 0.50))
    
    bt['Position_Strategy'] = np.where(bull_market, pos_bull, pos_bear)

    # --- BENCHMARK ALWAYS LONG AJUSTADO A MISMA EXPOSICIÓN ---
    gross_exposure = np.mean(abs(bt['Position_Strategy']))
    bt['Position_ConstLong'] = gross_exposure
    
    # Costes por Turnover Continuo
    c = cost_bps / 10000.0
    bt['Turnover_Strategy'] = bt['Position_Strategy'].diff().abs().fillna(abs(bt['Position_Strategy'].iloc[0]))
    bt['Costs_Strategy'] = bt['Turnover_Strategy'] * c

    # Coste Always Long (únicamente coste de entrada inicial en t=0)
    bt['Costs_ConstLong'] = 0.0
    bt.loc[bt.index[0], 'Costs_ConstLong'] = np.mean(abs(bt['Position_Strategy'])) * c

    # Coste de Buy & Hold
    bt['Costs_BnH'] = 0.0
    bt.loc[bt.index[0], 'Costs_BnH'] = 1.0 * c
    
    # Retornos y Curva
    bt['Net_Return_Strategy'] = bt['Position_Strategy'] * bt['Market_Return'] - bt['Costs_Strategy']
    bt['Net_Return_ConstLong'] = bt['Position_ConstLong'] * bt['Market_Return'] - bt['Costs_ConstLong']
    bt['Net_Return_BnH'] = 1.0 * bt['Market_Return'] - bt['Costs_BnH']
    bt = bt.dropna()
    
    bt['Cum_Strategy'] = (1 + bt['Net_Return_Strategy']).cumprod()
    bt['Cum_ConstLong'] = (1 + bt['Net_Return_ConstLong']).cumprod()
    bt['Cum_BnH'] = (1 + bt['Net_Return_BnH']).cumprod()
    
    # Métricas Financieras
    n_days = len(bt)
    gamma = risk_aversion
    
    # Utilidad de referencia del mercado (Buy & Hold)
    r_mkt = bt['Net_Return_BnH'].values
    util_mkt = np.mean(r_mkt) - 0.5 * gamma * np.var(r_mkt, ddof=1)

    # FUNCIÓN MODULAR DE MÉTRICAS FINANCIERAS Y DISTRIBUCIONALES
    def get_metrics(r_series, cum_series):
        r_arr = r_series.values
        cum_arr = cum_series.values
        # Rendimiento y Riesgo
        cagr = (cum_arr[-1]) ** (252.0 / n_days) - 1.0
        ann_ret = np.mean(r_arr) * 252
        ann_vol = np.std(r_arr, ddof=1) * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-9)
        # Drawdown
        peak = np.maximum.accumulate(cum_arr)
        mdd = np.max((peak - cum_arr) / peak)
        # Momentos de Orden Superior
        skew = stats.skew(r_arr)
        kurt = stats.kurtosis(r_arr)
        # Ganancia de Utilidad (Certainty Equivalent Return / Delta U)
        util = np.mean(r_arr) - 0.5 * gamma * np.var(r_arr, ddof=1)
        delta_util_ann = 252 * (util - util_mkt) * 100
        
        return cagr, ann_ret, ann_vol, sharpe, mdd, skew, kurt, delta_util_ann

    cagr_strat, ann_ret_strat, ann_vol_strat, sharpe_strat, mdd_strat, skew_strat, kurt_strat, delta_util_strat = get_metrics(bt['Net_Return_Strategy'], bt['Cum_Strategy'])
    cagr_lon, ann_ret_lon, ann_vol_lon, sharpe_lon, mdd_lon, skew_lon, kurt_lon, delta_util_lon = get_metrics(bt['Net_Return_ConstLong'], bt['Cum_ConstLong'])
    cagr_bnh, ann_ret_bnh, ann_vol_bnh, sharpe_bnh, mdd_bnh, skew_bnh, kurt_bnh, _ = get_metrics(bt['Net_Return_BnH'], bt['Cum_BnH'])

    # Cálculo de Campbell & Thompson R^2_OOS
    # Retorno bruto de mercado sin costes de transacción
    r_mkt_raw = bt['Market_Return'].values

    # Convicción centrada en la mediana móvil multiplicada por la volatilidad realizada
    prob_center = bt['Prob'].rolling(60, min_periods=15).median().fillna(0.50).values
    vol_rolling = bt['Market_Return'].rolling(20, min_periods=5).std().bfill().values
    implied_expected_ret = (bt['Prob'].values - prob_center) * vol_rolling
    # Media histórica prevalente (Expanding Window)
    prevailing_mean = bt['Market_Return'].expanding(min_periods=20).mean().bfill().values
    # Filtrar máscaras válidas
    valid_mask = ~(np.isnan(implied_expected_ret) | np.isnan(prevailing_mean) | np.isnan(r_mkt_raw))
    ss_model = np.sum((r_mkt_raw[valid_mask] - implied_expected_ret[valid_mask]) ** 2)
    ss_bench = np.sum((r_mkt_raw[valid_mask] - prevailing_mean[valid_mask]) ** 2)
    r2_oos = 1.0 - (ss_model / ss_bench)

    # Exposición media
    num_long_days = np.sum(bt['Position_Strategy'] > 0.05)
    num_short_days = np.sum(bt['Position_Strategy'] < -0.05)
    num_neutral_days = np.sum(np.abs(bt['Position_Strategy']) <= 0.05)

    # Diagnóstico Operativo
    total_costs_pct = np.sum(bt['Costs_Strategy']) * 100
    ann_turnover = np.mean(bt['Turnover_Strategy']) * 252

    # Cálculo del Brier Skill Score (BSS)
    y_test_real = df_test.loc[bt.index, 'Target'].values
    probs_eval = bt['Prob'].values
    brier_model = brier_score_loss(y_test_real, probs_eval)
    base_rate = np.mean(y_test_real)
    brier_base = np.mean((base_rate - y_test_real) ** 2)
    bss = 1.0 - (brier_model / (brier_base + 1e-9))

    # Informe en Consola
    print("\n" + "═" * 86)
    print(f"{'INFORME DE RENDIMIENTO ECONÓMICO Y GESTIÓN DE RIESGO':^86}")
    print("═" * 86)
    print(f" Parámetros de Simulación : Costes: {cost_bps} bps │ Aversión al Riesgo (γ): {risk_aversion:.1f}")
    print(f" Distribución de Posición : Largos: {num_long_days}d ({num_long_days/len(bt)*100:.1f}%) │ Cortos: {num_short_days}d ({num_short_days/len(bt)*100:.1f}%) │ Neutro: {num_neutral_days}d ({num_neutral_days/len(bt)*100:.1f}%)")
    print(f" Exposición Bruta/Turnover : {gross_exposure*100:.1f}% invertido │ Rotación Anualizada: {ann_turnover:.2f}x │ Costes Totales: {total_costs_pct:.2f}%")
    print("─" * 86)
    print(f" {'MÉTRICA FINANCIERA':<28} │ {'ESTRATEGIA IA':>15} │ {f'CONST. LONG ({gross_exposure*100:.0f}%)':>16} │ {'S&P 500 (100%)':>16}")
    print("─" * 86)
    print(f" {'CAGR (Rentabilidad Compuesta)':<28} │ {cagr_strat*100:>14.2f}% │ {cagr_lon*100:>15.2f}% │ {cagr_bnh*100:>15.2f}%")
    print(f" {'Retorno Aritmético Anual':<28} │ {ann_ret_strat*100:>14.2f}% │ {ann_ret_lon*100:>15.2f}% │ {ann_ret_bnh*100:>15.2f}%")
    print(f" {'Volatilidad Anualizada (σ)':<28} │ {ann_vol_strat*100:>14.2f}% │ {ann_vol_lon*100:>15.2f}% │ {ann_vol_bnh*100:>15.2f}%")
    print(f" {'Ratio de Sharpe (Rf = 0)':<28} │ {sharpe_strat:>15.3f} │ {sharpe_lon:>16.3f} │ {sharpe_bnh:>16.3f}")
    print(f" {'Máximo Drawdown (MDD)':<28} │ {mdd_strat*100:>14.2f}% │ {mdd_lon*100:>15.2f}% │ {mdd_bnh*100:>15.2f}%")
    print(f" {'Asimetría (Skewness)':<28} │ {skew_strat:>15.3f} │ {skew_lon:>16.3f} │ {skew_bnh:>16.3f}")
    print(f" {'Curtosis Excesiva':<28} │ {kurt_strat:>15.3f} │ {kurt_lon:>16.3f} │ {kurt_bnh:>16.3f}")
    print("─" * 86)
    print(f" {'Ganancia Utilidad (ΔU / CER)':<28} │ {delta_util_strat:>+14.2f}% │ {delta_util_lon:>+15.2f}% │ {'0.00% (Base)':>16}")
    print(f" {'Brier Skill Score (BSS)':<28} │ {bss:>+15.4f} │ {'N/A (Estático)':>16} │ {'0.0000 (Base)':>16}")
    print(f" {'R² Fuera de Muestra (C&T)':<28} │ {r2_oos:>15.4f} │ {'N/A (Estático)':>16} │ {'0.0000 (Base)':>16}")
    print("═" * 86 + "\n")
    
    # Gráficas
    if plot_curves:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Gráfico superior: Curvas de Riqueza
        ax1.plot(bt.index, bt['Cum_Strategy'], label=f'Estrategia {model_name.upper()} Híbrida (Neta)', color='tab:blue', lw=2)
        ax1.plot(bt.index, bt['Cum_ConstLong'], label=f'Constant Long ({gross_exposure*100:.0f}%)', color='tab:orange', linestyle=':', lw=1.5)
        ax1.plot(bt.index, bt['Cum_BnH'], label='Buy & Hold (S&P 500)', color='tab:gray', linestyle='--', lw=1.5)
        ax1.set_title('Backtesting Económico: Estrategia Híbrida vs Benchmarks', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Riqueza Acumulada')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Gráfico inferior: Asignación de Capital
        ax2.plot(bt.index, bt['Position_Strategy'], label='Exposición Estrategia ($w_t$)', color='tab:purple', lw=1.2)
        ax2.axhline(gross_exposure, color='tab:orange', linestyle=':', lw=1.2, label=f'Exposición Media ({gross_exposure*100:.0f}%)')
        ax2.axhline(0, color='black', linestyle=':', lw=1)
        ax2.set_ylabel('Exposición')
        ax2.set_xlabel('Fecha')
        ax2.set_ylim(-1.1, 1.1)
        ax2.legend(loc='lower left')
        ax2.grid(True, linestyle=':', alpha=0.6)

        if save_results:
            fig_path = os.path.join(FIGURES_DIR, f'{model_name.lower()}_backtest_curves.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Gráfico de backtest guardado en: {fig_path}")

        plt.tight_layout()
        plt.show()

    # 8. Guardado Estructurado de Métricas Económicas
    summary_data = {}
    if save_results:
        summary_data = {
            'Model': model_name.upper(),
            'CAGR_Strategy': cagr_strat,
            'CAGR_ConstLong': cagr_lon,
            'CAGR_BnH': cagr_bnh,
            'Sharpe_Strategy': sharpe_strat,
            'Sharpe_ConstLong': sharpe_lon,
            'Sharpe_BnH': sharpe_bnh,
            'MDD_Strategy': mdd_strat,
            'MDD_ConstLong': mdd_lon,
            'MDD_BnH': mdd_bnh,
            'Delta_Util_Strategy': delta_util_strat,
            'Delta_Util_ConstLong': delta_util_lon,
            'BSS': bss,
            'R2_OOS': r2_oos,
            'Gross_Exposure': gross_exposure,
            'Ann_Turnover': ann_turnover,
            'Total_Costs_Pct': total_costs_pct
        }
        
        # Guardar JSON completo
        json_path = os.path.join(METRICS_DIR, f'{model_name.lower()}_financial_metrics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)

        # Guardar CSV tabular
        csv_path = os.path.join(METRICS_DIR, f'{model_name.lower()}_financial_summary.csv')
        pd.DataFrame([summary_data]).to_csv(csv_path, index=False)
        print(f"[INFO] Métricas financieras guardadas en: {json_path} y {csv_path}")

    return bt, summary_data