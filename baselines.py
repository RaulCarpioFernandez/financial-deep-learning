import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
from arch import arch_model



# ==============================================================================
# BASELINES DE REGRESIÓN
# ==============================================================================
class ARIMABaseline:
    def __init__(self, order=(1, 0, 1), k_steps=5, refit_interval=50):
        self.order = order
        self.k_steps = k_steps
        self.refit_interval = refit_interval

    def evaluate_walk_forward(self, daily_series, test_start_idx, n_test_steps):
        """
        daily_series: Serie continua de retornos diarios (Log_Return_1)
        test_start_idx: Índice donde empieza el fold de test
        n_test_steps: Número de pasos a evaluar (len(y_test))
        """
        # Historial de retornos diarios conocidos hasta justo antes del test
        history = list(np.asarray(daily_series)[:test_start_idx])
        predictions = []

        model = ARIMA(history, order=self.order)
        fitted = model.fit()

        for t in range(n_test_steps):
            # 1. Pronosticar K pasos y sumar los retornos logarítmicos futuros
            forecast_k = fitted.forecast(steps=self.k_steps)
            predictions.append(float(np.sum(forecast_k)))

            # 2. Cerrar la sesión t: observar el retorno de hoy y añadirlo
            realized_today = daily_series[test_start_idx + t]
            history.append(realized_today)

            # 3. Actualizar estado sin reentrenar MLE desde cero cada día
            if (t + 1) % self.refit_interval == 0 and t < n_test_steps - 1:
                fitted = ARIMA(history, order=self.order).fit()
            else:
                fitted = fitted.apply(history, refit=False)

        return np.array(predictions)



class GARCHBaseline:
    def __init__(self, p=1, q=1, k_steps=5):
        self.p = p
        self.q = q
        self.k_steps = k_steps

    def evaluate_walk_forward(self, returns_series, test_start_idx, n_test_steps, refit_interval=1):
        # Multiplicar por 100 para estabilidad numérica en MLE (estándar en arch)
        raw_series = np.asarray(returns_series, dtype=np.float64) * 100.0
        history = list(raw_series[:test_start_idx])
        predictions = []

        # Ajuste inicial con media constante o cero y tolerancia relajada
        am = arch_model(history, vol='Garch', p=self.p, q=self.q, mean='Constant', dist='normal')
        res = am.fit(disp='off', show_warning=False)

        for t in range(n_test_steps):
            # Proyección de varianza a K pasos
            forecast = res.forecast(horizon=self.k_steps)
            # Media de las varianzas proyectadas en la ventana de K pasos (en escala %^2)
            mean_var_step = np.mean(forecast.variance.iloc[-1].values)
            
            # Anualización y desescalado (/ 100) para volver al espacio decimal del target
            vol_ann = np.sqrt(mean_var_step * 252.0) / 100.0
            
            # Clip de seguridad defensivo frente a colapsos numéricos
            vol_ann = float(np.clip(vol_ann, 0.02, 1.50))
            predictions.append(vol_ann)

            # Ingesta del dato diario
            history.append(raw_series[test_start_idx + t])

            # Reentrenamiento periódico
            if (t + 1) % refit_interval == 0 and t < n_test_steps - 1:
                try:
                    am = arch_model(history, vol='Garch', p=self.p, q=self.q, mean='Constant', dist='normal')
                    res = am.fit(disp='off', show_warning=False, starting_values=res.params)
                except Exception:
                    pass  # Si falla la optimización puntual, mantiene los pesos anteriores

        return np.array(predictions)
    

class XGBoostRegressorBaseline:
    def __init__(self, **params):
        default_params = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1
        }
        default_params.update(params)
        self.model = XGBRegressor(**default_params)

    def fit_predict(self, X_train, y_train, X_test):
        self.model.fit(X_train, y_train)
        return self.model.predict(X_test)


# ==============================================================================
# BASELINES DE CLASIFICACIÓN
# ==============================================================================
class LogisticRegressionBaseline:
    def __init__(self, C=0.1, solver="lbfgs", random_state=42):
        # Regularización L2 (Ridge) moderada (C=0.1) para evitar sobreajuste en features financieras ruidosas
        self.model = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        )

    def fit_predict_proba(self, X_train, y_train, X_test):
        self.model.fit(X_train, y_train)
        # Extraemos la probabilidad de la clase positiva (1)
        return self.model.predict_proba(X_test)[:, 1]


class XGBoostClassifierBaseline:
    def __init__(self, **params):
        default_params = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1
        }
        default_params.update(params)
        self.model = XGBClassifier(**default_params)

    def fit_predict_proba(self, X_train, y_train, X_test):
        self.model.fit(X_train, y_train)
        return self.model.predict_proba(X_test)[:, 1]