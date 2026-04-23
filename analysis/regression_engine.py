"""
Regression Engine for Phase 1 Research
=======================================
OLS regression with robust standard errors for testing
the incremental information content of FOMC language.

Models:
1. Sentiment ~ Surprise (H1: related but not collinear)
2. Asset_Return ~ Surprise + Sentiment (H2: incremental R²)
3. Sentiment ~ Policy_Shock + Info_Shock (H3: info channel)
4. Asset_Return ~ Surprise + Sentiment + Sentiment×FG (H4: FG period)
"""

import numpy as np
import pandas as pd
from typing import Optional


class RegressionEngine:
    """
    Run OLS regressions with Newey-West robust standard errors.
    Uses numpy for portability (no statsmodels dependency).
    """

    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: DataFrame with all variables as columns.
                  Index should be FOMC dates.
        """
        self.data = data.dropna()
        self.results = {}

    def ols(self, y_col: str, x_cols: list, robust: bool = True) -> dict:
        """
        OLS regression: y = Xβ + ε

        Args:
            y_col: Name of dependent variable column
            x_cols: List of independent variable column names
            robust: Use Newey-West standard errors

        Returns:
            dict with coefficients, std_errors, t_stats, p_values, R², N
        """
        df = self.data[[y_col] + x_cols].dropna()
        y = df[y_col].values
        n = len(y)

        # Add constant
        X = np.column_stack([np.ones(n)] + [df[c].values for c in x_cols])
        k = X.shape[1]

        # OLS: β = (X'X)^{-1} X'y
        XtX = X.T @ X
        Xty = X.T @ y
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            return {"error": "Singular matrix", "n": n}

        beta = XtX_inv @ Xty
        residuals = y - X @ beta
        sse = np.sum(residuals ** 2)
        sst = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - sse / sst if sst > 0 else 0
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k) if n > k else 0
        sigma2 = sse / (n - k)

        # Standard errors
        if robust:
            # Newey-West (lag = int(4*(n/100)^(2/9)))
            lag = max(1, int(4 * (n / 100) ** (2 / 9)))
            cov = np.zeros((k, k))
            for i in range(n):
                xi = X[i:i + 1, :]
                ei = residuals[i]
                for l in range(lag + 1):
                    w = 1 - l / (lag + 1)
                    j = i + l
                    if j < n:
                        xj = X[j:j + 1, :]
                        cov += w * (xi.T @ xj) * ei * residuals[j]
            se = np.sqrt(np.diag(XtX_inv @ cov @ XtX_inv))
        else:
            se = np.sqrt(np.diag(sigma2 * XtX_inv))

        t_stats = beta / np.where(se > 0, se, 1e-10)
        # Approximate p-values using normal distribution
        from scipy import stats
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

        col_names = ["const"] + x_cols
        result = {
            "coefficients": dict(zip(col_names, beta)),
            "std_errors": dict(zip(col_names, se)),
            "t_stats": dict(zip(col_names, t_stats)),
            "p_values": dict(zip(col_names, p_values)),
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "n": n,
            "k": k,
            "sigma2": sigma2,
        }
        return result

    def format_table(self, result: dict, title: str = "OLS Regression") -> pd.DataFrame:
        """Format regression results as a publication-ready table."""
        if "error" in result:
            return pd.DataFrame({"Error": [result["error"]]})

        cols = list(result["coefficients"].keys())
        rows = []
        for c in cols:
            sig = ""
            p = result["p_values"][c]
            if p < 0.01:
                sig = "***"
            elif p < 0.05:
                sig = "**"
            elif p < 0.10:
                sig = "*"
            rows.append({
                "Variable": c,
                "Coefficient": f"{result['coefficients'][c]:.6f}",
                "Std Error": f"{result['std_errors'][c]:.6f}",
                "t-stat": f"{result['t_stats'][c]:.3f}{sig}",
                "p-value": f"{p:.4f}",
            })

        # Add fit statistics
        rows.append({"Variable": "", "Coefficient": "", "Std Error": "", "t-stat": "", "p-value": ""})
        rows.append({"Variable": "R²", "Coefficient": f"{result['r_squared']:.4f}", "Std Error": "", "t-stat": "", "p-value": ""})
        rows.append({"Variable": "Adj R²", "Coefficient": f"{result['adj_r_squared']:.4f}", "Std Error": "", "t-stat": "", "p-value": ""})
        rows.append({"Variable": "N", "Coefficient": str(result["n"]), "Std Error": "", "t-stat": "", "p-value": ""})

        return pd.DataFrame(rows)

    def incremental_r2(self, y_col: str, base_cols: list, full_cols: list) -> dict:
        """
        Test incremental R² from adding variables.

        Args:
            y_col: Dependent variable
            base_cols: Base model predictors (e.g., ['surprise'])
            full_cols: Full model predictors (e.g., ['surprise', 'sentiment'])

        Returns:
            dict with base_r2, full_r2, incremental_r2, f_stat, p_value
        """
        base = self.ols(y_col, base_cols)
        full = self.ols(y_col, full_cols)

        if "error" in base or "error" in full:
            return {"error": "Regression failed"}

        inc_r2 = full["r_squared"] - base["r_squared"]
        n = full["n"]
        k_full = full["k"]
        k_base = base["k"]
        q = k_full - k_base  # Number of added variables

        if q <= 0 or n <= k_full:
            return {"error": "Invalid model comparison"}

        # F-test for incremental R²
        f_stat = (inc_r2 / q) / ((1 - full["r_squared"]) / (n - k_full))
        from scipy import stats
        p_value = 1 - stats.f.cdf(f_stat, q, n - k_full)

        return {
            "base_r2": base["r_squared"],
            "full_r2": full["r_squared"],
            "incremental_r2": inc_r2,
            "f_stat": f_stat,
            "p_value": p_value,
            "n": n,
        }

    def run_phase1_models(self, sentiment_col: str = "sentiment_score",
                          surprise_col: str = "surprise_bp") -> dict:
        """
        Run all four Phase 1 regression models.

        Returns:
            dict with model1 through model4 results
        """
        results = {}

        # Model 1: Sentiment ~ Surprise
        if sentiment_col in self.data.columns and surprise_col in self.data.columns:
            results["model1"] = self.ols(sentiment_col, [surprise_col])

        # Model 2: Asset returns ~ Surprise + Sentiment (for each asset)
        asset_cols = [c for c in self.data.columns if c not in
                      [sentiment_col, surprise_col, "label", "hawkish_count",
                       "dovish_count", "readability", "f_pre", "f_post"]]
        results["model2"] = {}
        for asset in asset_cols:
            r = self.ols(asset, [surprise_col, sentiment_col])
            if "error" not in r:
                results["model2"][asset] = r

        # Model 3: Sentiment ~ Policy_Shock + Info_Shock (if available)
        if "policy_shock" in self.data.columns and "info_shock" in self.data.columns:
            results["model3"] = self.ols(sentiment_col, ["policy_shock", "info_shock"])

        # Model 4: Interaction with Forward Guidance period
        if "fg_period" in self.data.columns:
            interaction = f"{sentiment_col}_x_fg"
            if interaction not in self.data.columns:
                self.data[interaction] = self.data[sentiment_col] * self.data["fg_period"]
            results["model4"] = self.ols(
                sentiment_col,
                [surprise_col, sentiment_col, interaction],
            )

        self.results = results
        return results
