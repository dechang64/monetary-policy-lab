"""
Generate publication-quality figures for the paper.
"""
import pandas as pd
import numpy as np
import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, "analysis_dataset.csv"), parse_dates=["date"])
with open(os.path.join(RESULTS_DIR, "regression_results.json")) as f:
    results = json.load(f)

COLORS = {
    "conventional": "#4C72B0",
    "forward_guidance": "#C44E52",
    "normalization": "#55A868",
}

def fig1_sentiment_vs_surprise():
    """Figure 1: Sentiment vs Surprise scatter, colored by regime."""
    fig = go.Figure()
    for regime, color in COLORS.items():
        mask = df["regime"] == regime
        d = df[mask]
        fig.add_trace(go.Scatter(
            x=d["surprise"], y=d["sentiment"],
            mode="markers", name=regime.replace("_", " ").title(),
            marker=dict(color=color, size=8, opacity=0.7),
            showlegend=True,
        ))
    
    # Regression line
    x = df["surprise"].values
    y = df["sentiment"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    slope, intercept, r, p, se = __import__('scipy').stats.linregress(x[mask], y[mask])
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=slope * x_line + intercept,
        mode="lines", name=f"OLS (R²={r**2:.3f}, p={p:.3f})",
        line=dict(color="black", dash="dash", width=2),
    ))
    
    fig.update_layout(
        title=dict(text="Figure 1: FOMC Statement Sentiment vs. Monetary Policy Surprise",
                    font=dict(size=14)),
        xaxis_title="Monetary Policy Surprise (DFF change, pp)",
        yaxis_title="Statement Sentiment (LM + CB Dictionary)",
        template="plotly_white", height=500, width=700,
        legend=dict(x=0.02, y=0.98),
    )
    fig.write_html(os.path.join(FIG_DIR, "fig1_sentiment_vs_surprise.html"))
    print("  ✅ Figure 1 saved")


def fig2_sentiment_timeline():
    """Figure 2: Sentiment over time, colored by Chair."""
    fig = go.Figure()
    chairs = df["chair"].unique()
    chair_colors = {"Greenspan": "#4C72B0", "Bernanke": "#C44E52", "Yellen": "#55A868", "Powell": "#CCB974"}
    
    for chair in chairs:
        mask = df["chair"] == chair
        d = df[mask].sort_values("date")
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["sentiment"],
            mode="lines+markers", name=chair,
            line=dict(color=chair_colors.get(chair, "#999"), width=1.5),
            marker=dict(size=4),
        ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    
    # Regime shading
    fig.add_vrect(x0="2008-01-01", x1="2015-12-31", fillcolor="#C44E52", opacity=0.05,
                  annotation_text="Forward Guidance Era")
    
    fig.update_layout(
        title=dict(text="Figure 2: FOMC Statement Sentiment Over Time (1994\u20132025)",
                    font=dict(size=14)),
        xaxis_title="Date", yaxis_title="Statement Sentiment",
        template="plotly_white", height=400, width=900,
    )
    fig.write_html(os.path.join(FIG_DIR, "fig2_sentiment_timeline.html"))
    print("  ✅ Figure 2 saved")


def fig3_incremental_r2():
    """Figure 3: Incremental R² bar chart."""
    h2 = results["H2"]
    assets = []
    inc_r2 = []
    p_vals = []
    for name, r in h2.items():
        if isinstance(r, dict):
            assets.append(name.replace("_ret", " Return").replace("_chg", " ΔYield").replace("sp500", "S&P 500").replace("nasdaq", "NASDAQ").replace("gold", "Gold").replace("ty10", "10Y Treasury").replace("tb13w", "13W T-Bill"))
            inc_r2.append(r["inc_r2"] * 100)
            p_vals.append(r["p_value"])
    
    colors = ["#C44E52" if p < 0.10 else "#4C72B0" for p in p_vals]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=assets, y=inc_r2,
        marker_color=colors,
        text=[f"{v:.2f}%" for v in inc_r2],
        textposition="outside",
    ))
    
    fig.update_layout(
        title=dict(text="Figure 3: Incremental R² from Adding Sentiment to Surprise Model",
                    font=dict(size=14)),
        yaxis_title="Incremental R² (%)",
        template="plotly_white", height=450, width=700,
        annotations=[
            dict(x=0.02, y=0.95, text="■ p < 0.10", showarrow=False,
                 font=dict(color="#C44E52", size=11)),
            dict(x=0.02, y=0.90, text="■ p ≥ 0.10", showarrow=False,
                 font=dict(color="#4C72B0", size=11)),
        ],
    )
    fig.write_html(os.path.join(FIG_DIR, "fig3_incremental_r2.html"))
    print("  ✅ Figure 3 saved")


def fig4_coefficient_plot():
    """Figure 4: Sentiment coefficient by regime."""
    h4 = results["H4"]
    regimes = ["conventional", "forward_guidance", "normalization"]
    labels = ["Conventional\n(1994\u20132007)", "Forward Guidance\n(2008\u20132015)", "Normalization\n(2016\u20132025)"]
    betas = [h4[r].get("beta2", 0) for r in regimes]
    ses = [h4[r].get("se_beta2", 0) for r in regimes]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=betas,
        error_y=dict(type="data", array=ses, visible=True),
        marker_color=[COLORS[r] for r in regimes],
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title=dict(text="Figure 4: Sentiment Coefficient (β₂) by Monetary Policy Regime",
                    font=dict(size=14)),
        yaxis_title="β₂ (Sentiment Coefficient)",
        template="plotly_white", height=450, width=700,
    )
    fig.write_html(os.path.join(FIG_DIR, "fig4_regime_coefficients.html"))
    print("  ✅ Figure 4 saved")


def fig5_two_shocks_radar():
    """Figure 5: Two-shocks loading radar."""
    h3 = results["H3"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[abs(h3["policy_loading"]), abs(h3["info_loading"]), 
           abs(h3["policy_loading"])],
        theta=["Policy Shock", "Information Shock", "Policy Shock"],
        fill="toself", name="Sentiment Loading",
        line=dict(color="#4C72B0", width=2),
        marker=dict(size=8),
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
        title=dict(text="Figure 5: Sentiment Loading on Two-Shocks Decomposition",
                    font=dict(size=14)),
        height=450, width=500,
        showlegend=False,
    )
    fig.write_html(os.path.join(FIG_DIR, "fig5_two_shocks_radar.html"))
    print("  ✅ Figure 5 saved")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_sentiment_vs_surprise()
    fig2_sentiment_timeline()
    fig3_incremental_r2()
    fig4_coefficient_plot()
    fig5_two_shocks_radar()
    print("\nAll figures saved to", FIG_DIR)
