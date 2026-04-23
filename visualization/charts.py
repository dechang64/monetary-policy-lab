"""
Visualization Module
====================
Plotly-based charts for Monetary Policy Research Lab.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Color Palette ──
COLORS = {
    "policy": "#e74c3c",
    "information": "#3498db",
    "hawkish": "#e74c3c",
    "dovish": "#27ae60",
    "neutral": "#95a5a6",
    "risk_on": "#27ae60",
    "risk_off": "#e74c3c",
    "positive": "#27ae60",
    "negative": "#e74c3c",
    "bg": "#fafbfc",
    "grid": "#ecf0f1",
}


def event_study_bar(df, title="Cumulative Abnormal Returns by Asset Class"):
    """Horizontal bar chart of CAR by asset class."""
    if df.empty or "avg_CAR_pct" not in df.columns:
        return go.Figure().update_layout(
            title=dict(text="No event study results", font=dict(size=14)),
            template="plotly_white", height=300,
        )
    df = df.sort_values("avg_CAR_pct")
    colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in df["avg_CAR_pct"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["asset"], x=df["avg_CAR_pct"], orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}%" for v in df["avg_CAR_pct"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>CAR: %{x:.4f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Cumulative Abnormal Return (%)", yaxis_title="",
        template="plotly_white", height=max(400, len(df) * 40),
        margin=dict(l=150), showlegend=False,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def event_study_timeline(df, asset, title=None):
    """Timeline of CAR for a single asset across FOMC events."""
    asset_df = df[df["asset"] == asset].sort_values("fomc_date")
    colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in asset_df["CAR"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=asset_df["fomc_date"], y=asset_df["CAR"] * 100,
        marker_color=colors,
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>CAR: %{y:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title or f"FOMC Event Study: {asset}", font=dict(size=16)),
        xaxis_title="FOMC Date", yaxis_title="CAR (%)",
        template="plotly_white", height=400, showlegend=False,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def two_shocks_radar(response_df, title="Two-Shocks Radar: Asset Response by Shock Type"):
    """Radar chart: policy vs information shock responses."""
    policy_data = response_df[response_df["shock_type"] == "Policy"]
    info_data = response_df[response_df["shock_type"] == "Information"]
    categories = policy_data["asset"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=policy_data["avg_response_pct"].tolist(), theta=categories,
        fill="toself", name="Policy Shock",
        line_color=COLORS["policy"], fillcolor="rgba(231,76,60,0.15)", marker=dict(size=6),
    ))
    fig.add_trace(go.Scatterpolar(
        r=info_data["avg_response_pct"].tolist(), theta=categories,
        fill="toself", name="Information Shock",
        line_color=COLORS["information"], fillcolor="rgba(52,152,219,0.15)", marker=dict(size=6),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, gridcolor=COLORS["grid"])),
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.1),
    )
    return fig


def two_shocks_bar(response_df, title="Asset Response: Policy vs Information Shock"):
    """Grouped bar chart comparing shock responses."""
    fig = go.Figure()
    policy = response_df[response_df["shock_type"] == "Policy"]
    info = response_df[response_df["shock_type"] == "Information"]
    fig.add_trace(go.Bar(name="Policy Shock", x=policy["asset"], y=policy["avg_response_pct"],
                         marker_color=COLORS["policy"]))
    fig.add_trace(go.Bar(name="Information Shock", x=info["asset"], y=info["avg_response_pct"],
                         marker_color=COLORS["information"]))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title="Average Response (%)", barmode="group",
        template="plotly_white", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.1), xaxis_tickangle=-30,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def sentiment_trajectory(sentiment_df, title="FOMC Sentiment Trajectory"):
    """Line chart of sentiment scores over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sentiment_df.index, y=sentiment_df["sentiment_score"],
        mode="lines+markers", name="Sentiment Score",
        line=dict(color="#2c3e50", width=2),
        marker=dict(
            color=[
                COLORS["hawkish"] if s < -0.15 else COLORS["dovish"] if s > 0.15 else COLORS["neutral"]
                for s in sentiment_df["sentiment_score"]
            ], size=8,
        ),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Score: %{y:.3f}<extra></extra>",
    ))
    fig.add_hrect(y0=-1, y1=-0.15, fillcolor=COLORS["hawkish"], opacity=0.05,
                 annotation_text="Hawkish", annotation_position="top left")
    fig.add_hrect(y0=0.15, y1=1, fillcolor=COLORS["dovish"], opacity=0.05,
                 annotation_text="Dovish", annotation_position="top left")
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title="Sentiment (-1=Hawkish, +1=Dovish)", xaxis_title="FOMC Date",
        template="plotly_white", height=400, showlegend=False,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def sankey_diagram(sankey_data, title="Capital Flow Sankey"):
    """Sankey diagram from dict with nodes/links."""
    if not isinstance(sankey_data, dict) or "nodes" not in sankey_data:
        return go.Figure().update_layout(
            title=dict(text="No Sankey data available", font=dict(size=14)),
            template="plotly_white", height=300,
        )
    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                  label=sankey_data["nodes"], color=sankey_data["node_colors"]),
        link=dict(
            source=[l["source"] for l in sankey_data["links"]],
            target=[l["target"] for l in sankey_data["links"]],
            value=[l["value"] for l in sankey_data["links"]],
            color=[l["color"] for l in sankey_data["links"]],
        ),
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=16)),
                      template="plotly_white", height=500, font=dict(size=11))
    return fig


def capital_flow_sankey(flow_series, title="Capital Flow Sankey Diagram"):
    """Sankey diagram showing capital flow changes by asset category."""
    categories = flow_series.index.tolist()
    values = flow_series.values.tolist()
    n = len(categories)
    nodes = ["Pre-FOMC\nPortfolio"] + categories + ["Post-FOMC\nPortfolio"]
    links = []
    for i, (cat, val) in enumerate(zip(categories, values)):
        color = "rgba(39,174,96,0.4)" if val > 0 else "rgba(231,76,60,0.4)"
        links.append({"source": 0, "target": i + 1, "value": abs(val), "color": color})
        links.append({"source": i + 1, "target": n + 1, "value": abs(val), "color": color})
    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                  label=nodes, color=["#2c3e50"] + ["#3498db"] * n + ["#2c3e50"]),
        link=dict(
            source=[l["source"] for l in links],
            target=[l["target"] for l in links],
            value=[l["value"] for l in links],
            color=[l["color"] for l in links],
        ),
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=16)),
                      template="plotly_white", height=500, font=dict(size=11))
    return fig


def correlation_heatmap(corr_matrix, title="Cross-Asset Correlation Matrix"):
    """Heatmap of cross-asset correlations."""
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr_matrix.values.round(2), texttemplate="%{text}", textfont=dict(size=9),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>ρ = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=16)),
                      template="plotly_white", height=max(500, len(corr_matrix) * 35),
                      xaxis_tickangle=-30)
    return fig


def regime_timeline(regime_df, title="Risk Regime Changes Around FOMC"):
    """Timeline showing risk-on/risk-off regime changes."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=regime_df["fomc_date"], y=regime_df["risk_spread_change"] * 100,
        mode="markers+lines", name="Risk Spread Change",
        line=dict(color="#2c3e50", width=1),
        marker=dict(
            color=[COLORS["risk_on"] if v > 0 else COLORS["risk_off"] for v in regime_df["risk_spread_change"]],
            size=10,
            symbol=["triangle-up" if v > 0 else "triangle-down" for v in regime_df["risk_spread_change"]],
        ),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Spread Δ: %{y:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title="Risk-Safe Spread Change (%)", xaxis_title="FOMC Date",
        template="plotly_white", height=400, showlegend=False,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def gap_heatmap(df, title="Research Gap Analysis"):
    """Heatmap showing research gaps."""
    fig = go.Figure(go.Heatmap(
        z=df.values, x=df.columns, y=df.index, colorscale="YlOrRd",
        text=df.values.round(2), texttemplate="%{text}", textfont=dict(size=10),
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=16)),
                      template="plotly_white", height=max(400, len(df) * 40), xaxis_tickangle=-30)
    return fig


def literature_network(nodes, edges, title="Literature Network"):
    """Network graph of papers connected by shared topics."""
    import networkx as nx
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"], **n)
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e["value"])
    pos = nx.spring_layout(G, seed=42, k=2 / np.sqrt(len(nodes)))
    edge_traces = []
    for e in edges:
        i, j = e["source"], e["target"]
        edge_traces.append(go.Scatter(
            x=[pos[i, 0], pos[j, 0]], y=[pos[i, 1], pos[j, 1]],
            mode="lines", line=dict(width=1, color="#CCC"), hoverinfo="none", showlegend=False,
        ))
    node_trace = go.Scatter(
        x=[pos[n["id"], 0] for n in nodes], y=[pos[n["id"], 1] for n in nodes],
        mode="markers+text",
        marker=dict(size=[n["size"] for n in nodes], color=[n["color"] for n in nodes],
                    line=dict(width=1, color="white")),
        text=[n["label"] for n in nodes], textposition="top center", textfont=dict(size=10),
        showlegend=False,
    )
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)), template="plotly_white", height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def communication_timeline(df, title="FOMC Communication Timeline"):
    """Timeline of FOMC events with sentiment."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index, y=df["sentiment_score"],
        marker_color=[
            COLORS["hawkish"] if s < -0.15 else COLORS["dovish"] if s > 0.15 else COLORS["neutral"]
            for s in df["sentiment_score"]
        ],
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title="Sentiment (-1=Hawkish, +1=Dovish)",
        template="plotly_white", height=400, showlegend=False,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def impulse_response(irf_df, title="Impulse Response Functions"):
    """Plot impulse response functions."""
    fig = go.Figure()
    for col in irf_df.columns:
        fig.add_trace(go.Scatter(x=irf_df.index, y=irf_df[col], mode="lines+markers", name=col))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Horizon", yaxis_title="Response",
        template="plotly_white", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.1),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def rebalancing_heatmap(df, title="Portfolio Rebalancing Heatmap"):
    """Heatmap of portfolio weight changes."""
    fig = go.Figure(go.Heatmap(
        z=df.values, x=df.columns, y=df.index, colorscale="RdBu_r", zmid=0,
        text=df.values.round(1), texttemplate="%{text}", textfont=dict(size=9),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white", height=max(400, len(df) * 30), xaxis_tickangle=-30,
    )
    return fig


# ── Phase 1 Research Charts ──

def sentiment_vs_surprise_scatter(df, sentiment_col="sentiment_score",
                                   surprise_col="surprise_bp",
                                   title="FOMC Sentiment vs. Kuttner Surprise"):
    """Scatter plot of sentiment score vs surprise with regression line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[surprise_col], y=df[sentiment_col],
        mode="markers", name="FOMC Meetings",
        marker=dict(size=8, color=df[sentiment_col], colorscale="RdYlGn_r",
                    cmin=-1, cmax=1, line=dict(width=1, color="white")),
        text=[str(d)[:10] for d in df.index],
        hovertemplate="%{text}<br>Surprise: %{x:.1f}bp<br>Sentiment: %{y:.3f}<extra></extra>",
    ))
    # Regression line
    from numpy.polynomial.polynomial import polyfit
    x = df[surprise_col].values
    y = df[sentiment_col].values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() > 2:
        b, m = polyfit(x[mask], y[mask], 1)
        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=b + m * x_line, mode="lines",
            name=f"Fit (R²={np.corrcoef(x[mask], y[mask])[0,1]**2:.3f})",
            line=dict(dash="dash", color=COLORS["neutral"]),
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Kuttner Surprise (bp)", yaxis_title="Sentiment Score",
        template="plotly_white", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.1),
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
    return fig


def sentiment_trajectory_by_chair(df, sentiment_col="sentiment_score",
                                   chair_col="fed_chair",
                                   title="FOMC Sentiment Trajectory by Fed Chair"):
    """Sentiment score time series colored by Fed Chair."""
    if chair_col not in df.columns:
        return sentiment_trajectory(df, title=title)
    
    chair_colors = {
        "Greenspan": "#3498db", "Bernanke": "#e74c3c",
        "Yellen": "#27ae60", "Powell": "#f39c12",
    }
    fig = go.Figure()
    for chair in df[chair_col].unique():
        mask = df[chair_col] == chair
        sub = df[mask].sort_index()
        fig.add_trace(go.Scatter(
            x=sub.index, y=sub[sentiment_col],
            mode="lines+markers", name=chair,
            line=dict(color=chair_colors.get(chair, "#95a5a6"), width=2),
            marker=dict(size=6),
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Date", yaxis_title="Sentiment Score",
        template="plotly_white", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.1),
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    return fig


def incremental_r2_bar(results_dict, title="Incremental R²: Sentiment Beyond Surprise"):
    """Bar chart of incremental R² for each asset."""
    assets = list(results_dict.keys())
    inc_r2 = [results_dict[a]["incremental_r2"] * 100 for a in assets]
    p_vals = [results_dict[a]["p_value"] for a in assets]
    colors = [COLORS["positive"] if p < 0.05 else COLORS["neutral"] for p in p_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=assets, y=inc_r2, marker_color=colors,
        text=[f"{v:.2f}%{'***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''}"
              for v, p in zip(inc_r2, p_vals)],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Asset", yaxis_title="Incremental R² (%)",
        template="plotly_white", height=450,
        xaxis_tickangle=-30,
    )
    return fig


def regression_coefficient_plot(result: dict, title="Regression Coefficients"):
    """Coefficient plot with confidence intervals."""
    if "error" in result:
        return go.Figure().update_layout(
            title=dict(text=f"{title} — Error", font=dict(size=14)),
            template="plotly_white", height=300,
        )
    
    cols = [c for c in result["coefficients"].keys() if c != "const"]
    coefs = [result["coefficients"][c] for c in cols]
    ses = [result["std_errors"][c] for c in cols]
    p_vals = [result["p_values"][c] for c in cols]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cols, y=coefs,
        error_y=dict(type="data", array=[1.96 * s for s in ses]),
        marker_color=[COLORS["positive"] if p < 0.05 else COLORS["neutral"] for p in p_vals],
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white", height=400,
        xaxis_tickangle=-30,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig
