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


def event_study_bar(
    df: pd.DataFrame,
    title: str = "Cumulative Abnormal Returns by Asset Class",
) -> go.Figure:
    """Horizontal bar chart of CAR by asset class."""
    df = df.sort_values("avg_CAR_pct")
    
    colors = [
        COLORS["positive"] if v > 0 else COLORS["negative"]
        for v in df["avg_CAR_pct"]
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["asset"],
        x=df["avg_CAR_pct"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}%" for v in df["avg_CAR_pct"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>CAR: %{x:.4f}%<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Cumulative Abnormal Return (%)",
        yaxis_title="",
        template="plotly_white",
        height=max(400, len(df) * 40),
        margin=dict(l=150),
        showlegend=False,
    )
    
    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def event_study_timeline(
    df: pd.DataFrame,
    asset: str,
    title: str = None,
) -> go.Figure:
    """Timeline of CAR for a single asset across FOMC events."""
    asset_df = df[df["asset"] == asset].sort_values("fomc_date")
    
    colors = [
        COLORS["positive"] if v > 0 else COLORS["negative"]
        for v in asset_df["CAR"]
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=asset_df["fomc_date"],
        y=asset_df["CAR"] * 100,
        marker_color=colors,
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>CAR: %{y:.3f}%<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(
            text=title or f"FOMC Event Study: {asset}",
            font=dict(size=16),
        ),
        xaxis_title="FOMC Date",
        yaxis_title="CAR (%)",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def two_shocks_radar(
    response_df: pd.DataFrame,
    title: str = "Two-Shocks Radar: Asset Response by Shock Type",
) -> go.Figure:
    """
    Radar chart showing how each asset responds to policy vs information shocks.
    This is the platform's signature visualization.
    """
    policy_data = response_df[response_df["shock_type"] == "Policy"]
    info_data = response_df[response_df["shock_type"] == "Information"]
    
    categories = policy_data["asset"].tolist()
    policy_vals = policy_data["avg_response_pct"].tolist()
    info_vals = info_data["avg_response_pct"].tolist()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=policy_vals,
        theta=categories,
        fill="toself",
        name="Policy Shock",
        line_color=COLORS["policy"],
        fillcolor=f"rgba(231, 76, 60, 0.15)",
        marker=dict(size=6),
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=info_vals,
        theta=categories,
        fill="toself",
        name="Information Shock",
        line_color=COLORS["information"],
        fillcolor=f"rgba(52, 152, 219, 0.15)",
        marker=dict(size=6),
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, gridcolor=COLORS["grid"])),
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.1),
    )
    
    return fig


def two_shocks_bar(
    response_df: pd.DataFrame,
    title: str = "Asset Response: Policy vs Information Shock",
) -> go.Figure:
    """Grouped bar chart comparing policy vs information shock responses."""
    fig = go.Figure()
    
    policy = response_df[response_df["shock_type"] == "Policy"]
    info = response_df[response_df["shock_type"] == "Information"]
    
    fig.add_trace(go.Bar(
        name="Policy Shock",
        x=policy["asset"],
        y=policy["avg_response_pct"],
        marker_color=COLORS["policy"],
        hovertemplate="<b>%{x}</b><br>Policy: %{y:.3f}%<extra></extra>",
    ))
    
    fig.add_trace(go.Bar(
        name="Information Shock",
        x=info["asset"],
        y=info["avg_response_pct"],
        marker_color=COLORS["information"],
        hovertemplate="<b>%{x}</b><br>Info: %{y:.3f}%<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title="Average Response (%)",
        barmode="group",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.1),
        xaxis_tickangle=-30,
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def sentiment_trajectory(
    sentiment_df: pd.DataFrame,
    title: str = "FOMC Sentiment Trajectory",
) -> go.Figure:
    """Line chart of sentiment scores over time."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sentiment_df.index,
        y=sentiment_df["sentiment_score"],
        mode="lines+markers",
        name="Sentiment Score",
        line=dict(color="#2c3e50", width=2),
        marker=dict(
            color=[
                COLORS["hawkish"] if s < -0.15
                else COLORS["dovish"] if s > 0.15
                else COLORS["neutral"]
                for s in sentiment_df["sentiment_score"]
            ],
            size=8,
        ),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Score: %{y:.3f}<extra></extra>",
    ))
    
    # Add sentiment change arrows
    if "sentiment_change" in sentiment_df.columns:
        changes = sentiment_df["sentiment_change"].dropna()
        for date, change in changes.items():
            if abs(change) > 0.2:
                fig.add_annotation(
                    x=date,
                    y=sentiment_df.loc[date, "sentiment_score"],
                    text=f"{'🔺' if change > 0 else '🔻'} {change:+.2f}",
                    showarrow=False,
                    font=dict(size=10),
                    yshift=15,
                )
    
    # Add regime bands
    fig.add_hrect(
        y0=-1, y1=-0.15, fillcolor=COLORS["hawkish"], opacity=0.05,
        annotation_text="Hawkish Zone", annotation_position="top left",
    )
    fig.add_hrect(
        y0=0.15, y1=1, fillcolor=COLORS["dovish"], opacity=0.05,
        annotation_text="Dovish Zone", annotation_position="top left",
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title="Sentiment Score (-1 = Hawkish, +1 = Dovish)",
        xaxis_title="FOMC Date",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def sankey_diagram(sankey_data: dict, title: str = "Capital Flow Sankey") -> go.Figure:
    """Sankey diagram of capital flows between asset classes."""
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_data["nodes"],
            color=sankey_data["node_colors"],
        ),
        link=dict(
            source=[l["source"] for l in sankey_data["links"]],
            target=[l["target"] for l in sankey_data["links"]],
            value=[l["value"] for l in sankey_data["links"]],
            color=[l["color"] for l in sankey_data["links"]],
            hovertemplate="%{label}<extra></extra>",
        ),
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        height=500,
        font=dict(size=11),
    )
    
    return fig


def correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Cross-Asset Correlation Matrix",
) -> go.Figure:
    """Heatmap of cross-asset correlations."""
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>ρ = %{z:.3f}<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        height=max(500, len(corr_matrix) * 35),
        xaxis_tickangle=-30,
    )
    
    return fig


def regime_timeline(
    regime_df: pd.DataFrame,
    title: str = "Risk Regime Changes Around FOMC",
) -> go.Figure:
    """Timeline showing risk-on/risk-off regime changes."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=regime_df["fomc_date"],
        y=regime_df["risk_spread_change"] * 100,
        mode="markers+lines",
        name="Risk Spread Change",
        line=dict(color="#2c3e50", width=1),
        marker=dict(
            color=[
                COLORS["risk_on"] if v > 0 else COLORS["risk_off"]
                for v in regime_df["risk_spread_change"]
            ],
            size=10,
            symbol=[
                "triangle-up" if v > 0 else "triangle-down"
                for v in regime_df["risk_spread_change"]
            ],
        ),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Spread Δ: %{y:.3f}%<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title="Risk-Safe Spread Change (%)",
        xaxis_title="FOMC Date",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig
