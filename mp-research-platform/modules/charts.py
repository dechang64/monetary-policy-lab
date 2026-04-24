"""Charts — all custom Plotly visualizations for the research platform."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# ─── Color Palette ─────────────────────────────────────────────
COLORS = {
    "S&P 500": "#4C72B0",
    "NASDAQ": "#55A868",
    "2Y Treasury": "#C44E52",
    "10Y Treasury": "#8172B2",
    "DXY": "#CCB974",
    "Gold": "#64B5CD",
    "Oil": "#E5AE38",
    "VIX": "#8C564B",
}


def create_event_study_chart(event_assets, event_name, window_before, window_after):
    """Interactive event study chart with cumulative abnormal returns."""
    fig = go.Figure()

    for asset, df in event_assets.items():
        color = COLORS.get(asset, "#999999")
        fig.add_trace(go.Scatter(
            x=df["minute"],
            y=df["cumulative_return_pct"],
            mode="lines",
            name=asset,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{asset}</b><br>Minute: %{{x}}<br>CAR: %{{y:.3f}}%<extra></extra>",
        ))

    # Event line
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2,
                  annotation_text="FOMC Announcement", annotation_position="top left")

    fig.update_layout(
        title=dict(text=f"Cumulative Abnormal Returns — {event_name}", font=dict(size=16)),
        xaxis_title="Minutes relative to FOMC announcement",
        yaxis_title="Cumulative Return (%)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_shock_radar(event_assets, assets, surprise_bp):
    """Radar chart showing cross-asset impact magnitude."""
    categories = [a for a in assets if a in event_assets]
    if not categories:
        return go.Figure()

    # Compute impact magnitude (absolute CAR at +60min)
    values = []
    for asset in categories:
        df = event_assets[asset]
        car_60 = df.loc[df["minute"] == 60, "cumulative_return_pct"].values
        values.append(abs(car_60[0]) if len(car_60) > 0 else 0)

    # Normalize to 0-10 scale
    max_val = max(values) if max(values) > 0 else 1
    values = [v / max_val * 10 for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(78, 121, 167, 0.3)",
        line=dict(color="#4C72B0", width=2),
        name=f"Surprise: {surprise_bp:+.0f}bp",
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        title=dict(text=f"Cross-Asset Impact Radar (|Surprise| = {abs(surprise_bp):.0f}bp)", font=dict(size=14)),
        height=400,
        showlegend=True,
    )
    return fig


def create_impulse_response(shock_data, assets, horizon):
    """Impulse response functions for policy and information shocks."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Policy Shock → Asset Response", "Information Shock → Asset Response"),
        horizontal_spacing=0.12,
    )

    days = np.arange(0, horizon + 1)

    # Simulated impulse responses based on empirical patterns
    responses = {
        "S&P 500": {"policy": -0.04, "info": 0.02, "decay": 0.85},
        "10Y Treasury": {"policy": -0.008, "info": -0.005, "decay": 0.90},
        "DXY": {"policy": 0.02, "info": 0.01, "decay": 0.88},
        "VIX": {"policy": 0.03, "info": -0.02, "decay": 0.80},
        "Gold": {"policy": -0.015, "info": 0.005, "decay": 0.87},
    }

    for asset in assets:
        if asset not in responses:
            continue
        r = responses[asset]
        color = COLORS.get(asset, "#999")

        # Policy shock IRF
        irf_policy = [r["policy"] * (r["decay"] ** d) + np.random.normal(0, abs(r["policy"]) * 0.1) for d in days]
        fig.add_trace(go.Scatter(
            x=days, y=irf_policy, mode="lines", name=asset,
            line=dict(color=color, width=2), legendgroup=asset, showlegend=True,
        ), row=1, col=1)

        # Information shock IRF
        irf_info = [r["info"] * (r["decay"] ** d) + np.random.normal(0, abs(r["info"]) * 0.1) for d in days]
        fig.add_trace(go.Scatter(
            x=days, y=irf_info, mode="lines", name=asset,
            line=dict(color=color, width=2, dash="dash"), legendgroup=asset, showlegend=False,
        ), row=1, col=2)

    # Zero line
    for col in [1, 2]:
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=col)

    fig.update_layout(
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(title_text="Days after shock")
    fig.update_yaxes(title_text="Response")
    return fig


def create_communication_timeline(nlp_results, stmt1, stmt2):
    """Compare sentiment of two FOMC statements."""
    categories = ["hawkish_score", "dovish_score", "net_tone"]
    labels = ["Hawkish Score", "Dovish Score", "Net Tone (Hawk-Dove)"]

    vals_a = [nlp_results["sentiment_a"][c] for c in categories]
    vals_b = [nlp_results["sentiment_b"][c] for c in categories]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=vals_a, name=stmt1,
        marker_color="#E74C3C", opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=labels, y=vals_b, name=stmt2,
        marker_color="#3498DB", opacity=0.7,
    ))

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        height=350,
        yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_sankey_flow(shock_scenario, investor_type, time_horizon):
    """Sankey diagram showing capital flows between asset classes."""
    # Define flow based on shock scenario
    if "Hawkish" in shock_scenario:
        magnitude = 50 if "+50bp" in shock_scenario else 25
        flows = [
            ("US Equities", "Cash", magnitude * 0.4),
            ("US Equities", "Treasuries", magnitude * 0.3),
            ("HY Bonds", "IG Bonds", magnitude * 0.2),
            ("Intl Equities", "US Equities", magnitude * 0.15),
            ("Gold", "Cash", magnitude * 0.1),
        ]
    elif "Dovish" in shock_scenario:
        magnitude = 50 if "-50bp" in shock_scenario else 25
        flows = [
            ("Cash", "US Equities", magnitude * 0.4),
            ("Treasuries", "US Equities", magnitude * 0.3),
            ("IG Bonds", "HY Bonds", magnitude * 0.2),
            ("US Equities", "Intl Equities", magnitude * 0.15),
            ("Cash", "Gold", magnitude * 0.1),
        ]
    elif "Strong" in shock_scenario:
        flows = [
            ("Cash", "US Equities", 30),
            ("Treasuries", "US Equities", 20),
            ("IG Bonds", "HY Bonds", 15),
            ("Cash", "Real Assets", 10),
        ]
    else:  # Weak Economy
        flows = [
            ("US Equities", "Cash", 35),
            ("HY Bonds", "IG Bonds", 20),
            ("Intl Equities", "Treasuries", 15),
            ("Real Assets", "Cash", 10),
        ]

    # Build Sankey data
    all_nodes = set()
    for src, tgt, val in flows:
        all_nodes.add(src)
        all_nodes.add(tgt)
    node_list = list(all_nodes)
    node_map = {n: i for i, n in enumerate(node_list)}

    sources = [node_map[src] for src, _, _ in flows]
    targets = [node_map[tgt] for _, tgt, _ in flows]
    values = [val for _, _, val in flows]

    # Color: outflows red, inflows green
    node_colors = []
    for n in node_list:
        is_source = any(src == n for src, _, _ in flows)
        is_target = any(tgt == n for _, tgt, _ in flows)
        if is_source and not is_target:
            node_colors.append("#E74C3C")
        elif is_target and not is_source:
            node_colors.append("#3498DB")
        else:
            node_colors.append("#95A5A6")

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=node_list,
            color=node_colors,
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=[f"rgba(78,121,167,{0.3 + v/max(values)*0.5})" for v in values],
        ),
    ))

    fig.update_layout(
        title=dict(text=f"Capital Flow: {shock_scenario} ({investor_type}, {time_horizon})", font=dict(size=14)),
        height=450,
        font=dict(size=11),
    )
    return fig


def create_rebalancing_heatmap(shock_scenario, investor_type):
    """Heatmap showing rebalancing intensity across asset pairs."""
    assets = ["US Equities", "Intl Equities", "IG Bonds", "HY Bonds", "Cash", "Treasuries"]
    n = len(assets)

    # Generate rebalancing matrix
    if "Hawkish" in shock_scenario:
        base_intensity = 0.7 if "+50bp" in shock_scenario else 0.4
    elif "Dovish" in shock_scenario:
        base_intensity = 0.7 if "-50bp" in shock_scenario else 0.4
    else:
        base_intensity = 0.5

    np.random.seed(hash(shock_scenario) % 2**31)
    matrix = np.random.uniform(0, base_intensity, (n, n))
    np.fill_diagonal(matrix, 0)

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=assets,
        y=assets,
        colorscale=[[0, "#FFFFFF"], [0.5, "#F39C12"], [1, "#E74C3C"]],
        text=matrix.round(2),
        texttemplate="%{text}",
        hovertemplate="From: %{y}<br>To: %{x}<br>Flow: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Rebalancing Intensity Matrix — {shock_scenario}", font=dict(size=14)),
        template="plotly_white",
        height=450,
        xaxis_title="To",
        yaxis_title="From",
    )
    return fig


def create_gap_heatmap():
    """Literature gap analysis heatmap: asset class × transmission channel."""
    assets = ["US Equities", "Intl Equities", "EM Equities", "IG Bonds", "HY Bonds", "FX", "Commodities"]
    channels = ["Interest Rate", "Risk-Taking", "Portfolio Balance", "Information", "Expectations", "Liquidity"]

    # Coverage matrix (0=no research, 1=extensive)
    # Based on literature review findings
    coverage = np.array([
        [0.9, 0.6, 0.5, 0.7, 0.9, 0.4],  # US Equities
        [0.5, 0.4, 0.3, 0.5, 0.6, 0.3],  # Intl Equities
        [0.4, 0.3, 0.2, 0.3, 0.4, 0.2],  # EM Equities
        [0.8, 0.3, 0.4, 0.5, 0.7, 0.5],  # IG Bonds
        [0.5, 0.5, 0.3, 0.3, 0.4, 0.3],  # HY Bonds
        [0.6, 0.3, 0.3, 0.4, 0.5, 0.4],  # FX
        [0.3, 0.2, 0.2, 0.2, 0.3, 0.2],  # Commodities
    ])

    fig = go.Figure(go.Heatmap(
        z=coverage,
        x=channels,
        y=assets,
        colorscale=[[0, "#FDEDEC"], [0.3, "#F5B7B1"], [0.6, "#F39C12"], [1, "#27AE60"]],
        text=coverage.round(1),
        texttemplate="%{text}",
        hovertemplate="%{y} × %{x}<br>Coverage: %{z:.1f}<extra></extra>",
        colorbar=dict(title="Research Coverage"),
    ))

    fig.update_layout(
        title=dict(text="Literature Coverage Heatmap — Research Gap Identification", font=dict(size=14)),
        template="plotly_white",
        height=450,
        xaxis_title="Transmission Channel",
        yaxis_title="Asset Class",
        xaxis_tickangle=-30,
    )
    return fig


def create_literature_network(topics, eras):
    """Interactive network graph of related papers."""
    from modules.data_engine import DataEngine
    de = DataEngine()
    nodes, edges = de.get_literature_network_data(topics, eras)

    if not nodes:
        return go.Figure()

    # Build Plotly scatter network
    # Simple force-directed layout approximation
    n = len(nodes)
    np.random.seed(42)
    positions = np.random.randn(n, 2) * 3

    # Simple force simulation (few iterations)
    for _ in range(50):
        for e in edges:
            i, j = e["source"], e["target"]
            diff = positions[j] - positions[i]
            dist = max(np.linalg.norm(diff), 0.1)
            force = (dist - 2) * 0.05
            positions[i] += diff / dist * force
            positions[j] -= diff / dist * force
        # Repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[j] - positions[i]
                dist = max(np.linalg.norm(diff), 0.1)
                if dist < 2:
                    force = (2 - dist) * 0.02
                    positions[i] -= diff / dist * force
                    positions[j] += diff / dist * force

    # Draw edges
    edge_traces = []
    for e in edges:
        i, j = e["source"], e["target"]
        edge_traces.append(go.Scatter(
            x=[positions[i, 0], positions[j, 0]],
            y=[positions[i, 1], positions[j, 1]],
            mode="lines",
            line=dict(width=1, color="#CCCCCC"),
            hoverinfo="none",
            showlegend=False,
        ))

    # Draw nodes
    node_trace = go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode="markers+text",
        marker=dict(
            size=[n["size"] for n in nodes],
            color=[n["color"] for n in nodes],
            line=dict(width=1, color="white"),
        ),
        text=[n["label"] for n in nodes],
        textposition="top center",
        textfont=dict(size=10),
        customdata=[f"{n['title']}<br>Topic: {n['group']}" for n in nodes],
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=dict(text="Literature Network — Papers connected by shared topics & methods", font=dict(size=14)),
        template="plotly_white",
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
    )
    return fig
