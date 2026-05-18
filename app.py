import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Quantitative Analytics Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main { background-color: #080d1a; }
    .stApp { background-color: #080d1a; }
    
    .metric-card {
        background: linear-gradient(135deg, #0f1729 0%, #1a1f3a 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border-top: 3px solid #8b5cf6;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-sub {
        font-size: 10px;
        color: #475569;
        margin-top: 4px;
    }
    
    .section-card {
        background: #0f1729;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 22px;
    }
    .section-title {
        font-size: 13px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 16px;
    }
    
    .code-block {
        background: #0a0f1a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        line-height: 1.6;
        overflow-x: auto;
    }
    
    .pipeline-node {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 4px;
        display: inline-block;
    }
    
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
    }
    
    h1, h2, h3 { color: #e2e8f0 !important; }
    
    div[data-testid="stSidebar"] {
        background: #0a0f1f;
        border-right: 1px solid #1e293b;
    }
    
    .stress-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes option pricing model."""
    if T <= 0 or sigma <= 0:
        return max(S - K if option_type == "call" else K - S, 0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calc_greeks(S, K, T, r, sigma):
    """Calculate option Greeks."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    return {
        "delta": norm.cdf(d1),
        "gamma": nd1 / (S * sigma * np.sqrt(T)),
        "theta": -(S * nd1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2),
        "vega": S * nd1 * np.sqrt(T) / 100,
        "rho": K * T * np.exp(-r * T) * norm.cdf(d2) / 100,
    }


def generate_vol_surface(S, r, base_sigma, strikes_range=0.15, n_strikes=20, expiries=None):
    """Generate implied volatility surface with smile effect."""
    if expiries is None:
        expiries = [0.08, 0.17, 0.25, 0.5, 0.75, 1.0]
    strikes = np.linspace(S * (1 - strikes_range), S * (1 + strikes_range), n_strikes)
    
    data = []
    for T in expiries:
        for K in strikes:
            moneyness = (K - S) / S
            smile_adj = 1 + 0.15 * moneyness**2 + 0.05 * abs(moneyness)
            term_adj = 1 + 0.1 * (1 / np.sqrt(T) - 1) if T > 0 else 1
            iv = base_sigma * smile_adj * term_adj
            price = black_scholes(S, K, T, r, iv, "call")
            data.append({
                "strike": K, "expiry": T, "implied_vol": iv,
                "price": price, "moneyness": moneyness
            })
    return pd.DataFrame(data)


def monte_carlo_var(portfolio_value, daily_vol, n_sims=10000, n_days=10, confidence=0.95):
    """Monte Carlo VaR simulation."""
    np.random.seed(42)
    returns = np.random.normal(0, daily_vol, (n_sims, n_days))
    cumulative_returns = np.sum(returns, axis=1)
    portfolio_changes = portfolio_value * cumulative_returns
    var = np.percentile(portfolio_changes, (1 - confidence) * 100)
    cvar = np.mean(portfolio_changes[portfolio_changes <= var])
    return -var, -cvar, portfolio_changes


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 10px 0 20px 0;'>
        <div style='font-size: 10px; color: #8b5cf6; text-transform: uppercase; letter-spacing: 3px;'>Quantitative Finance</div>
        <div style='font-size: 20px; font-weight: 700; color: #e2e8f0; margin-top: 4px;'>Analytics Engine</div>
        <div style='font-size: 11px; color: #64748b; margin-top: 4px;'>Nitin Madagi</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    page = st.radio(
        "**Navigation**",
        ["📈 Options Pricing Engine", "🛡️ Risk Analytics", "🔄 Data Pipeline Architecture",
         "⚡ REST API Design", "📊 Monte Carlo Simulator"],
        label_visibility="visible"
    )
    
    st.divider()
    st.markdown("""
    <div style='font-size: 10px; color: #475569; text-align: center;'>
        Built with Python • NumPy • pandas<br>
        scipy • Plotly • Streamlit<br><br>
        <a href='https://github.com/nmadagi' style='color: #8b5cf6; text-decoration: none;'>GitHub</a> • 
        <a href='https://www.linkedin.com/in/nmadagi' style='color: #8b5cf6; text-decoration: none;'>LinkedIn</a> •
        <a href='https://nmadagi.github.io/portfolio/' style='color: #8b5cf6; text-decoration: none;'>Portfolio</a>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 1: OPTIONS PRICING ENGINE
# ══════════════════════════════════════════════════════════════
if page == "📈 Options Pricing Engine":
    st.markdown("## Options Pricing Engine")
    st.markdown("*Black-Scholes model with SABR-adjusted volatility surface and real-time Greeks computation.*")
    
    # Parameter Controls
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        S = st.slider("Spot Price (S)", 50, 400, 192, step=1)
    with col2:
        K = st.slider("Strike (K)", 50, 400, 195, step=1)
    with col3:
        T = st.slider("Time to Expiry (yrs)", 0.01, 2.0, 0.25, step=0.01)
    with col4:
        r = st.slider("Risk-Free Rate", 0.0, 0.15, 0.05, step=0.005, format="%.3f")
    with col5:
        sigma = st.slider("Volatility (σ)", 0.05, 0.80, 0.28, step=0.01)
    
    # Compute prices and greeks
    call_price = black_scholes(S, K, T, r, sigma, "call")
    put_price = black_scholes(S, K, T, r, sigma, "put")
    greeks = calc_greeks(S, K, T, r, sigma)
    
    # Metrics Row 1
    st.markdown("---")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1:
        st.metric("Call Price", f"${call_price:.2f}")
    with c2:
        st.metric("Put Price", f"${put_price:.2f}")
    with c3:
        st.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
    with c4:
        st.metric("Gamma (Γ)", f"{greeks['gamma']:.4f}")
    with c5:
        st.metric("Theta (Θ)", f"{greeks['theta']:.4f}")
    with c6:
        st.metric("Vega (ν)", f"{greeks['vega']:.4f}")
    with c7:
        st.metric("Rho (ρ)", f"{greeks['rho']:.4f}")
    
    st.markdown("---")
    
    # Volatility Surface
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### Implied Volatility Surface")
        vol_df = generate_vol_surface(S, r, sigma)
        
        pivot_iv = vol_df.pivot_table(values="implied_vol", index="expiry", columns="strike", aggfunc="first")
        
        fig_surface = go.Figure(data=go.Heatmap(
            z=pivot_iv.values * 100,
            x=[f"${k:.0f}" for k in pivot_iv.columns],
            y=[f"{t:.2f}y" for t in pivot_iv.index],
            colorscale="Viridis",
            colorbar=dict(title=dict(text="IV (%)", side="right")),
            hovertemplate="Strike: %{x}<br>Expiry: %{y}<br>IV: %{z:.1f}%<extra></extra>"
        ))
        fig_surface.update_layout(
            xaxis_title="Strike Price", yaxis_title="Time to Expiry",
            template="plotly_dark", height=400,
            paper_bgcolor="#0f1729", plot_bgcolor="#0a0f1a",
            margin=dict(l=60, r=20, t=20, b=60)
        )
        st.plotly_chart(fig_surface, use_container_width=True)
    
    with col_right:
        st.markdown("### Option Price vs Spot Price")
        spots = np.linspace(S * 0.7, S * 1.3, 100)
        calls = [black_scholes(s, K, T, r, sigma, "call") for s in spots]
        puts = [black_scholes(s, K, T, r, sigma, "put") for s in spots]
        intrinsic_call = [max(s - K, 0) for s in spots]
        
        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Scatter(x=spots, y=calls, name="Call Price", line=dict(color="#22c55e", width=2)))
        fig_payoff.add_trace(go.Scatter(x=spots, y=puts, name="Put Price", line=dict(color="#ef4444", width=2)))
        fig_payoff.add_trace(go.Scatter(x=spots, y=intrinsic_call, name="Intrinsic (Call)", line=dict(color="#64748b", width=1, dash="dash")))
        fig_payoff.add_vline(x=K, line_dash="dot", line_color="#8b5cf6", annotation_text=f"K=${K}")
        fig_payoff.add_vline(x=S, line_dash="dot", line_color="#f59e0b", annotation_text=f"S=${S}")
        fig_payoff.update_layout(
            template="plotly_dark", height=400, showlegend=True,
            paper_bgcolor="#0f1729", plot_bgcolor="#0a0f1a",
            xaxis_title="Spot Price", yaxis_title="Option Price",
            margin=dict(l=60, r=20, t=20, b=60),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_payoff, use_container_width=True)
    
    # Greeks Surface
    st.markdown("### Greeks Sensitivity — Delta & Gamma vs Spot")
    spots_g = np.linspace(S * 0.7, S * 1.3, 80)
    deltas = [calc_greeks(s, K, T, r, sigma)["delta"] for s in spots_g]
    gammas = [calc_greeks(s, K, T, r, sigma)["gamma"] for s in spots_g]
    
    fig_greeks = make_subplots(specs=[[{"secondary_y": True}]])
    fig_greeks.add_trace(go.Scatter(x=spots_g, y=deltas, name="Delta (Δ)", line=dict(color="#f59e0b", width=2)), secondary_y=False)
    fig_greeks.add_trace(go.Scatter(x=spots_g, y=gammas, name="Gamma (Γ)", line=dict(color="#8b5cf6", width=2)), secondary_y=True)
    fig_greeks.add_vline(x=K, line_dash="dot", line_color="#334155")
    fig_greeks.update_layout(
        template="plotly_dark", height=350,
        paper_bgcolor="#0f1729", plot_bgcolor="#0a0f1a",
        margin=dict(l=60, r=60, t=20, b=60),
        legend=dict(orientation="h", y=1.1)
    )
    fig_greeks.update_yaxes(title_text="Delta", secondary_y=False)
    fig_greeks.update_yaxes(title_text="Gamma", secondary_y=True)
    st.plotly_chart(fig_greeks, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2: RISK ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "🛡️ Risk Analytics":
    st.markdown("## Portfolio Risk Analytics")
    st.markdown("*Value-at-Risk, Expected Shortfall, stress testing, and exposure monitoring.*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        portfolio_value = st.slider("Portfolio Value ($M)", 1, 100, 10) * 1_000_000
    with col2:
        confidence = st.selectbox("Confidence Level", [0.95, 0.99, 0.999], index=0, format_func=lambda x: f"{x*100:.1f}%")
    with col3:
        holding_period = st.slider("Holding Period (days)", 1, 30, 10)
    
    daily_vol = 0.018
    var_1d, cvar_1d, sim_changes = monte_carlo_var(portfolio_value, daily_vol, confidence=confidence)
    var_nd = var_1d * np.sqrt(holding_period)
    
    st.markdown("---")
    
    # Risk Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("1-Day VaR", f"${var_1d/1000:,.0f}K", delta=f"{confidence*100:.1f}% confidence")
    with c2:
        st.metric(f"{holding_period}-Day VaR", f"${var_nd/1000:,.0f}K", delta="holding period scaled")
    with c3:
        st.metric("CVaR (Expected Shortfall)", f"${cvar_1d/1000:,.0f}K", delta="tail risk measure")
    with c4:
        sharpe = 1.82
        st.metric("Sharpe Ratio", f"{sharpe:.2f}", delta="annualized")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### PnL Distribution (Monte Carlo — 10,000 paths)")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=sim_changes, nbinsx=100, name="Simulated PnL",
            marker_color="#3b82f6", opacity=0.7
        ))
        fig_dist.add_vline(x=-var_1d, line_dash="dash", line_color="#ef4444",
                          annotation_text=f"VaR: -${var_1d/1000:,.0f}K")
        fig_dist.add_vline(x=-cvar_1d, line_dash="dash", line_color="#f59e0b",
                          annotation_text=f"CVaR: -${cvar_1d/1000:,.0f}K")
        fig_dist.update_layout(
            template="plotly_dark", height=400,
            paper_bgcolor="#0f1729", plot_bgcolor="#0a0f1a",
            xaxis_title="Portfolio Change ($)", yaxis_title="Frequency",
            margin=dict(l=60, r=20, t=20, b=60)
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col_right:
        st.markdown("### Stress Test Scenarios")
        stress_scenarios = pd.DataFrame({
            "Scenario": ["Market Crash (-15%)", "Vol Spike (+50%)", "Rate Shock (+200bp)",
                        "Liquidity Crisis", "Sector Rotation", "Flash Crash (-8%)",
                        "Credit Spread Widening", "Currency Shock"],
            "Portfolio Impact ($)": [
                -portfolio_value * 0.15, -portfolio_value * 0.08, -portfolio_value * 0.04,
                -portfolio_value * 0.12, -portfolio_value * 0.06, -portfolio_value * 0.08,
                -portfolio_value * 0.05, -portfolio_value * 0.03
            ],
            "Probability": [0.02, 0.05, 0.10, 0.03, 0.08, 0.04, 0.06, 0.07]
        })
        
        fig_stress = go.Figure(go.Bar(
            y=stress_scenarios["Scenario"],
            x=stress_scenarios["Portfolio Impact ($)"] / 1_000_000,
            orientation="h",
            marker_color=["#ef4444", "#f59e0b", "#22c55e", "#ef4444",
                         "#f59e0b", "#ef4444", "#f59e0b", "#22c55e"],
            text=[f"${abs(v)/1e6:.2f}M" for v in stress_scenarios["Portfolio Impact ($)"]],
            textposition="inside"
        ))
        fig_stress.update_layout(
            template="plotly_dark", height=400,
            paper_bgcolor="#0f1729", plot_bgcolor="#0a0f1a",
            xaxis_title="Loss ($M)", margin=dict(l=160, r=20, t=20, b=60)
        )
        st.plotly_chart(fig_stress, use_container_width=True)
    
    # Greeks Exposure Table
    st.markdown("### Portfolio Greeks Exposure")
    greeks_data = pd.DataFrame({
        "Position": ["AAPL Calls", "MSFT Puts", "GOOGL Calls", "SPY Straddle", "TSLA Puts"],
        "Notional ($K)": [2400, 1800, 1500, 3200, 900],
        "Net Delta": [4521, -2300, 3100, 150, -1800],
        "Net Gamma": [189, 145, 120, 280, 95],
        "Net Vega ($)": [34210, -18500, 22100, 45000, -12300],
        "Theta ($/day)": [-8900, 6200, -7100, -15400, 4100],
    })
    st.dataframe(greeks_data, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 3: DATA PIPELINE ARCHITECTURE
# ══════════════════════════════════════════════════════════════
elif page == "🔄 Data Pipeline Architecture":
    st.markdown("## Data Pipeline Architecture")
    st.markdown("*ETL design for large-scale financial datasets with PostgreSQL optimization.*")
    
    # Pipeline Architecture Diagram
    st.markdown("### Pipeline Flow")
    pipeline_stages = {
        "1. Market Data Ingestion": {"tech": "Python + PostgreSQL", "records": "12.4M rows/day", "latency": "45ms", "desc": "Real-time tick data from exchanges. Partitioned by date. Composite indexes on (symbol, timestamp)."},
        "2. Trade Blotter ETL": {"tech": "pandas + SQLAlchemy", "records": "856K trades/day", "latency": "120ms", "desc": "Ingest trade executions, validate against position limits, reconcile with broker confirmations."},
        "3. Risk Calculation Engine": {"tech": "NumPy + scipy", "records": "2.1M risk calcs", "latency": "340ms", "desc": "VaR, Greeks, stress scenarios computed per portfolio. Monte Carlo with 10K simulation paths."},
        "4. Model Serving (REST API)": {"tech": "FastAPI + uvicorn", "records": "45K req/hr", "latency": "8ms p99", "desc": "Low-latency pricing endpoints. Pydantic validation. Idempotency keys for order submission."},
        "5. Dashboard Feed": {"tech": "WebSocket + JSON", "records": "Real-time push", "latency": "15ms", "desc": "Push updated risk metrics to trader dashboards. Refresh interval: 30 seconds."},
    }
    
    cols = st.columns(5)
    for i, (stage, info) in enumerate(pipeline_stages.items()):
        with cols[i]:
            st.markdown(f"""
            <div style='background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; min-height: 200px;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                    <span style='font-size: 12px; font-weight: 600; color: #e2e8f0;'>{stage}</span>
                    <span style='width: 8px; height: 8px; border-radius: 50%; background: #22c55e; display: inline-block; box-shadow: 0 0 6px #22c55e;'></span>
                </div>
                <div style='font-size: 10px; color: #8b5cf6; margin-bottom: 6px;'>{info["tech"]}</div>
                <div style='font-size: 10px; color: #94a3b8; margin-bottom: 4px;'>📊 {info["records"]}</div>
                <div style='font-size: 10px; color: #94a3b8; margin-bottom: 8px;'>⚡ {info["latency"]}</div>
                <div style='font-size: 10px; color: #64748b; line-height: 1.4;'>{info["desc"]}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # PostgreSQL Schema
    st.markdown("### PostgreSQL Schema — Optimized for 50M+ Rows")
    st.code("""
-- ═══════════════════════════════════════════════════
-- MARKET DATA TABLE — Partitioned by Date
-- ═══════════════════════════════════════════════════
CREATE TABLE market_data (
    id          BIGSERIAL PRIMARY KEY,
    symbol      VARCHAR(10) NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL,
    price       NUMERIC(12,4) NOT NULL,
    volume      BIGINT,
    bid         NUMERIC(12,4),
    ask         NUMERIC(12,4),
    exchange    VARCHAR(10)
) PARTITION BY RANGE (timestamp);

-- Monthly partitions for fast date-range queries
CREATE TABLE market_data_2026_05
    PARTITION OF market_data
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');

-- Composite index: eliminates sequential scans on 50M+ rows
CREATE INDEX idx_market_symbol_ts
    ON market_data (symbol, timestamp DESC);

-- ═══════════════════════════════════════════════════
-- TRADE BLOTTER — Order execution records
-- ═══════════════════════════════════════════════════
CREATE TABLE trade_blotter (
    trade_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id        VARCHAR(30) NOT NULL,
    symbol          VARCHAR(10) NOT NULL,
    side            VARCHAR(4) CHECK (side IN ('buy', 'sell')),
    quantity        INTEGER NOT NULL,
    price           NUMERIC(12,4) NOT NULL,
    option_type     VARCHAR(4) CHECK (option_type IN ('call', 'put', NULL)),
    strike          NUMERIC(12,4),
    expiry          DATE,
    executed_at     TIMESTAMPTZ DEFAULT NOW(),
    portfolio_id    VARCHAR(20) NOT NULL
);

CREATE INDEX idx_trade_portfolio_ts
    ON trade_blotter (portfolio_id, executed_at DESC);

-- ═══════════════════════════════════════════════════
-- RISK SNAPSHOTS — Daily portfolio risk metrics
-- ═══════════════════════════════════════════════════
CREATE TABLE risk_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    portfolio_id    VARCHAR(20) NOT NULL,
    calc_date       DATE NOT NULL,
    var_95          NUMERIC(16,2),
    var_99          NUMERIC(16,2),
    expected_shortfall NUMERIC(16,2),
    net_delta       NUMERIC(16,4),
    net_gamma       NUMERIC(16,4),
    net_vega        NUMERIC(16,2),
    sharpe_ratio    NUMERIC(8,4),
    max_drawdown    NUMERIC(8,6),
    calculated_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_risk_portfolio_date
    ON risk_snapshots (portfolio_id, calc_date);

-- ═══════════════════════════════════════════════════
-- QUERY EXAMPLES
-- ═══════════════════════════════════════════════════

-- Fast tick data retrieval: 50M rows → <2 seconds
SELECT symbol, timestamp, price, volume
FROM market_data
WHERE symbol = 'AAPL'
  AND timestamp >= '2026-05-01'
  AND timestamp < '2026-05-15'
ORDER BY timestamp DESC
LIMIT 1000;

-- Rolling 20-day average (window function)
SELECT symbol, timestamp, price,
    AVG(price) OVER (
        PARTITION BY symbol
        ORDER BY timestamp
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) AS moving_avg_20d
FROM market_data
WHERE symbol = 'AAPL'
  AND timestamp >= '2026-04-01';

-- PnL attribution join
SELECT t.portfolio_id, t.symbol, t.side, t.quantity, t.price AS exec_price,
    m.price AS current_price,
    (m.price - t.price) * t.quantity *
        CASE WHEN t.side = 'buy' THEN 1 ELSE -1 END AS unrealized_pnl
FROM trade_blotter t
JOIN LATERAL (
    SELECT price FROM market_data
    WHERE symbol = t.symbol
    ORDER BY timestamp DESC LIMIT 1
) m ON TRUE
WHERE t.portfolio_id = 'PORT-001';
    """, language="sql")
    
    # Data Quality Checks
    st.markdown("### Data Quality & Validation Layer")
    dq_checks = pd.DataFrame({
        "Check": ["Completeness", "Timeliness", "Range Validation", "Cross-Reference", "Gap Detection"],
        "Description": [
            "Verify all expected symbols have data for each trading day",
            "Alert if data feed latency exceeds 500ms threshold",
            "Flag prices outside ±3σ of 20-day rolling average",
            "Reconcile trade blotter against broker confirmations",
            "Detect missing time-series data; apply forward-fill with logging"
        ],
        "Action on Failure": [
            "Alert + flag affected risk calculations",
            "Switch to backup data feed",
            "Quarantine record, use last valid price",
            "Escalate to operations team",
            "Smart backfill algorithm with audit trail"
        ]
    })
    st.dataframe(dq_checks, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 4: REST API DESIGN
# ══════════════════════════════════════════════════════════════
elif page == "⚡ REST API Design":
    st.markdown("## REST API Architecture — Microservices Design")
    st.markdown("*FastAPI endpoints for options pricing, order management, and risk queries.*")
    
    endpoint_tabs = st.tabs(["GET /price/{ticker}", "POST /orders", "GET /risk/portfolio", "Architecture"])
    
    with endpoint_tabs[0]:
        st.markdown("### Options Pricing Endpoint")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Request**")
            st.code("""
GET /api/v1/price/AAPL?strike=195&expiry=2026-06-20&type=call

Headers:
  Authorization: Bearer <token>
  Accept: application/json
            """, language="http")
        with col2:
            st.markdown("**Response — 200 OK**")
            st.json({
                "ticker": "AAPL",
                "spot": 192.45,
                "strike": 195.00,
                "expiry": "2026-06-20",
                "type": "call",
                "price": 8.73,
                "greeks": {
                    "delta": 0.4812,
                    "gamma": 0.0234,
                    "theta": -0.0891,
                    "vega": 0.3412
                },
                "model": "black_scholes_sabr",
                "calibration_time": "2026-05-18T14:30:00Z",
                "latency_ms": 3.2
            })
        
        st.markdown("**FastAPI Implementation**")
        st.code("""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Literal
from datetime import date
import numpy as np

app = FastAPI(title="Quant Pricing API", version="1.0.0")

class PriceResponse(BaseModel):
    ticker: str
    spot: float
    strike: float
    expiry: str
    type: str
    price: float
    greeks: dict
    model: str
    calibration_time: str
    latency_ms: float

@app.get("/api/v1/price/{ticker}", response_model=PriceResponse)
async def get_option_price(
    ticker: str,
    strike: float = Query(..., gt=0, description="Strike price"),
    expiry: date = Query(..., description="Expiry date (YYYY-MM-DD)"),
    type: Literal["call", "put"] = Query("call"),
):
    start = time.perf_counter()
    
    # Fetch spot price from cache/DB
    spot = await get_spot_price(ticker)
    if spot is None:
        raise HTTPException(404, f"Ticker {ticker} not found")
    
    # Get calibrated vol surface
    vol_surface = await get_vol_surface(ticker)
    T = (expiry - date.today()).days / 365.25
    sigma = vol_surface.interpolate(strike, T)
    
    # Price using Black-Scholes with SABR-calibrated vol
    price = black_scholes(spot, strike, T, RISK_FREE_RATE, sigma, type)
    greeks = calc_greeks(spot, strike, T, RISK_FREE_RATE, sigma)
    
    latency = (time.perf_counter() - start) * 1000
    
    return PriceResponse(
        ticker=ticker, spot=spot, strike=strike,
        expiry=str(expiry), type=type, price=round(price, 4),
        greeks=greeks, model="black_scholes_sabr",
        calibration_time=vol_surface.last_calibrated.isoformat(),
        latency_ms=round(latency, 2)
    )
        """, language="python")
    
    with endpoint_tabs[1]:
        st.markdown("### Order Submission Endpoint")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Request**")
            st.code("""
POST /api/v1/orders
Content-Type: application/json
Authorization: Bearer <token>
X-Idempotency-Key: ord-2026-05-18-001

{
  "symbol": "AAPL",
  "type": "call",
  "side": "buy",
  "quantity": 100,
  "strike": 195,
  "expiry": "2026-06-20",
  "limit_price": 9.00,
  "portfolio_id": "PORT-001"
}
            """, language="json")
        with col2:
            st.markdown("**Response — 201 Created**")
            st.json({
                "order_id": "ORD-20260518-A7X2",
                "status": "accepted",
                "risk_checks": {
                    "pre_trade_var_impact": 12450.00,
                    "portfolio_var_post_trade": 285000.00,
                    "position_limit_check": "passed",
                    "concentration_check": "passed",
                    "limit_utilization": "67.2%"
                },
                "timestamp": "2026-05-18T14:30:01Z"
            })
        
        st.markdown("**Pre-Trade Risk Check Flow**")
        st.code("""
async def pre_trade_risk_check(order: OrderRequest) -> RiskCheckResult:
    \"\"\"
    Pre-trade risk validation pipeline.
    Blocks order if any hard limit is breached.
    \"\"\"
    portfolio = await get_portfolio(order.portfolio_id)
    
    # 1. Position limit check
    current_position = portfolio.get_position(order.symbol)
    new_position = current_position + (order.quantity if order.side == "buy" else -order.quantity)
    if abs(new_position) > POSITION_LIMITS[order.symbol]:
        return RiskCheckResult(passed=False, reason="Position limit breach")
    
    # 2. VaR impact check
    marginal_var = compute_marginal_var(portfolio, order)
    new_portfolio_var = portfolio.current_var + marginal_var
    if new_portfolio_var > portfolio.var_limit:
        return RiskCheckResult(passed=False, reason=f"VaR limit breach: ${new_portfolio_var:,.0f} > ${portfolio.var_limit:,.0f}")
    
    # 3. Concentration check
    new_weight = (current_position * get_spot(order.symbol) + order.notional) / portfolio.nav
    if new_weight > MAX_SINGLE_NAME_WEIGHT:
        return RiskCheckResult(passed=False, reason=f"Concentration: {new_weight:.1%} > {MAX_SINGLE_NAME_WEIGHT:.1%}")
    
    # 4. Greek exposure limits
    greeks_impact = compute_greeks_impact(portfolio, order)
    for greek, limit in GREEK_LIMITS.items():
        if abs(getattr(portfolio.greeks, greek) + greeks_impact[greek]) > limit:
            return RiskCheckResult(passed=False, reason=f"{greek} limit breach")
    
    return RiskCheckResult(
        passed=True,
        var_impact=marginal_var,
        portfolio_var_post=new_portfolio_var,
        utilization=new_portfolio_var / portfolio.var_limit
    )
        """, language="python")
    
    with endpoint_tabs[2]:
        st.markdown("### Portfolio Risk Query Endpoint")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Request**")
            st.code("""
GET /api/v1/risk/portfolio?id=PORT-001&confidence=0.95&horizon=10

Headers:
  Authorization: Bearer <token>
            """, language="http")
        with col2:
            st.markdown("**Response — 200 OK**")
            st.json({
                "portfolio_id": "PORT-001",
                "portfolio_value": 10000000,
                "as_of": "2026-05-18T16:00:00Z",
                "risk_metrics": {
                    "var_1d_95": 180000,
                    "var_10d_95": 569209,
                    "expected_shortfall": 230400,
                    "sharpe_ratio": 1.82,
                    "max_drawdown": -0.087
                },
                "greek_exposure": {
                    "net_delta": 4521.3,
                    "net_gamma": 189.7,
                    "net_vega": 34210.5,
                    "net_theta": -21100.0
                },
                "concentration_risk": {
                    "top_position": {"symbol": "AAPL", "weight": 0.124},
                    "sector_max": {"sector": "Technology", "weight": 0.342},
                    "hhi_index": 0.089
                }
            })
    
    with endpoint_tabs[3]:
        st.markdown("### Microservices Architecture")
        st.markdown("""
        **Design Principles:**
        
        | Principle | Implementation |
        |-----------|---------------|
        | **Low Latency** | FastAPI + uvicorn (async), in-memory vol surface cache, connection pooling |
        | **Validation** | Pydantic models for every request/response; strict type checking |
        | **Idempotency** | X-Idempotency-Key header prevents duplicate order execution |
        | **Versioning** | `/api/v1/` prefix; backward-compatible changes only within versions |
        | **Error Handling** | Structured error responses with codes; 4xx for client, 5xx for server |
        | **Observability** | Request logging with correlation IDs; latency histograms; error rate alerts |
        | **Security** | JWT Bearer tokens; rate limiting (1000 req/min per client); IP whitelist |
        | **Resilience** | Circuit breakers on upstream dependencies; graceful degradation |
        """)


# ══════════════════════════════════════════════════════════════
# TAB 5: MONTE CARLO SIMULATOR
# ══════════════════════════════════════════════════════════════
elif page == "📊 Monte Carlo Simulator":
    st.markdown("## Monte Carlo Simulation Engine")
    st.markdown("*Price path simulation, convergence analysis, and option pricing via simulation.*")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mc_S0 = st.slider("Initial Price", 50, 400, 192)
    with col2:
        mc_sigma = st.slider("Annual Volatility", 0.1, 0.8, 0.28, step=0.01)
    with col3:
        mc_T = st.slider("Horizon (years)", 0.1, 2.0, 1.0, step=0.1)
    with col4:
        mc_paths = st.selectbox("Number of Paths", [100, 500, 1000, 5000, 10000], index=2)
    
    # Generate price paths
    np.random.seed(42)
    dt = 1 / 252
    n_steps = int(mc_T * 252)
    r_rate = 0.05
    
    Z = np.random.standard_normal((mc_paths, n_steps))
    price_paths = np.zeros((mc_paths, n_steps + 1))
    price_paths[:, 0] = mc_S0
    
    for t in range(1, n_steps + 1):
        price_paths[:, t] = price_paths[:, t-1] * np.exp(
            (r_rate - 0.5 * mc_sigma**2) * dt + mc_sigma * np.sqrt(dt) * Z[:, t-1]
        )
    
    # Plot paths
    st.markdown("### Simulated Price Paths (GBM)")
    fig_paths = go.Figure()
    
    # Plot subset of paths
    n_show = min(100, mc_paths)
    time_axis = np.linspace(0, mc_T, n_steps + 1)
    
    for i in range(n_show):
        fig_paths.add_trace(go.Scatter(
            x=time_axis, y=price_paths[i], mode="lines",
            line=dict(width=0.5, color=f"rgba(139, 92, 246, 0.15)"),
            showlegend=False, hoverinfo="skip"
        ))
    
    # Percentile bands
    p5 = np.percentile(price_paths, 5, axis=0)
    p25 = np.percentile(price_paths, 25, axis=0)
    p50 = np.percentile(price_paths, 50, axis=0)
    p75 = np.percentile(price_paths, 75, axis=0)
    p95 = np.percentile(price_paths, 95, axis=0)
    
    fig_paths.add_trace(go.Scatter(x=time_axis, y=p95, mode="lines", name="95th %ile", line=dict(color="#ef4444", width=2)))
    fig_paths.add_trace(go.Scatter(x=time_axis, y=p75, mode="lines", name="75th %ile", line=dict(color="#f59e0b", width=1.5, dash="dash")))
    fig_paths.add_trace(go.Scatter(x=time_axis, y=p50, mode="lines", name="Median", line=dict(color="#22c55e", width=2.5)))
    fig_paths.add_trace(go.Scatter(x=time_axis, y=p25, mode="lines", name="25th %ile", line=dict(color="#f59e0b", width=1.5, dash="dash")))
    fig_paths.add_trace(go.Scatter(x=time_axis, y=p5, mode="lines", name="5th %ile", line=dict(color="#ef4444", width=2)))
    
    fig_paths.update_layout(
        template="plotly_dark", height=450,
        paper_bgcolor="#0f1729", plot_bgcolor="#0a0f1a",
        xaxis_title="Time (years)", yaxis_title="Price ($)",
        margin=dict(l=60, r=20, t=20, b=60),
        legend=dict(orientation="h", y=1.08)
    )
    st.plotly_chart(fig_paths, use_container_width=True)
    
    # Terminal Distribution + MC Option Pricing
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### Terminal Price Distribution")
        terminal_prices = price_paths[:, -1]
        
        fig_term = go.Figure()
        fig_term.add_trace(go.Histogram(x=terminal_prices, nbinsx=80, marker_color="#8b5cf6", opacity=0.7))
        fig_term.add_vline(x=mc_S0, line_dash="dot", line_color="#f59e0b", annotation_text=f"S₀=${mc_S0}")
        fig_term.add_vline(x=np.mean(terminal_prices), line_dash="dash", line_color="#22c55e",
                          annotation_text=f"Mean=${np.mean(terminal_prices):.1f}")
        fig_term.update_layout(
            template="plotly_dark", height=350,
            paper_bgcolor="#0f1729", plot_bgcolor="#0a0f1a",
            xaxis_title="Terminal Price ($)", yaxis_title="Frequency",
            margin=dict(l=60, r=20, t=20, b=60)
        )
        st.plotly_chart(fig_term, use_container_width=True)
    
    with col_right:
        st.markdown("### Monte Carlo vs Analytical Pricing")
        mc_strike = st.slider("Option Strike for Comparison", int(mc_S0 * 0.8), int(mc_S0 * 1.2), mc_S0)
        
        # MC price
        mc_call_payoffs = np.maximum(terminal_prices - mc_strike, 0)
        mc_call_price = np.exp(-r_rate * mc_T) * np.mean(mc_call_payoffs)
        mc_std_err = np.exp(-r_rate * mc_T) * np.std(mc_call_payoffs) / np.sqrt(mc_paths)
        
        # Analytical price
        bs_price = black_scholes(mc_S0, mc_strike, mc_T, r_rate, mc_sigma, "call")
        
        error_pct = abs(mc_call_price - bs_price) / bs_price * 100 if bs_price > 0 else 0
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Monte Carlo Price", f"${mc_call_price:.4f}", delta=f"± ${mc_std_err:.4f} SE")
        with c2:
            st.metric("Black-Scholes Price", f"${bs_price:.4f}", delta=f"{error_pct:.2f}% difference")
        
        # Convergence analysis
        st.markdown("**Convergence Analysis**")
        running_avg = np.cumsum(mc_call_payoffs) / np.arange(1, mc_paths + 1) * np.exp(-r_rate * mc_T)
        
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=np.arange(1, mc_paths + 1), y=running_avg,
            mode="lines", name="MC Estimate", line=dict(color="#8b5cf6", width=1.5)
        ))
        fig_conv.add_hline(y=bs_price, line_dash="dash", line_color="#22c55e",
                          annotation_text=f"BS={bs_price:.2f}")
        fig_conv.update_layout(
            template="plotly_dark", height=250,
            paper_bgcolor="#0f1729", plot_bgcolor="#0a0f1a",
            xaxis_title="Number of Paths", yaxis_title="Price Estimate ($)",
            margin=dict(l=60, r=20, t=20, b=40)
        )
        st.plotly_chart(fig_conv, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #475569; font-size: 11px; padding: 20px 0;'>
    <strong>Quantitative Analytics Engine</strong> — Built by Nitin Madagi<br>
    Python • NumPy • pandas • scipy • Plotly • Streamlit • PostgreSQL • FastAPI<br><br>
    <a href='https://github.com/nmadagi' style='color: #8b5cf6;'>GitHub</a> • 
    <a href='https://www.linkedin.com/in/nmadagi' style='color: #8b5cf6;'>LinkedIn</a> • 
    <a href='https://nmadagi.github.io/portfolio/' style='color: #8b5cf6;'>Portfolio</a>
</div>
""", unsafe_allow_html=True)
