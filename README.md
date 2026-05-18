# Quantitative Analytics Engine

A production-grade quantitative finance analytics platform built with Python, featuring options pricing, portfolio risk management, data pipeline architecture, and REST API design for capital markets applications.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## 🔗 Live Demo

**[▶ Launch App](https://quant-analytics-engine.streamlit.app)** *(update this URL after deployment)*

---

## Features

### 📈 Options Pricing Engine
- Black-Scholes model with SABR-adjusted volatility surface
- Real-time Greeks computation (Delta, Gamma, Theta, Vega, Rho)
- Interactive volatility surface heatmap
- Option price vs spot price visualization
- Greeks sensitivity analysis

### 🛡️ Risk Analytics
- Value-at-Risk (VaR) — Historical, Parametric, and Monte Carlo methods
- Conditional VaR / Expected Shortfall
- Stress testing across 8 macroeconomic scenarios
- Portfolio Greeks exposure monitoring
- Position-level risk attribution

### 🔄 Data Pipeline Architecture
- ETL design for 50M+ row financial datasets
- PostgreSQL schema with date partitioning and composite indexes
- Trade blotter, market data, and risk snapshot tables
- Data quality validation framework with smart backfill
- Query optimization patterns (window functions, LATERAL joins)

### ⚡ REST API Design
- FastAPI microservice architecture for pricing and order management
- Pre-trade risk check pipeline (VaR impact, position limits, concentration, Greeks)
- Pydantic request/response validation
- Idempotency keys for order deduplication
- Full endpoint documentation with request/response examples

### 📊 Monte Carlo Simulator
- Geometric Brownian Motion price path simulation (up to 10,000 paths)
- Percentile band visualization (5th, 25th, 50th, 75th, 95th)
- Terminal price distribution analysis
- Monte Carlo vs Black-Scholes convergence comparison
- Standard error and convergence analysis

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Computation** | Python, NumPy, pandas, scipy |
| **Visualization** | Plotly, Streamlit |
| **Database Design** | PostgreSQL (schema, partitioning, indexing) |
| **API Framework** | FastAPI, Pydantic, uvicorn |
| **Models** | Black-Scholes, SABR, Monte Carlo, VaR/CVaR |

---

## Getting Started

```bash
# Clone
git clone https://github.com/nmadagi/quant-analytics-engine.git
cd quant-analytics-engine

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

---

## Project Structure

```
quant-analytics-engine/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit theme configuration
└── README.md              # This file
```

---

## Key Technical Highlights

- **Vectorized computation**: All pricing and risk calculations use NumPy vectorization for performance
- **Production-grade schema**: PostgreSQL design handles 50M+ rows with partitioning and composite indexes
- **Pre-trade risk gates**: Order submission includes VaR impact, position limits, concentration, and Greeks exposure checks
- **Monte Carlo convergence**: Demonstrates MC pricing converging to analytical Black-Scholes as path count increases
- **Clean OOP design**: Pricing models follow the Strategy pattern for easy extensibility

---

## Author

**Nitin Madagi** — Quantitative Risk & Financial Engineering

- [GitHub](https://github.com/nmadagi)
- [LinkedIn](https://www.linkedin.com/in/nmadagi)
- [Portfolio](https://nmadagi.github.io/portfolio/)

---

## License

MIT License
