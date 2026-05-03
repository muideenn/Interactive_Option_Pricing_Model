# Multi-Company Options Pricing Model

**Author:** Muhideen Ogunlowo

An interactive financial dashboard for pricing equity options across 10 major companies using three quantitative models: the Binomial (CRR) tree, the Black-Scholes closed-form formula, and GARCH(1,1) volatility forecasting.

---

## Companies Covered

| Ticker | Company | Sector |
|--------|---------|--------|
| DIS | Walt Disney Company | Communications |
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corporation | Technology |
| GOOGL | Alphabet Inc. | Technology |
| AMZN | Amazon.com Inc. | Consumer Discretionary |
| TSLA | Tesla Inc. | Consumer Discretionary |
| NVDA | NVIDIA Corporation | Technology |
| META | Meta Platforms Inc. | Technology |
| NFLX | Netflix Inc. | Communications |
| JPM | JPMorgan Chase & Co. | Financials |

---

## Models Implemented

### Phase 1 — Market Data
- 5-year daily OHLCV data via `yfinance`
- Log return computation: `r_t = ln(P_t / P_{t-1})`
- Rolling historical volatility (21d, 63d, 126d windows), annualised by `× √252`
- Return distribution analysis (skewness, kurtosis)

### Phase 2 — Binomial Option Pricing (CRR)
- Cox-Ross-Rubinstein recombining lattice
- Up/down factors: `u = e^(σ√Δt)`, `d = 1/u`
- Risk-neutral probability: `p = (e^(rΔt) − d) / (u − d)`
- Supports **European** and **American** exercise
- Convergence analysis vs Black-Scholes (N = 5 → 300)

### Phase 3 — Black-Scholes
- Closed-form European option pricing
- All five Greeks: Δ Delta, Γ Gamma, Θ Theta, V Vega, ρ Rho
- Delta surface heatmap (Spot × Implied Vol)
- Payoff and time-value decomposition diagram

### Phase 4 — GARCH(1,1) Volatility Forecasting
- Fitted on daily log returns using the `arch` library
- Model: `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}`
- 60-day conditional variance forecast with confidence band
- GARCH-implied option prices vs live market quotes
- Implied volatility smile extraction (bisection solver)

### Phase 5 — Cross-Company Comparison
- Normalised return chart (all 10 names, base = 100)
- 21d HV bar chart sorted low → high
- ATM option premium as % of spot (vol-normalised comparison)
- Full summary table with GARCH persistence (α+β)

---

## Project Structure

```
Multi_Option_Pricing_Model/
├── src/
│   ├── models.py          # Binomial, Black-Scholes, GARCH, implied vol
│   ├── data.py            # Data fetching and preprocessing
│   ├── style.py           # Dash design system and layout helpers
│   └── tabs.py            # Tab render functions (all 5 tabs)
├── Multi_Company_Options_Dashboard.ipynb   # Main interactive dashboard
├── DIS_Options_Dashboard.ipynb             # Single-company DIS version
├── .env                   # Environment variables (not committed)
├── .env.example           # Template for environment setup
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/muideenn/Multi_Option_Pricing_Model.git
cd Multi_Option_Pricing_Model
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your values
```

### 5. Run the dashboard
Open `Multi_Company_Options_Dashboard.ipynb` in Jupyter and run all cells.
The Dash app will launch at **http://127.0.0.1:8050**

```bash
jupyter notebook Multi_Company_Options_Dashboard.ipynb
```

---

## Requirements

See `requirements.txt`. Key dependencies:

| Package | Purpose |
|---------|---------|
| `dash` | Interactive web dashboard framework |
| `dash-bootstrap-components` | Bootstrap layout components |
| `plotly` | Interactive charting |
| `yfinance` | Market data (OHLCV, options chain) |
| `numpy` / `pandas` | Numerical computing and data wrangling |
| `scipy` | Normal distribution functions for B-S |
| `arch` | GARCH model estimation and forecasting |

---

## Dashboard Features

- **Company selector** — switch between all 10 tickers; every chart and accent colour updates instantly
- **Strike as % of spot** — strike slider is vol-normalised (70%–130% of current price)
- **Global controls** — risk-free rate, maturity, call/put, European/American exercise
- **5 tabbed views** — Market Data, Binomial, Black-Scholes, GARCH Forecast, Cross-Company
- **Presentation mode** — each tab has a phase intro panel with methodology explanation and key insight callout
- **Synthetic fallback** — realistic mock data is used automatically if `yfinance` is offline

---

## Notes

- American option pricing via the binomial tree is exact (no approximation)
- The implied vol solver uses bisection with 150 iterations and 1e-7 tolerance
- GARCH is fitted on daily log returns scaled by 100 for numerical stability
- Live options chain data requires network access to Yahoo Finance

---

## License

MIT License — free to use, modify, and distribute with attribution.
