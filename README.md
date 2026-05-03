# Multi-Company Options Pricing Dashboard

Interactive Dash dashboard for comparing equity option pricing models across multiple companies. The app includes:

- Market data phase
- Binomial option pricing
- Black-Scholes option pricing and Greeks
- GARCH-based volatility forecast pricing
- Cross-company option comparison

The model controls for company, strike, risk-free rate, maturity, option type, and exercise style are wired into the Dash callbacks, so the displayed model outputs refresh when selections change.

## Project structure

```text
.
├── app.py                  # Production Dash application
├── requirements.txt        # Python dependencies
├── .env.example            # Safe environment variable template
├── Procfile                # Heroku/Railway-style web process
├── render.yaml             # Render Blueprint config
├── runtime.txt             # Python runtime for Heroku-style platforms
├── Dockerfile              # Optional Docker deployment
└── README.md
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Then open: `http://127.0.0.1:8090`

## Push to GitHub

```bash
git init
git add .
git commit -m "Add options pricing dashboard deployment files"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

The `.gitignore` keeps `.env` out of GitHub. Commit `.env.example`, not your private `.env`.

## Deploy on Render

1. Push this folder to GitHub.
2. In Render, choose **New Web Service**.
3. Connect the GitHub repository.
4. Use these settings:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:server`
5. Add environment variables if needed, for example `DASH_DEBUG=False`.

The included `render.yaml` can also be used as a Render Blueprint.

## Deploy with Docker

```bash
docker build -t options-dashboard .
docker run -p 8090:8090 options-dashboard
```

Then open: `http://127.0.0.1:8090`

## Notes

The dashboard currently uses deterministic synthetic company data inside the render callback, which makes deployed versions reliable even when live market data access is unavailable. The app still includes a `fetch_data` helper if you want to switch back to live `yfinance` data later.
