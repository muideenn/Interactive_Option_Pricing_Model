import os
from dotenv import load_dotenv
load_dotenv()

import io
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import norm
from arch import arch_model
import yfinance as yf
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Company registry ─────────────────────────────────────────────
COMPANIES = {
    'DIS':  dict(name='Walt Disney Company',   sector='Communications', mu=0.0001, sig=0.014, start=100),
    'AAPL': dict(name='Apple Inc.',             sector='Technology',     mu=0.0005, sig=0.016, start=180),
    'MSFT': dict(name='Microsoft Corporation',  sector='Technology',     mu=0.0006, sig=0.015, start=380),
    'GOOGL':dict(name='Alphabet Inc.',          sector='Technology',     mu=0.0004, sig=0.017, start=170),
    'AMZN': dict(name='Amazon.com Inc.',        sector='Consumer Disc.', mu=0.0003, sig=0.019, start=190),
    'TSLA': dict(name='Tesla Inc.',             sector='Consumer Disc.', mu=0.0002, sig=0.032, start=200),
    'NVDA': dict(name='NVIDIA Corporation',     sector='Technology',     mu=0.0010, sig=0.028, start=800),
    'META': dict(name='Meta Platforms Inc.',    sector='Technology',     mu=0.0005, sig=0.023, start=500),
    'NFLX': dict(name='Netflix Inc.',           sector='Communications', mu=0.0003, sig=0.021, start=600),
    'JPM':  dict(name='JPMorgan Chase & Co.',   sector='Financials',     mu=0.0003, sig=0.013, start=200),
}

def _mock_data(ticker):
    c = COMPANIES.get(ticker, COMPANIES['DIS'])
    np.random.seed(abs(hash(ticker)) % (2**31))
    rng = pd.bdate_range(end=pd.Timestamp.now(), periods=5*252)
    lr  = np.random.normal(float(c['mu']), float(c['sig']), len(rng))
    # Inject a brief stress period 2 years ago
    mid = len(rng) - 2*252
    lr[mid:mid+20] *= 3.0
    px  = np.exp(np.cumsum(lr)) * c['start']
    df  = pd.DataFrame({'Close': px, 'Log_Return': lr}, index=rng)
    for w in (21, 63, 126):
        df[f'HV_{w}'] = df['Log_Return'].rolling(w).std() * np.sqrt(252)
    return df.dropna()

def fetch_data(ticker='DIS', period='5y'):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty: raise ValueError('empty')
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        for w in (21, 63, 126):
            df[f'HV_{w}'] = df['Log_Return'].rolling(w).std() * np.sqrt(252)
        return df.dropna()
    except Exception as e:
        print(f'yfinance unavailable ({e}) -- synthetic data for {ticker}')
        return _mock_data(ticker)

# ── Pricing models ───────────────────────────────────────────────
def binomial_price(S,K,T,r,sigma,N=200,option_type='call',exercise='european'):
    if T<1e-9 or sigma<1e-9: return max(S-K,0) if option_type=='call' else max(K-S,0)
    dt=T/N; u=np.exp(sigma*np.sqrt(dt)); d=1/u
    p=(np.exp(r*dt)-d)/(u-d); disc=np.exp(-r*dt)
    j=np.arange(N+1); ST=S*u**(N-j)*d**j
    V=np.maximum(ST-K,0) if option_type=='call' else np.maximum(K-ST,0)
    for i in range(N-1,-1,-1):
        V=disc*(p*V[:-1]+(1-p)*V[1:])
        if exercise=='american':
            ji=np.arange(i+1); Si=S*u**(i-ji)*d**ji
            pf=np.maximum(Si-K,0) if option_type=='call' else np.maximum(K-Si,0)
            V=np.maximum(V,pf)
    return float(V[0])

def bs_price(S,K,T,r,sigma,option_type='call'):
    if T<1e-9 or sigma<1e-9: return max(S-K,0) if option_type=='call' else max(K-S,0)
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T)); d2=d1-sigma*np.sqrt(T)
    return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2) if option_type=='call' else K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

def bs_greeks(S,K,T,r,sigma,option_type='call'):
    if T<1e-9 or sigma<1e-9: return dict(delta=0.,gamma=0.,theta=0.,vega=0.,rho=0.)
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T)); d2=d1-sigma*np.sqrt(T)
    pdf1=norm.pdf(d1)
    delta=norm.cdf(d1) if option_type=='call' else norm.cdf(d1)-1
    gamma=pdf1/(S*sigma*np.sqrt(T))
    base=-S*pdf1*sigma/(2*np.sqrt(T))
    theta=(base-r*K*np.exp(-r*T)*norm.cdf(d2))/365 if option_type=='call' else (base+r*K*np.exp(-r*T)*norm.cdf(-d2))/365
    vega=S*pdf1*np.sqrt(T)/100
    rho=(K*T*np.exp(-r*T)*norm.cdf(d2) if option_type=='call' else -K*T*np.exp(-r*T)*norm.cdf(-d2))/100
    return dict(delta=delta,gamma=gamma,theta=theta,vega=vega,rho=rho)

def fit_garch(log_returns,p=1,q=1):
    model=arch_model(log_returns*100,vol='Garch',p=p,q=q,dist='normal')
    result=model.fit(disp='off')
    pm=result.params; keys=list(pm.index)
    omega=float(pm[next(k for k in keys if 'omega' in k.lower())])
    alpha=float(pm[next(k for k in keys if 'alpha' in k.lower())])
    beta =float(pm[next(k for k in keys if 'beta'  in k.lower())])
    return result, dict(omega=omega, alpha=alpha, beta=beta)

def garch_forecast(result, horizon=60):
    fc=result.forecast(horizon=horizon)
    return np.sqrt(fc.variance.values[-1,:])/100*np.sqrt(252)

def implied_vol(mkt_price,S,K,T,r,option_type='call',tol=1e-7):
    lo,hi=1e-6,6.0
    for _ in range(150):
        mid=(lo+hi)/2
        lo,hi=(mid,hi) if bs_price(S,K,T,r,mid,option_type)<mkt_price else (lo,mid)
        if hi-lo<tol: break
    return float(mid)

print('Models defined')

C = dict(
    bg='#0d1117', surface='#161b22', surface2='#21262d', border='#30363d',
    text='#e6edf3', muted='#8b949e',
    teal='#39d0a0', purple='#a78bfa', amber='#fbbf24',
    coral='#fb7185', blue='#60a5fa', green='#4ade80',
    orange='#fb923c', pink='#f472b6',
)
MONO = "'JetBrains Mono','Fira Code',monospace"

# Per-company accent colours (cycles through palette)
TICKER_COLORS = {
    'DIS': C['teal'],  'AAPL': C['blue'],  'MSFT': C['purple'],
    'GOOGL':C['amber'],'AMZN': C['orange'],'TSLA': C['coral'],
    'NVDA': C['green'],'META': C['pink'],   'NFLX': C['coral'],
    'JPM':  C['blue'],
}

BASE_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=C['text'],family='Inter,sans-serif',size=12),
    colorway=[C['teal'],C['purple'],C['amber'],C['coral'],C['blue'],
              C['green'],C['orange'],C['pink']],
    xaxis=dict(gridcolor='#21262d',linecolor=C['border'],zerolinecolor='#21262d'),
    yaxis=dict(gridcolor='#21262d',linecolor=C['border'],zerolinecolor='#21262d'),
    legend=dict(bgcolor='rgba(0,0,0,0)',bordercolor=C['border'],borderwidth=1,font=dict(size=11)),
    margin=dict(l=48,r=20,t=44,b=36),
    hoverlabel=dict(bgcolor=C['surface2'],bordercolor=C['border'],font=dict(color=C['text'],family='Inter')),
)
CARD  = {'background':C['surface'], 'border':f"1px solid {C['border']}", 'borderRadius':'12px','padding':'14px 18px'}
CARD2 = {'background':C['surface2'],'border':f"1px solid {C['border']}", 'borderRadius':'8px', 'padding':'12px 16px'}
LBL   = {'color':C['muted'],'fontSize':'10px','fontWeight':'600','letterSpacing':'0.09em','textTransform':'uppercase','marginBottom':'4px'}
VAL   = {'fontSize':'20px','fontWeight':'600','fontFamily':MONO,'lineHeight':'1.15'}

def apply_layout(fig,title='',height=300,legend_h=False,**kw):
    lo=dict(**BASE_LAYOUT,height=height,title=dict(text=title,font=dict(size=13,color=C['muted']),x=0,xanchor='left'))
    if legend_h: lo['legend'].update(orientation='h',y=1.12,x=0)
    lo.update(kw); fig.update_layout(**lo); return fig

def metric(label, value, color, width=2):
    """Responsive metric tile. Text is clipped inside the card instead of spilling into neighbours."""
    return dbc.Col(
        html.Div([
            html.Div(label, style={**LBL, 'lineHeight': '1.15'}),
            html.Div(value, style={**VAL, 'color': color, 'fontSize': 'clamp(15px, 1.55vw, 20px)',
                                   'whiteSpace': 'nowrap', 'overflow': 'hidden', 'textOverflow': 'ellipsis'}),
        ], style={**CARD, 'height': '100%', 'minWidth': 0, 'overflow': 'hidden'}),
        width=width, className='mb-2 metric-col'
    )

def metric_grid(items, min_width='142px'):
    """Clean horizontal/auto-fit metric strip for dense KPI sections."""
    return html.Div([
        html.Div([
            html.Div(label, style={**LBL, 'lineHeight': '1.15'}),
            html.Div(value, style={**VAL, 'color': color, 'fontSize': 'clamp(15px, 1.4vw, 20px)',
                                   'whiteSpace': 'nowrap', 'overflow': 'hidden', 'textOverflow': 'ellipsis'}),
        ], style={**CARD, 'minWidth': min_width, 'overflow': 'hidden'})
        for label, value, color in items
    ], className='metric-strip')

def parameter_strip(S, K, T, r, sigma, opt_type, exercise, acc):
    """Visible proof that every dashboard phase is using the selected inputs."""
    items = [
        ('Spot', f'${S:.2f}', acc),
        ('Strike', f'${K:.2f}', C['coral']),
        ('Maturity', f'{T*12:.0f} mo', C['amber']),
        ('Risk-free', f'{r*100:.2f}%', C['blue']),
        ('Vol used', f'{sigma*100:.1f}%', C['purple']),
        ('Option', f'{opt_type.title()} / {exercise.title()}', C['green']),
    ]
    return html.Div([
        html.Div('Current selected inputs', style={**LBL, 'marginBottom': '8px'}),
        metric_grid(items, min_width='132px')
    ], style={**CARD2, 'marginBottom': '16px', 'overflowX': 'auto'})

def slide_intro(phase_num,accent,title,body_lines,insight):
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div(f'Phase {phase_num}',style={'color':accent,'fontSize':'11px','fontWeight':'700',
                    'letterSpacing':'0.12em','textTransform':'uppercase','marginBottom':'6px'}),
                html.Div(title,style={'color':C['text'],'fontSize':'19px','fontWeight':'600','marginBottom':'10px'}),
                *[html.P(ln,style={'color':C['muted'],'fontSize':'13px','lineHeight':'1.7','margin':'0 0 5px 0'}) for ln in body_lines],
            ],width=8),
            dbc.Col(html.Div([
                html.Div('Key insight',style={'color':accent,'fontSize':'10px','fontWeight':'700',
                    'letterSpacing':'0.1em','textTransform':'uppercase','marginBottom':'8px'}),
                html.Div(insight,style={'color':C['text'],'fontSize':'13px','lineHeight':'1.65'}),
            ],style={'background':f"{accent}12",'border':f'1px solid {accent}40',
                'borderLeft':f'3px solid {accent}','borderRadius':'8px','padding':'14px 16px','height':'100%'}),
            width=4),
        ],className='g-3'),
    ],style={'background':C['surface'],'border':f"1px solid {C['border']}",
        'borderRadius':'12px','padding':'20px 24px','marginBottom':'20px'})

def chart_note(*cols):
    return html.Div(dbc.Row([dbc.Col(html.Div(txt,style={'color':C['muted'],'fontSize':'12px','lineHeight':'1.7'}),width=w)
        for txt,w in cols],className='g-3'),style=CARD2|{'marginTop':'14px'})

print('Style system ready')

def _rgba(hex_color, alpha):
    """Convert #RRGGBB + alpha float to rgba(...) string for Plotly."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

def render_market(df, S, hv21, ticker, acc):
    meta  = COMPANIES[ticker]
    cname = meta['name']
    sector = meta['sector']

    # ── Derived statistics ──────────────────────────────────────────────
    ret = df['Log_Return'].dropna()
    years = len(df) / 252
    total_ret   = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    ann_ret     = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (1 / years) - 1) * 100
    ann_vol     = float(ret.std() * np.sqrt(252) * 100)
    sharpe      = ann_ret / ann_vol if ann_vol else 0.0
    rolling_max = df['Close'].cummax()
    drawdown    = (df['Close'] - rolling_max) / rolling_max * 100
    max_dd      = float(drawdown.min())
    var_95      = float(np.percentile(ret, 5) * 100)
    cvar_95     = float(ret[ret <= np.percentile(ret, 5)].mean() * 100)
    w52_high    = float(df['Close'].iloc[-252:].max()) if len(df) >= 252 else float(df['Close'].max())
    w52_low     = float(df['Close'].iloc[-252:].min()) if len(df) >= 252 else float(df['Close'].min())
    w52_pct     = (S - w52_low) / (w52_high - w52_low) * 100 if w52_high != w52_low else 50.0

    # ── Company profile card ────────────────────────────────────────────
    profile = html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span(ticker, style={'color': acc, 'fontSize': '28px', 'fontWeight': '700',
                        'fontFamily': MONO, 'marginRight': '14px'}),
                    html.Span(sector, style={'background': f'{acc}20', 'color': acc,
                        'fontSize': '10px', 'fontWeight': '600', 'letterSpacing': '0.1em',
                        'textTransform': 'uppercase', 'padding': '3px 10px',
                        'borderRadius': '20px', 'border': f'1px solid {acc}50'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '6px'}),
                html.Div(cname, style={'color': C['text'], 'fontSize': '18px', 'fontWeight': '600',
                    'marginBottom': '8px'}),
                html.Div([
                    html.Span('5-year price and return analysis  ·  ', style={'color': C['muted'], 'fontSize': '12px'}),
                    html.Span('Daily log returns  ·  ', style={'color': C['muted'], 'fontSize': '12px'}),
                    html.Span('Rolling historical volatility  ·  ', style={'color': C['muted'], 'fontSize': '12px'}),
                    html.Span('Drawdown & risk metrics', style={'color': C['muted'], 'fontSize': '12px'}),
                ]),
            ], width=7),
            dbc.Col([
                # 52-week range bar
                html.Div('52-Week Range', style=LBL | {'marginBottom': '8px'}),
                html.Div([
                    html.Div(f'${w52_low:.2f}', style={'color': C['coral'], 'fontSize': '12px',
                        'fontFamily': MONO, 'marginBottom': '4px'}),
                    html.Div(style={
                        'background': f'linear-gradient(to right, {acc} {w52_pct:.1f}%, {C["surface2"]} {w52_pct:.1f}%)',
                        'height': '6px', 'borderRadius': '3px', 'margin': '4px 0',
                        'border': f'1px solid {C["border"]}',
                    }),
                    html.Div([
                        html.Span(f'Current  ${S:.2f}', style={'color': acc, 'fontSize': '12px', 'fontFamily': MONO}),
                        html.Span(f'${w52_high:.2f}', style={'color': C['green'], 'fontSize': '12px',
                            'fontFamily': MONO, 'marginLeft': 'auto'}),
                    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                ]),
            ], width=5),
        ], className='g-3', style={'alignItems': 'center'}),
    ], style={**CARD, 'marginBottom': '16px',
              'borderLeft': f'3px solid {acc}',
              'background': f'linear-gradient(135deg, {C["surface"]} 0%, #1a1f28 100%)'})

    # ── Key stats row ───────────────────────────────────────────────────
    ret_color = C['green'] if ann_ret >= 0 else C['coral']
    stats = [
        ('Last Close',   f'${S:.2f}',               acc),
        ('52W High',     f'${w52_high:.2f}',         C['green']),
        ('52W Low',      f'${w52_low:.2f}',          C['coral']),
        ('Ann. Return',  f'{ann_ret:+.1f}%',         ret_color),
        ('5Y Total Ret', f'{total_ret:+.1f}%',       ret_color),
        ('Ann. Vol',     f'{ann_vol:.1f}%',          C['amber']),
        ('HV 21d',       f'{hv21*100:.1f}%',         C['amber']),
        ('Sharpe',       f'{sharpe:.2f}',            C['purple']),
        ('Max Drawdown', f'{max_dd:.1f}%',           C['coral']),
        ('VaR 95%',      f'{var_95:.2f}%/day',      C['coral']),
        ('CVaR 95%',     f'{cvar_95:.2f}%/day',     C['coral']),
        ('Excess Kurt',  f'{float(ret.kurt()):.2f}', C['muted']),
    ]
    stat_row = metric_grid(stats, min_width='132px')

    # ── Price chart with 50d / 200d MA ──────────────────────────────────
    df['MA50']  = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=ticker,
        line=dict(color=acc, width=1.5), fill='tozeroy', fillcolor=_rgba(acc, 0.05)))
    fig_p.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='50d MA',
        line=dict(color=C['amber'], width=1.2, dash='dot')))
    fig_p.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='200d MA',
        line=dict(color=C['muted'], width=1.2, dash='dash')))
    apply_layout(fig_p, f'{ticker} — {cname}  ·  Closing Price + Moving Averages (5Y)',
                 height=300, legend_h=True)
    fig_p.update_xaxes(title_text='Date')
    fig_p.update_yaxes(title_text='Price ($)')

    # ── Drawdown chart ──────────────────────────────────────────────────
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=df.index, y=drawdown, mode='lines', name='Drawdown',
        line=dict(color=C['coral'], width=1.2),
        fill='tozeroy', fillcolor='rgba(251,113,133,0.12)'))
    fig_dd.add_hline(y=max_dd, line_color=C['coral'], line_dash='dot', line_width=1,
        annotation_text=f'Max DD  {max_dd:.1f}%', annotation_font_color=C['coral'])
    apply_layout(fig_dd, 'Drawdown from Peak (%)', height=220)
    fig_dd.update_yaxes(title_text='Drawdown (%)', ticksuffix='%')

    # ── Return distribution ─────────────────────────────────────────────
    mu, sd = float(ret.mean()), float(ret.std())
    x_fit = np.linspace(float(ret.min()), float(ret.max()), 300)
    bw = (float(ret.max()) - float(ret.min())) / 80
    y_fit = norm.pdf(x_fit, mu, sd) * len(ret) * bw
    fig_r = go.Figure()
    fig_r.add_trace(go.Histogram(x=ret, nbinsx=80, name='Log returns',
        marker_color=acc, opacity=0.7))
    fig_r.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines',
        line=dict(color=C['amber'], width=2), name='Normal fit'))
    fig_r.add_vline(x=float(np.percentile(ret, 5)), line_color=C['coral'],
        line_dash='dot', annotation_text='VaR 95%', annotation_font_color=C['coral'])
    apply_layout(fig_r, 'Log-Return Distribution  ·  Fat Tails vs Normal', height=260, legend_h=True)

    # ── Rolling volatility ──────────────────────────────────────────────
    fig_v = go.Figure()
    for col, clr, lbl in [('HV_21', acc, '21d HV'), ('HV_63', C['amber'], '63d HV'),
                           ('HV_126', C['muted'], '126d HV')]:
        fig_v.add_trace(go.Scatter(x=df.index, y=df[col] * 100, mode='lines',
            name=lbl, line=dict(color=clr, width=1.5)))
    apply_layout(fig_v, 'Annualised Historical Volatility (%)', height=260, legend_h=True)
    fig_v.update_yaxes(title_text='Vol (%)', ticksuffix='%')

    return html.Div([
        profile,
        stat_row,
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_p,  config={'displayModeBar': False}), width=12)], className='mb-2'),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_dd, config={'displayModeBar': False}), width=12)], className='mb-2'),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_r, config={'displayModeBar': False}), width=6),
            dbc.Col(dcc.Graph(figure=fig_v, config={'displayModeBar': False}), width=6),
        ]),
        chart_note(
            (f'{ticker} price chart with 50d (amber dotted) and 200d (grey dashed) moving averages. '
             'A price above the 200d MA signals a long-term uptrend.', 4),
            ('Drawdown measures the percentage decline from the running peak. '
             f'The deepest drawdown for {ticker} was {max_dd:.1f}% — the red dotted line.', 4),
            ('Return histogram vs normal fit: fat tails mean more extreme days than theory predicts. '
             'The red VaR 95% line is the worst 5% of daily moves.', 4),
        ),
    ])


# ══ Tab 2: Binomial ══════════════════════════════════════════════
def _build_tree(S, K, T, r, sigma, N_viz, opt_type, exercise):
    """Return stock-price tree and backward-induced option-value tree as dicts."""
    dt = T / N_viz
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    S_tree = {(t, j): S * u**(t - j) * d**j
              for t in range(N_viz + 1) for j in range(t + 1)}

    V_tree = {}
    for j in range(N_viz + 1):
        sv = S_tree[(N_viz, j)]
        V_tree[(N_viz, j)] = max(sv - K, 0) if opt_type == 'call' else max(K - sv, 0)
    for t in range(N_viz - 1, -1, -1):
        for j in range(t + 1):
            hold = disc * (p * V_tree[(t + 1, j)] + (1 - p) * V_tree[(t + 1, j + 1)])
            if exercise == 'american':
                sv = S_tree[(t, j)]
                intrinsic = max(sv - K, 0) if opt_type == 'call' else max(K - sv, 0)
                V_tree[(t, j)] = max(hold, intrinsic)
            else:
                V_tree[(t, j)] = hold
    return S_tree, V_tree, dt, u, d, p


def _tree_figure(tree_vals, K, N_viz, opt_type, acc, title, value_type='stock'):
    """Render one side of the lattice (stock prices or option values)."""
    # Build edges
    ex, ey = [], []
    for t in range(N_viz):
        for j in range(t + 1):
            x0, y0 = t, t - 2 * j
            ex += [x0, t + 1, None];  ey += [y0, (t + 1) - 2 * j,       None]  # up
            ex += [x0, t + 1, None];  ey += [y0, (t + 1) - 2 * (j + 1), None]  # down

    # Build nodes
    nx_list, ny_list, colors, texts, hovers = [], [], [], [], []
    for t in range(N_viz + 1):
        for j in range(t + 1):
            v = tree_vals[(t, j)]
            nx_list.append(t)
            ny_list.append(t - 2 * j)
            texts.append(f'${v:.2f}')
            hovers.append(f'Step {t}, Down {j}<br>{"Stock" if value_type=="stock" else "Option"}: ${v:.4f}')

            if value_type == 'stock':
                itm = (v > K) if opt_type == 'call' else (v < K)
                colors.append(C['green'] if itm else C['coral'])
            else:
                if v > 0.005:
                    colors.append(acc)
                else:
                    colors.append(C['border'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode='lines',
        line=dict(color=C['border'], width=1),
        hoverinfo='skip', showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=nx_list, y=ny_list, mode='markers+text',
        marker=dict(size=28, color=colors, line=dict(width=1.5, color=C['surface2'])),
        text=texts,
        textfont=dict(color=C['bg'], size=8, family=MONO),
        textposition='middle center',
        hovertext=hovers, hoverinfo='text',
        showlegend=False,
    ))
    _base = {k: v for k, v in BASE_LAYOUT.items() if k not in ('xaxis', 'yaxis', 'margin', 'legend')}
    fig.update_layout(
        **_base,
        height=420,
        title=dict(text=title, font=dict(size=13, color=C['muted'])),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=True,
                   tickmode='array', tickvals=list(range(N_viz + 1)),
                   ticktext=[f't={i}' for i in range(N_viz + 1)],
                   title='Time Step', linecolor=C['border']),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   linecolor=C['border']),
        margin=dict(l=20, r=20, t=44, b=36),
    )
    return fig


def render_binomial(S, K, T, r, sigma, opt_type, exercise, ticker, acc):
    intro = slide_intro(2, acc,
        f'CRR Binomial Tree — {ticker}  ({opt_type.title()} / {exercise.title()})',
        [
            'The Cox-Ross-Rubinstein (1979) binomial model builds a recombining price lattice. '
            'At each node the stock can move up by u = e^(σ√Δt) or down by d = 1/u. '
            'The risk-neutral probability p = (e^(rΔt) − d)/(u − d) prices the option '
            'without arbitrage by backward induction from the terminal payoffs.',
            'American exercise compares the hold value against immediate exercise at every node. '
            'Convergence to the Black-Scholes price is oscillatory — even/odd N alternate '
            'above and below — and the absolute error shrinks as O(1/N).',
        ],
        'The left lattice shows the stock price at each node; green = in-the-money, '
        'red = out-of-the-money. The right lattice shows the back-propagated option value. '
        'Terminal payoffs are max(S−K, 0) for calls; internal nodes discount one step at a time.'
    )

    # ── Tree visualisation (N=6 steps for clarity) ──────────────────────
    N_viz = 6
    S_tree, V_tree, dt_viz, u_viz, d_viz, p_viz = _build_tree(S, K, T, r, sigma, N_viz, opt_type, exercise)
    fig_s = _tree_figure(S_tree, K, N_viz, opt_type, acc,
                         f'{ticker} Stock Price Lattice (N={N_viz})', 'stock')
    fig_v = _tree_figure(V_tree, K, N_viz, opt_type, acc,
                         f'{ticker} Option Value Lattice (N={N_viz})', 'option')

    # Legend strip for tree
    legend_strip = html.Div([
        html.Div([
            html.Span(style={'display':'inline-block','width':'12px','height':'12px',
                'borderRadius':'50%','background':C['green'],'marginRight':'6px','verticalAlign':'middle'}),
            html.Span('In-the-money node', style={'color':C['muted'],'fontSize':'11px','marginRight':'18px'}),
            html.Span(style={'display':'inline-block','width':'12px','height':'12px',
                'borderRadius':'50%','background':C['coral'],'marginRight':'6px','verticalAlign':'middle'}),
            html.Span('Out-of-the-money node', style={'color':C['muted'],'fontSize':'11px','marginRight':'18px'}),
            html.Span(style={'display':'inline-block','width':'12px','height':'12px',
                'borderRadius':'50%','background':acc,'marginRight':'6px','verticalAlign':'middle'}),
            html.Span('Option has value > 0', style={'color':C['muted'],'fontSize':'11px','marginRight':'18px'}),
            html.Span(style={'display':'inline-block','width':'12px','height':'12px',
                'borderRadius':'50%','background':C['border'],'marginRight':'6px','verticalAlign':'middle'}),
            html.Span('Option expires worthless', style={'color':C['muted'],'fontSize':'11px'}),
        ]),
    ], style={**CARD2, 'marginBottom':'14px'})

    # ── CRR parameters panel ─────────────────────────────────────────────
    dt_crr = T / 100; u_crr = np.exp(sigma * np.sqrt(dt_crr)); d_crr = 1/u_crr
    p_crr  = (np.exp(r * dt_crr) - d_crr) / (u_crr - d_crr)
    params_box = html.Div([
        html.Div('CRR Parameters  (for N=100 production run)', style=LBL | {'marginBottom': '10px'}),
        dbc.Row([
            dbc.Col(html.Div([
                html.Span(k + ' = ', style={'color': C['muted'], 'fontFamily': MONO, 'fontSize': '12px'}),
                html.Span(v, style={'color': acc, 'fontFamily': MONO, 'fontSize': '13px'}),
                html.Br(),
                html.Span(desc, style={'color': C['muted'], 'fontSize': '10px', 'lineHeight': '1.5'}),
            ]), width=3)
            for k, v, desc in [
                ('u',  f'{u_crr:.6f}', 'Up factor  e^(σ√Δt)'),
                ('d',  f'{d_crr:.6f}', 'Down factor  1/u'),
                ('p*', f'{p_crr:.6f}', 'Risk-neutral up probability'),
                ('Δt', f'{dt_crr:.6f}', 'Length of one time step (years)'),
            ]
        ], className='g-2'),
    ], style=CARD | {'marginBottom': '16px'})

    # ── Convergence + strike chart ───────────────────────────────────────
    bs_val  = bs_price(S, K, T, r, sigma, opt_type)
    N_vals  = list(range(5, 305, 5))
    b_prices = [binomial_price(S, K, T, r, sigma, n, opt_type, exercise) for n in N_vals]
    errors   = [abs(p - bs_val) for p in b_prices]

    fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=['Binomial price vs N steps', '|Error| vs Black-Scholes'],
        vertical_spacing=0.14)
    fig_c.add_trace(go.Scatter(x=N_vals, y=b_prices, mode='lines', name='Binomial',
        line=dict(color=acc, width=2)), row=1, col=1)
    fig_c.add_hline(y=bs_val, line_color=C['amber'], line_dash='dash', line_width=1.5,
        annotation_text=f'B-S ${bs_val:.3f}', annotation_font_color=C['amber'], row=1, col=1)
    fig_c.add_trace(go.Scatter(x=N_vals, y=errors, mode='lines', name='|Error|',
        line=dict(color=C['coral'], width=1.5),
        fill='tozeroy', fillcolor='rgba(251,113,133,0.1)'), row=2, col=1)
    fig_c.update_layout(**BASE_LAYOUT, height=360, showlegend=False,
        title=dict(text=f'CRR Convergence to B-S — {ticker}', font=dict(size=13, color=C['muted'])))
    fig_c.update_xaxes(title_text='N (Steps)', row=2, gridcolor='#21262d')
    fig_c.update_yaxes(title_text='Price ($)',   row=1, gridcolor='#21262d')
    fig_c.update_yaxes(title_text='|Error| ($)', row=2, gridcolor='#21262d')

    strikes = np.linspace(max(S * 0.6, 1), S * 1.4, 60)
    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(x=strikes,
        y=[binomial_price(S, k, T, r, sigma, 100, opt_type, exercise) for k in strikes],
        mode='lines', name='Binomial N=100', line=dict(color=acc, width=2)))
    fig_k.add_trace(go.Scatter(x=strikes,
        y=[bs_price(S, k, T, r, sigma, opt_type) for k in strikes],
        mode='lines', name='Black-Scholes', line=dict(color=C['amber'], width=2, dash='dash')))
    fig_k.add_vline(x=S, line_color=acc, line_dash='dot',
        annotation_text='Spot', annotation_font_color=acc)
    fig_k.add_vline(x=K, line_color=C['coral'], line_dash='dot',
        annotation_text='K', annotation_font_color=C['coral'], annotation_position='top right')
    apply_layout(fig_k, f'{opt_type.title()} Price vs Strike — {ticker}', height=290, legend_h=True)
    fig_k.update_xaxes(title_text='Strike ($)'); fig_k.update_yaxes(title_text='Price ($)')

    # ── Price snapshots ──────────────────────────────────────────────────
    snaps   = [(n, binomial_price(S, K, T, r, sigma, n, opt_type, exercise)) for n in [10, 50, 100, 200]]
    err200  = abs(snaps[-1][1] - bs_val)
    snap_row = dbc.Row([
        *[metric(f'Binomial N={n}', f'${v:.4f}', acc, width=2) for n, v in snaps],
        metric('Black-Scholes', f'${bs_val:.4f}', C['amber'], width=2),
        metric('Error N=200',   f'${err200:.5f}', C['coral'], width=2),
    ], className='g-2 mb-3')

    return html.Div([
        intro,
        parameter_strip(S, K, T, r, sigma, opt_type, exercise, acc),
        snap_row,
        params_box,
        legend_strip,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_s, config={'displayModeBar': False}), width=6),
            dbc.Col(dcc.Graph(figure=fig_v, config={'displayModeBar': False}), width=6),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_c, config={'displayModeBar': False}), width=7),
            dbc.Col(dcc.Graph(figure=fig_k, config={'displayModeBar': False}), width=5),
        ]),
        chart_note(
            (f'Tree shows N={N_viz} steps for readability. The production price uses N=200. '
             'Each node shows the rounded dollar value; hover for exact figures.', 4),
            ('Convergence chart: price oscillates above/below B-S as N increases (even/odd effect). '
             'The error halves with each doubling of N.', 4),
            ('Strike curve: both models agree closely ATM. Deep ITM/OTM deviations are largest at '
             'small N and collapse as N grows toward the B-S limit.', 4),
        ),
    ])


# ══ Tab 3: Black-Scholes ═════════════════════════════════════════
def render_bs(S, K, T, r, sigma, opt_type, exercise, ticker, acc):
    # ── Core calculations ────────────────────────────────────────────────
    d1   = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2   = d1 - sigma * np.sqrt(T)
    Nd1  = float(norm.cdf(d1  if opt_type == 'call' else -d1))
    Nd2  = float(norm.cdf(d2  if opt_type == 'call' else -d2))
    disc = float(np.exp(-r * T))
    p_bs = bs_price(S, K, T, r, sigma, opt_type)
    p_bin = binomial_price(S, K, T, r, sigma, 200, opt_type, exercise)
    g    = bs_greeks(S, K, T, r, sigma, opt_type)

    # ── Formula decomposition card ───────────────────────────────────────
    lhs = 'C' if opt_type == 'call' else 'P'
    formula_lbl = (f'{lhs} = S·N(d1) − K·e^(−rT)·N(d2)'
                   if opt_type == 'call'
                   else f'{lhs} = K·e^(−rT)·N(−d2) − S·N(−d1)')

    def frow(label, expr, value, color=None):
        return html.Div([
            html.Span(label, style={'color': C['muted'], 'fontSize': '11px',
                'fontFamily': MONO, 'width': '120px', 'display': 'inline-block'}),
            html.Span(expr,  style={'color': C['muted'], 'fontSize': '11px',
                'fontFamily': MONO, 'width': '260px', 'display': 'inline-block'}),
            html.Span(f'= {value}', style={'color': color or acc, 'fontSize': '13px',
                'fontFamily': MONO, 'fontWeight': '600'}),
        ], style={'marginBottom': '8px'})

    formula_card = html.Div([
        html.Div([
            html.Span('Black-Scholes Formula', style={**LBL, 'fontSize': '11px', 'marginBottom': '10px',
                'display': 'block'}),
            html.Div(formula_lbl, style={'color': acc, 'fontFamily': MONO, 'fontSize': '15px',
                'fontWeight': '600', 'marginBottom': '16px',
                'borderBottom': f'1px solid {C["border"]}', 'paddingBottom': '12px'}),
        ]),
        dbc.Row([
            dbc.Col([
                frow('d1',    f'[ln(S/K) + (r + ½σ²)T] / σ√T', f'{d1:+.4f}'),
                frow('d2',    'd1 − σ√T',                        f'{d2:+.4f}'),
                frow('N(d1)', 'Prob stock ends ITM (risk-neutral)',f'{Nd1:.4f}', C['purple']),
                frow('N(d2)', 'Prob option exercised',            f'{Nd2:.4f}', C['purple']),
            ], width=6),
            dbc.Col([
                frow('S',    'Current spot price',          f'${S:.2f}'),
                frow('K',    'Strike price',                f'${K:.2f}'),
                frow('e^(-rT)', 'Discount factor',          f'{disc:.6f}', C['amber']),
                frow(f'{lhs}', 'Option price',              f'${p_bs:.4f}', acc),
            ], width=6),
        ], className='g-2'),
        html.Div([
            html.Span('B-S  ', style={'color': C['muted'], 'fontFamily': MONO, 'fontSize': '11px'}),
            html.Span(f'${p_bs:.4f}', style={'color': acc, 'fontFamily': MONO, 'fontSize': '13px',
                'fontWeight': '600', 'marginRight': '24px'}),
            html.Span('Binomial N=200  ', style={'color': C['muted'], 'fontFamily': MONO, 'fontSize': '11px'}),
            html.Span(f'${p_bin:.4f}', style={'color': C['purple'], 'fontFamily': MONO, 'fontSize': '13px',
                'fontWeight': '600', 'marginRight': '24px'}),
            html.Span('Diff  ', style={'color': C['muted'], 'fontFamily': MONO, 'fontSize': '11px'}),
            html.Span(f'${abs(p_bs - p_bin):.5f}',
                style={'color': C['coral'], 'fontFamily': MONO, 'fontSize': '13px', 'fontWeight': '600'}),
        ], style={'marginTop': '14px', 'paddingTop': '12px', 'borderTop': f'1px solid {C["border"]}'}),
    ], style={**CARD, 'marginBottom': '16px'})

    # ── Greek cards ──────────────────────────────────────────────────────
    greek_items = [
        ('Delta (Δ)', f'{g["delta"]:+.4f}', acc,
         '$1 stock move sensitivity. 0→1 for calls, −1→0 for puts.'),
        ('Gamma (Γ)', f'{g["gamma"]:.5f}', C['purple'],
         'Rate of change of Delta. Highest ATM near expiry.'),
        ('Theta (Θ)', f'${g["theta"]:+.4f}/d', C['amber'],
         'Daily time decay. Long options lose value each day.'),
        ('Vega (ν)',  f'${g["vega"]:.4f}/%', C['coral'],
         'Sensitivity to 1% vol change. Highest ATM.'),
        ('Rho (ρ)',   f'${g["rho"]:+.4f}/%', C['blue'],
         'Sensitivity to 1% rate change. Smaller for short-dated.'),
    ]
    greek_cards = dbc.Row([
        dbc.Col(html.Div([
            html.Div(lbl, style=LBL),
            html.Div(val, style=VAL | {'color': clr, 'fontSize': '18px', 'marginBottom': '5px'}),
            html.Div(desc, style={'color': C['muted'], 'fontSize': '11px', 'lineHeight': '1.5'}),
        ], style=CARD), width=2, className='mb-2')
        for lbl, val, clr, desc in greek_items
    ], className='g-2 mb-3')

    # ── Payoff + time-value diagram ──────────────────────────────────────
    spots    = np.linspace(S * 0.5, S * 1.5, 200)
    intr     = np.maximum(spots - K, 0) if opt_type == 'call' else np.maximum(K - spots, 0)
    bs_vals  = np.array([bs_price(s, K, T, r, sigma, opt_type) for s in spots])
    fig_pay = go.Figure()
    fig_pay.add_trace(go.Scatter(x=spots, y=bs_vals, mode='lines', name='B-S Value',
        line=dict(color=acc, width=2.5)))
    fig_pay.add_trace(go.Scatter(x=spots, y=intr, mode='lines', name='Intrinsic',
        line=dict(color=C['muted'], width=1.5, dash='dash')))
    fig_pay.add_trace(go.Scatter(
        x=list(spots) + list(spots[::-1]), y=list(bs_vals) + list(intr[::-1]),
        fill='toself', fillcolor=_rgba(acc, 0.06), line=dict(width=0),
        name='Time value', hoverinfo='skip'))
    fig_pay.add_vline(x=S, line_color=acc,        line_dash='dot', annotation_text='Spot',   annotation_font_color=acc)
    fig_pay.add_vline(x=K, line_color=C['coral'], line_dash='dot', annotation_text='Strike', annotation_font_color=C['coral'], annotation_position='top right')
    apply_layout(fig_pay, f'Option Value vs Spot — {ticker}', height=290, legend_h=True)
    fig_pay.update_xaxes(title_text='Spot ($)'); fig_pay.update_yaxes(title_text='Value ($)')

    # ── Greeks sensitivity curves vs spot ────────────────────────────────
    sp_range = np.linspace(S * 0.6, S * 1.4, 120)
    delta_v  = np.array([bs_greeks(s, K, T, r, sigma, opt_type)['delta'] for s in sp_range])
    gamma_v  = np.array([bs_greeks(s, K, T, r, sigma, opt_type)['gamma'] for s in sp_range])
    theta_v  = np.array([bs_greeks(s, K, T, r, sigma, opt_type)['theta'] for s in sp_range])
    vega_v   = np.array([bs_greeks(s, K, T, r, sigma, opt_type)['vega']  for s in sp_range])

    fig_gr = make_subplots(rows=2, cols=2,
        subplot_titles=['Delta (Δ)', 'Gamma (Γ)', 'Theta (Θ) — daily decay', 'Vega (ν) — per 1% vol'],
        vertical_spacing=0.18, horizontal_spacing=0.10)

    for row, col, ydata, clr, ylab in [
        (1, 1, delta_v, acc,          'Delta'),
        (1, 2, gamma_v, C['purple'],  'Gamma'),
        (2, 1, theta_v, C['amber'],   'Theta ($/day)'),
        (2, 2, vega_v,  C['coral'],   'Vega ($/%vol)'),
    ]:
        fig_gr.add_trace(go.Scatter(x=sp_range, y=ydata, mode='lines',
            line=dict(color=clr, width=2), showlegend=False), row=row, col=col)
        fig_gr.add_vline(x=S, line_color=clr, line_dash='dot', line_width=1, row=row, col=col)
        fig_gr.add_vline(x=K, line_color=C['muted'], line_dash='dot', line_width=1, row=row, col=col)

    fig_gr.update_layout(**BASE_LAYOUT, height=400,
        title=dict(text=f'Greeks vs Spot Price — {ticker}  (dotted lines: spot={S:.0f}, strike={K:.0f})',
                   font=dict(size=13, color=C['muted'])))
    for r_ in [1, 2]:
        for c_ in [1, 2]:
            fig_gr.update_xaxes(title_text='Spot ($)', gridcolor='#21262d', row=r_, col=c_)
            fig_gr.update_yaxes(gridcolor='#21262d', row=r_, col=c_)

    return html.Div([
        slide_intro(3, acc,
            f'Black-Scholes Model — {ticker}  ({opt_type.title()})',
            [
                'The Black-Scholes formula gives a closed-form option price under geometric Brownian '
                'motion with constant volatility. The formula panel below breaks down every intermediate '
                'value — d1, d2, and the cumulative normal probabilities N(d1) and N(d2) — computed '
                'for the current inputs.',
                'The five Greeks measure how the option price responds to each market variable. '
                'The sensitivity charts show how every Greek evolves as the spot price moves from '
                '60% to 140% of its current level, with the current spot and strike marked.',
            ],
            f'Vega peaks at-the-money because that is where uncertainty about final payoff is highest. '
            f'Delta is steepest near the strike; Gamma (the rate-of-change of Delta) also peaks ATM '
            f'and collapses deep ITM/OTM — making ATM options the most expensive to delta-hedge.'
        ),
        parameter_strip(S, K, T, r, sigma, opt_type, exercise, acc),
        formula_card,
        greek_cards,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_pay, config={'displayModeBar': False}), width=5),
            dbc.Col(dcc.Graph(figure=fig_gr,  config={'displayModeBar': False}), width=7),
        ]),
        chart_note(
            ('Payoff diagram: shaded area is time value. It peaks ATM and decays to zero at expiry '
             'or deep in/out of the money.', 4),
            ('Greeks curves: each panel shows how that sensitivity changes across the spot range. '
             'The coloured dotted line marks current spot; grey dotted line marks strike.', 4),
            (f'Current inputs: S={S:.2f}, K={K:.2f}, T={T*12:.0f}mo, r={r*100:.2f}%, σ={sigma*100:.1f}%. '
             f'Change any control above and all outputs update instantly.', 4),
        ),
    ])


# ══ Tab 4: GARCH ═════════════════════════════════════════════════
def render_garch(df, S, K, T, r, opt_type, ticker, acc):
    intro = slide_intro(4, acc,
        f'GARCH(1,1) Volatility Forecast — {ticker}',
        [
            'Generalised Autoregressive Conditional Heteroskedasticity (GARCH) models time-varying '
            'variance: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}. α captures the impact of recent shocks; '
            'β captures persistence. The sum α+β < 1 ensures mean-reversion.',
            'We fit the model to log returns, generate a 60-day conditional variance forecast, '
            'annualise it, and substitute into Black-Scholes. This gives a regime-aware option price '
            'that responds to current market conditions rather than a static historical average.',
        ],
        'High α+β (close to 1) means volatility is persistent — a shock today influences '
        'vol for many days ahead. High-β stocks like TSLA or NVDA show very persistent vol, '
        'while lower-β financials like JPM mean-revert faster after earnings or macro events.'
    )
    returns=df['Log_Return'].dropna()
    garch_result,gparams=fit_garch(returns)
    cond_vol=garch_result.conditional_volatility/100*np.sqrt(252)
    fc_vol=garch_forecast(garch_result,60)
    fc_dates=pd.bdate_range(df.index[-1]+pd.Timedelta(days=1),periods=60)

    fig_gv=go.Figure()
    fig_gv.add_trace(go.Scatter(x=df.index[-252:],y=cond_vol[-252:]*100,mode='lines',
        name='GARCH Cond. Vol',line=dict(color=acc,width=1.5)))
    fig_gv.add_trace(go.Scatter(x=df.index[-252:],y=df['HV_21'][-252:]*100,mode='lines',
        name='21d HV',line=dict(color=C['muted'],width=1,dash='dot')))
    fig_gv.add_trace(go.Scatter(x=fc_dates,y=fc_vol*100,mode='lines',
        name='60d Forecast',line=dict(color=C['amber'],width=2.5)))
    band=fc_vol*0.18
    fig_gv.add_trace(go.Scatter(
        x=list(fc_dates)+list(fc_dates[::-1]),
        y=list((fc_vol+band)*100)+list((fc_vol-band)*100),
        fill='toself',fillcolor='rgba(251,191,36,0.08)',
        line=dict(width=0),name='95% band',hoverinfo='skip'))
    fig_gv.add_vline(x=str(df.index[-1]),line_color=C['muted'],line_dash='dash',line_width=1)
    apply_layout(fig_gv, f'GARCH(1,1) — {ticker} | {opt_type.title()} K=${K:.2f}, T={T*12:.0f}mo, r={r*100:.2f}%',
        height=290, legend_h=True)
    fig_gv.update_yaxes(title_text='Annualised Vol (%)',ticksuffix='%')

    try:
        tkr_obj=yf.Ticker(ticker); exp_dates=tkr_obj.options
        exp_date=exp_dates[min(2,len(exp_dates)-1)]
        chain=getattr(tkr_obj.option_chain(exp_date),'calls' if opt_type=='call' else 'puts')
        chain=chain[(chain['bid']>0)&(chain['ask']>0)].copy()
        chain['mid']=(chain['bid']+chain['ask'])/2
        T_c=max((pd.to_datetime(exp_date)-pd.Timestamp.now()).days/365,0.01)
        gs=float(fc_vol[0])
        chain['garch_px']=chain['strike'].apply(lambda k: bs_price(S,k,T_c,r,gs,opt_type))
        chain['impl_vol']=chain.apply(
            lambda row: implied_vol(row['mid'],S,row['strike'],T_c,r,opt_type)*100
            if row['mid']>0.05 else np.nan,axis=1)
        chain=chain.dropna(subset=['impl_vol'])
        fig_iv=go.Figure()
        fig_iv.add_trace(go.Scatter(x=chain['strike'],y=chain['impl_vol'],mode='lines+markers',
            name='Impl. Vol',line=dict(color=acc,width=2),marker=dict(size=5)))
        fig_iv.add_hline(y=gs*100,line_color=C['amber'],line_dash='dash',
            annotation_text=f'GARCH: {gs*100:.1f}%',annotation_font_color=C['amber'])
        fig_iv.add_hline(y=float(df['HV_21'].iloc[-1])*100,line_color=C['teal'],line_dash='dot',
            annotation_text=f'21d HV',annotation_font_color=C['teal'])
        fig_iv.add_vline(x=S,line_color=C['muted'],line_dash='dot',annotation_text='Spot')
        apply_layout(fig_iv,f'Implied Vol Smile — {ticker} {exp_date}',height=290,legend_h=True)
        fig_iv.update_xaxes(title_text='Strike ($)'); fig_iv.update_yaxes(title_text='Impl. Vol (%)',ticksuffix='%')
        tbl_df=chain[['strike','mid','garch_px','impl_vol']].rename(columns={
            'strike':'Strike','mid':'Mkt Price','garch_px':'GARCH-BS','impl_vol':'Impl Vol %'}).copy()
        tbl_df['Diff']=( tbl_df['GARCH-BS']-tbl_df['Mkt Price']).round(4)
        tbl = dash_table.DataTable(
            data=tbl_df.head(14).round(3).to_dict('records'),
            columns=[{'name': c, 'id': c} for c in tbl_df.columns],
            style_table={'overflowX': 'auto', 'borderRadius': '8px', 'overflow': 'hidden'},
            style_cell={
            'backgroundColor': C['surface'],
            'color': C['text'],
            'border': f"1px solid {C['border']}",
            'fontFamily': "'JetBrains Mono',monospace",
            'fontSize': '11px',
            'padding': '7px 12px',
            'textAlign': 'right'
            },
            style_header={
            'backgroundColor': C['surface2'],
            'color': acc,
            'fontWeight': '600',
            'fontSize': '10px',
            'letterSpacing': '0.07em',
            'border': f"1px solid {C['border']}"
            },
            style_data_conditional=[  # pyright: ignore[reportArgumentType]
            {
                'if': {'filter_query': '{Diff} > 0', 'column_id': 'Diff'},
                'backgroundColor': f'{C["teal"]}22',
                'fontWeight': '600',
            },
            {
                'if': {'filter_query': '{Diff} < 0', 'column_id': 'Diff'},
                'backgroundColor': f'{C["coral"]}22',
                'fontWeight': '600',
            },
            ],
        )
        chain_sec=dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_iv,config={'displayModeBar':False}),width=7),
            dbc.Col(html.Div([html.Div('Market vs GARCH-BS',style=LBL|{'marginBottom':'8px'}),
                html.P('Teal = model cheap vs market. Coral = model expensive.',
                    style={'color':C['muted'],'fontSize':'11px','marginBottom':'8px'}),tbl],style=CARD),width=5),
        ],className='mt-3')
    except Exception as e:
        chain_sec=html.Div(f'Live options chain unavailable: {e}',
            style={'color':C['muted'],'padding':'12px','fontSize':'12px'})

    pers=gparams['alpha']+gparams['beta']
    selected_garch_px = bs_price(S, K, T, r, float(fc_vol[0]), opt_type)
    gcard_row=dbc.Row([
        metric('Selected GARCH-BS', f'${selected_garch_px:.4f}', acc),
        metric('Selected Strike',   f'${K:.2f}', C['coral']),
        metric('Selected T',        f'{T*12:.0f} mo', C['amber']),
        metric('GARCH Vol 1d',      f'{fc_vol[0]*100:.2f}%', C['amber']),
        metric('alpha+beta',        f'{pers:.6f}', C['purple']),
        metric('Log-Likeli.',       f'{garch_result.loglikelihood:.1f}', C['teal']),
    ],className='g-2 mb-3')
    return html.Div([intro,
        parameter_strip(S, K, T, r, float(fc_vol[0]), opt_type, 'European', acc),
        gcard_row,
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_gv,config={'displayModeBar':False}),width=12)],className='mb-2'),
        chain_sec,
        chart_note(
            ('GARCH conditional vol (solid) adapts to volatility clusters. The 60-day amber '
             'forecast mean-reverts toward the long-run unconditional vol level.', 6),
            ('The implied vol smile shows the market prices OTM puts at a premium (skew). '
             'If GARCH vol < smile, a volatility risk premium exists in the market.', 6),
        ),
    ])


# ══ Tab 5: Cross-Company Comparison ══════════════════════════════
def render_comparison(selected_ticker, r, opt_type, T, K_pct):
    intro = slide_intro(5, C['blue'],
        'Cross-Company Comparison — All 10 Names',
        [
            'This tab aggregates key metrics across all ten companies simultaneously. '
            'Prices are normalised to 100 at the start of the five-year window to allow '
            'direct performance comparison regardless of absolute price levels.',
            'Options are priced at the selected strike percentage using Black-Scholes '
            'with each company\'s own 21-day historical volatility. The GARCH(1,1) model '
            'is fitted independently for each name to compare volatility regimes.',
        ],
        'High-vol tech names (TSLA, NVDA, META) command significantly higher selected-strike option '
        'premiums than low-vol financials (JPM) or large-caps (MSFT, AAPL). '
        'GARCH persistence (α+β) reveals which stocks stay volatile after shocks.'
    )

    # Load all companies (uses mock data so it's fast)
    rows = []
    norm_data = {}
    hv_series = {}
    for tkr, meta in COMPANIES.items():
        df = _mock_data(tkr)
        S  = float(df['Close'].iloc[-1])
        hv = float(df['HV_21'].iloc[-1])
        K_sel = S * (K_pct / 100)
        p_bs  = bs_price(S, K_sel, T, r, hv, opt_type)
        p_pct = p_bs / S * 100  # as % of spot
        ret   = df['Log_Return'].dropna()
        try:
            _, gp = fit_garch(ret)
            pers  = gp['alpha'] + gp['beta']
        except Exception:
            pers = float('nan')
        rows.append(dict(Ticker=tkr, Name=meta['name'][:22], Spot=f'${S:.2f}',
                         HV_21=f'{hv*100:.1f}%',
                         Selected_K=f'{K_pct:.0f}% (${K_sel:.2f})',
                         Option_Price=f'${p_bs:.3f}',
                         Premium_Pct=f'{p_pct:.2f}%',
                         Persistence=f'{pers:.4f}',
                         Sector=meta['sector']))
        # Normalised price
        norm_data[tkr] = (df['Close'] / float(df['Close'].iloc[0]) * 100).values
        hv_series[tkr] = df['HV_21'].values
        idx_len = len(df)

    # Normalised price chart
    fig_norm = go.Figure()
    palette = [C['teal'],C['blue'],C['purple'],C['amber'],C['orange'],
               C['coral'],C['green'],C['pink'],C['coral'],C['blue']]
    for i,(tkr,vals) in enumerate(norm_data.items()):
        n = min(len(vals), idx_len)
        x_idx = list(range(n))
        lw = 2.5 if tkr == selected_ticker else 1.0
        op = 1.0 if tkr == selected_ticker else 0.45
        fig_norm.add_trace(go.Scatter(x=x_idx, y=vals[:n], mode='lines', name=tkr,
            line=dict(color=TICKER_COLORS.get(tkr,palette[i%len(palette)]),
                      width=lw), opacity=op))
    apply_layout(fig_norm, 'Normalised Total Return (Base = 100)',
        height=300, legend_h=True)
    fig_norm.update_xaxes(title_text='Trading Days')
    fig_norm.update_yaxes(title_text='Indexed Price')

    # Vol bar chart
    tickers_sorted = sorted(COMPANIES.keys(), key=lambda t: float(rows[[r['Ticker'] for r in rows].index(t)]['HV_21'].strip('%')))
    hv_vals = [float(rows[[r['Ticker'] for r in rows].index(t)]['HV_21'].strip('%')) for t in tickers_sorted]
    colors_bar = [TICKER_COLORS.get(t, C['teal']) for t in tickers_sorted]
    opacity_bar = [1.0 if t == selected_ticker else 0.5 for t in tickers_sorted]
    fig_vol = go.Figure(go.Bar(x=tickers_sorted, y=hv_vals,
        marker=dict(color=colors_bar, opacity=opacity_bar, line=dict(width=0)),
        text=[f'{v:.1f}%' for v in hv_vals], textposition='outside',
        textfont=dict(color=C['muted'], size=11)))
    apply_layout(fig_vol, '21-Day Historical Volatility (Annualised)', height=280)
    fig_vol.update_yaxes(title_text='Vol (%)', ticksuffix='%')

    # selected-strike option price as % of spot
    atm_pct_vals = [float(rows[[r['Ticker'] for r in rows].index(t)]['Premium_Pct'].strip('%')) for t in tickers_sorted]
    fig_atm = go.Figure(go.Bar(x=tickers_sorted, y=atm_pct_vals,
        marker=dict(color=colors_bar, opacity=opacity_bar, line=dict(width=0)),
        text=[f'{v:.2f}%' for v in atm_pct_vals], textposition='outside',
        textfont=dict(color=C['muted'], size=11)))
    apply_layout(fig_atm, f'{K_pct:.0f}% Strike {opt_type.title()} Price as % of Spot  (T={T*12:.0f}mo, r={r*100:.2f}%)', height=280)
    fig_atm.update_yaxes(title_text='Premium (%)', ticksuffix='%')

    # Summary table
    tbl = dash_table.DataTable(
        data=rows,
        columns=[{'name': c, 'id': c} for c in ['Ticker', 'Name', 'Sector', 'Spot', 'Selected_K', 'HV_21', 'Option_Price', 'Premium_Pct', 'Persistence']],
        style_table={'overflowX': 'auto', 'borderRadius': '8px', 'overflow': 'hidden'},
        style_cell={
            'backgroundColor': C['surface'],
            'color': C['text'],
            'border': f"1px solid {C['border']}",
            'fontFamily': "'JetBrains Mono',monospace",
            'fontSize': '11px',
            'padding': '8px 14px',
            'textAlign': 'left',
        },
        style_header={
            'backgroundColor': C['surface2'],
            'color': C['blue'],
            'fontWeight': '600',
            'fontSize': '10px',
            'letterSpacing': '0.07em',
            'border': f"1px solid {C['border']}",
        },
        style_data_conditional=[  # pyright: ignore[reportArgumentType]
            {
                'if': {'filter_query': '{{Ticker}} = "{}"'.format(selected_ticker)},
                'backgroundColor': f'{C["blue"]}15',
                'border': f'1px solid {C["blue"]}',
            },
        ],
        sort_action='native',
        page_size=10,
    )

    return html.Div([intro,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_norm, config={'displayModeBar':False}), width=12)
        ], className='mb-2'),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_vol, config={'displayModeBar':False}), width=6),
            dbc.Col(dcc.Graph(figure=fig_atm, config={'displayModeBar':False}), width=6),
        ], className='mb-3'),
        html.Div([html.Div('Full Summary Table',style=LBL|{'marginBottom':'10px'}), tbl], style=CARD),
        chart_note(
            ('Normalised returns index all prices to 100 at start. The selected company (brighter line) '
             'is highlighted. High-vol names diverge more from the baseline over time.', 4),
            ('Vol bar: sorted low-to-high. High-vol names (TSLA, NVDA) require larger option premiums. '
             'The selected ticker is shown at full opacity.', 4),
            ('selected-strike option premium as % of spot directly compares option expensiveness across price levels. '
             'It scales with vol — a cheap stock with high vol can be more expensive to option than a '
             'pricier low-vol name.', 4),
        ),
    ])




FONTS = ('https://fonts.googleapis.com/css2?'
         'family=Inter:wght@300;400;500;600&'
         'family=JetBrains+Mono:wght@400;500&display=swap')

CUSTOM_CSS = r"""
.tab-bar-wrap {
    display: flex;
    align-items: stretch;
    padding: 0 36px;
    background: rgba(22, 27, 34, 0.92);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid #30363d;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 4px 32px rgba(0,0,0,0.5);
}
.tab-btn {
    position: relative;
    background: transparent;
    border: none;
    padding: 12px 28px 12px 20px;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: #8b949e;
    transition: color 0.22s ease, background 0.22s ease;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 2px;
    overflow: hidden;
    outline: none;
    white-space: nowrap;
}
.tab-btn::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: currentColor;
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease;
    opacity: 0;
    border-radius: 2px 2px 0 0;
}
.tab-shimmer {
    position: absolute;
    top: 0;
    left: -100%;
    width: 60%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
    transform: skewX(-20deg);
    pointer-events: none;
}
.tab-btn:hover .tab-shimmer { animation: shimmer-slide 0.55s ease forwards; }
@keyframes shimmer-slide { to { left: 220%; } }
.tab-btn:hover {
    color: #c9d1d9;
    background: rgba(255,255,255,0.025);
}
.tab-btn:hover::after { transform: scaleX(0.5); opacity: 0.35; }
.tab-btn--active { font-weight: 600; background: rgba(255,255,255,0.035); }
.tab-btn--active::after {
    transform: scaleX(1) !important;
    opacity: 1 !important;
    box-shadow: 0 0 10px currentColor, 0 0 4px currentColor;
}
.tab-btn--active[data-color="#39d0a0"] { color: #39d0a0 !important; }
.tab-btn--active[data-color="#a78bfa"] { color: #a78bfa !important; }
.tab-btn--active[data-color="#fbbf24"] { color: #fbbf24 !important; }
.tab-btn--active[data-color="#fb7185"] { color: #fb7185 !important; }
.tab-btn--active[data-color="#60a5fa"] { color: #60a5fa !important; }
.tab-phase-lbl {
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.45;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 400;
    line-height: 1;
}
.tab-main-lbl { font-size: 13px; line-height: 1; }
.content-panel { animation: content-in 0.32s cubic-bezier(0.4, 0, 0.2, 1) both; }
@keyframes content-in {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.dark-dropdown .Select-control,
.dark-dropdown .Select-menu-outer {
    background: #0d1117 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}
.rc-slider-track { background: rgba(96,165,250,0.5) !important; }
.rc-slider-handle { border: 2px solid #60a5fa !important; box-shadow: 0 0 8px rgba(96,165,250,0.4) !important; }

/* Compact model control panel */
.model-control-grid {
    display: grid;
    grid-template-columns: minmax(250px, 1.35fr) minmax(132px, 0.75fr) minmax(260px, 1.4fr) minmax(230px, 1.25fr) minmax(230px, 1fr);
    gap: 12px;
    align-items: stretch;
    width: 100%;
}
.control-card {
    background: rgba(13, 17, 23, 0.72);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 10px 12px 8px 12px;
    min-height: 76px;
    overflow: visible;
}
.control-card--slider { padding-bottom: 4px; }
.control-card .Select-control {
    min-height: 34px !important;
    height: 34px !important;
    border-radius: 8px !important;
}
.control-card .Select-placeholder,
.control-card .Select-value-label {
    line-height: 32px !important;
    font-size: 12px !important;
}
.compact-input {
    height: 34px;
}
.segmented-control label {
    margin: 0 6px 6px 0 !important;
    padding: 4px 8px;
    border: 1px solid #30363d;
    border-radius: 999px;
    background: rgba(255,255,255,0.025);
    white-space: nowrap;
}
.segmented-control input { margin-right: 4px; }
.control-help {
    color: #8b949e;
    font-size: 10px;
    margin-top: 2px;
    line-height: 1.25;
}
@media (max-width: 1280px) {
    .model-control-grid {
        grid-template-columns: minmax(230px, 1.2fr) minmax(128px, 0.7fr) minmax(250px, 1.3fr) minmax(210px, 1.1fr) minmax(220px, 1fr);
    }
}

html, body {
    overflow-y: auto !important;
    overflow-x: auto !important;
    height: auto !important;
}
body { min-width: 1220px; }
#_dash-app-content, #react-entry-point {
    overflow: visible !important;
    min-width: 1220px;
}
.content-wrap { width: 100%; max-width: none !important; }
.metric-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
    gap: 8px;
    align-items: stretch;
    margin-bottom: 16px;
}
.metric-col > div { min-height: 86px; }
.dash-graph { min-width: 0; }
.js-plotly-plot, .plot-container, .svg-container { max-width: 100%; }
@media (max-width: 1220px) {
    .tab-bar-wrap, .dashboard-header, .dashboard-controls, .dashboard-footer { min-width: 1220px; }
}
'
"""

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, FONTS],
    title='Multi-Company Options Dashboard', suppress_callback_exceptions=True)
server = app.server

# Inject custom CSS into the HTML template (dash.html has no Style component)
app.index_string = (
    '<!DOCTYPE html>\n'
    '<html>\n'
    '    <head>\n'
    '        {%metas%}\n'
    '        <title>{%title%}</title>\n'
    '        {%favicon%}\n'
    '        {%css%}\n'
    '        <style>' + CUSTOM_CSS + '</style>\n'
    '    </head>\n'
    '    <body>\n'
    '        {%app_entry%}\n'
    '        <footer>\n'
    '            {%config%}\n'
    '            {%scripts%}\n'
    '            {%renderer%}\n'
    '        </footer>\n'
    '    </body>\n'
    '</html>\n'
)

TABS_CONFIG = [
    ('data',    '01', 'Market Data',    C['teal'],   'Phase 1'),
    ('binom',   '02', 'Binomial',       C['purple'], 'Phase 2'),
    ('bs',      '03', 'Black-Scholes',  C['amber'],  'Phase 3'),
    ('garch',   '04', 'GARCH Forecast', C['coral'],  'Phase 4'),
    ('compare', '05', 'Cross-Company',  C['blue'],   'Phase 5'),
]

TICKER_OPTIONS = [{'label': f'{t}  --  {m["name"]}', 'value': t} for t, m in COMPANIES.items()]

app.layout = html.Div([

    # HEADER
    html.Div([
        html.Div([
            html.Div([
                html.Span('Multi-Company ', style={'color':C['blue'],'fontSize':'24px','fontWeight':'700','fontFamily':MONO}),
                html.Span('Options Pricing Dashboard', style={'color':C['text'],'fontSize':'20px','fontWeight':'500'}),
            ]),
            html.Div('Binomial  ·  Black-Scholes  ·  GARCH(1,1)  ·  10-Company Cross-Comparison',
                style={'color':C['muted'],'fontSize':'12px','marginTop':'4px'}),
        ]),
        html.Div([
            html.Div('Muhideen Ogunlowo', style={'color':C['blue'],'fontSize':'13px','fontWeight':'600','letterSpacing':'0.03em','textAlign':'right','marginBottom':'3px'}),
            html.Div('Equities Options Pricing Project', style={'color':C['muted'],'fontSize':'11px','textAlign':'right'}),
        ]),
    ], className='dashboard-header', style={
        'display':'flex','justifyContent':'space-between','alignItems':'center',
        'padding':'16px 36px',
        'background':f'linear-gradient(135deg, {C["surface"]} 0%, #1a1f28 100%)',
        'borderBottom':f'1px solid {C["border"]}',
        'boxShadow':'0 2px 16px rgba(0,0,0,0.4)',
    }),

    # CONTROLS — compact grid so Company / Strike / RF / Maturity / Type do not feel clunky
    html.Div([
        html.Div([
            html.Div([
                html.Div('Company', style=LBL),
                dcc.Dropdown(
                    id='ticker', options=TICKER_OPTIONS, value='DIS', clearable=False,
                    className='dark-dropdown',
                    style={'background': C['bg'], 'color': C['text'], 'fontSize': '12px'},
                ),
                html.Div('Changing the company updates spot, volatility, strike, and all model tabs.', className='control-help'),
            ], className='control-card'),

            html.Div([
                html.Div('Risk-Free Rate (%)', style=LBL),
                dcc.Input(
                    id='rf', type='number', value=5.25, min=0, max=20, step=0.05,
                    debounce=False, className='compact-input',
                    style={'background': C['bg'], 'border': f'1px solid {C["border"]}',
                           'color': C['text'], 'borderRadius': '8px', 'padding': '6px 10px',
                           'width': '100%', 'fontFamily': MONO, 'fontSize': '12px'}
                ),
                html.Div('Used as r in Binomial, B-S, GARCH-BS, and comparison.', className='control-help'),
            ], className='control-card'),

            html.Div([
                html.Div('Strike (% of Spot)', style=LBL),
                dcc.Slider(
                    id='strike-pct', min=70, max=130, step=5, value=100, updatemode='drag',
                    marks={k: {'label': f'{k}%', 'style': {'color': C['muted'], 'fontSize': '10px'}}
                           for k in [70, 85, 100, 115, 130]},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                ),
            ], className='control-card control-card--slider'),

            html.Div([
                html.Div('Maturity (months)', style=LBL),
                dcc.Slider(
                    id='maturity', min=1, max=24, step=1, value=6, updatemode='drag',
                    marks={k: {'label': str(k), 'style': {'color': C['muted'], 'fontSize': '10px'}}
                           for k in [1, 3, 6, 12, 18, 24]},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                ),
            ], className='control-card control-card--slider'),

            html.Div([
                html.Div('Option Type / Exercise', style=LBL),
                dcc.RadioItems(
                    id='opt-type',
                    options=[{'label': 'Call', 'value': 'call'}, {'label': 'Put', 'value': 'put'}],
                    value='call', inline=True, className='segmented-control',
                    labelStyle={'color': C['text'], 'cursor': 'pointer', 'fontSize': '12px'},
                ),
                dcc.RadioItems(
                    id='exercise',
                    options=[{'label': 'European', 'value': 'european'}, {'label': 'American', 'value': 'american'}],
                    value='european', inline=True, className='segmented-control',
                    labelStyle={'color': C['text'], 'cursor': 'pointer', 'fontSize': '12px'},
                ),
            ], className='control-card'),
        ], className='model-control-grid'),
    ], className='dashboard-controls', style={
        'padding': '12px 36px', 'background': C['surface'],
        'borderBottom': f'1px solid {C["border"]}', 'minWidth': '1220px'
    }),

    # GLOSSY CUSTOM TAB BAR
    html.Div([
        html.Button(
            [
                html.Span(phase, className='tab-phase-lbl'),
                html.Span(label, className='tab-main-lbl'),
                html.Span(className='tab-shimmer')
            ],
            id=f'tab-btn-{val}',
            className='tab-btn' + (' tab-btn--active' if i == 0 else ''),
            n_clicks=0,
            **{'data-color': color, 'data-tab': val},  # type: ignore[arg-type]
        )
        for i, (val, num, label, color, phase) in enumerate(TABS_CONFIG)
    ], className='tab-bar-wrap'),

    # CONTENT AREA
    dcc.Loading(
        html.Div(id='content', style={'padding':'22px 36px','overflowX':'auto','width':'100%','minWidth':'1220px'}),
        type='circle', color=C['blue'],
    ),

    # FOOTER
    html.Div([
        html.Div('Muhideen Ogunlowo  |  Multi-Company Equities Options Pricing  |  Binomial · Black-Scholes · GARCH(1,1)',
            style={'color':C['muted'],'fontSize':'11px','textAlign':'center'}),
    ], className='dashboard-footer', style={'padding':'14px 36px','borderTop':f'1px solid {C["border"]}','background':C['surface'], 'minWidth':'1220px'}),

    dcc.Store(id='store'),
    dcc.Store(id='active-tab', data='data'),

], style={'background':C['bg'],'fontFamily':'Inter,sans-serif','color':C['text'],'minHeight':'100vh','minWidth':'1220px'})


# ── Clientside callback: instant tab-button highlight ────────────
app.clientside_callback(
    """
function(n1, n2, n3, n4, n5, current) {
    var ctx = window.dash_clientside ? window.dash_clientside.callback_context : null;
    if (!ctx || !ctx.triggered || !ctx.triggered.length) {
        return current || 'data';
    }
    var prop_id = ctx.triggered[0].prop_id;
    var tabMap = {
        'tab-btn-data.n_clicks':    'data',
        'tab-btn-binom.n_clicks':   'binom',
        'tab-btn-bs.n_clicks':      'bs',
        'tab-btn-garch.n_clicks':   'garch',
        'tab-btn-compare.n_clicks': 'compare'
    };
    var newTab = tabMap[prop_id];
    if (!newTab) { return current || 'data'; }
    document.querySelectorAll('.tab-btn').forEach(function(btn) {
        btn.classList.remove('tab-btn--active');
    });
    var activeBtn = document.getElementById('tab-btn-' + newTab);
    if (activeBtn) { activeBtn.classList.add('tab-btn--active'); }
    return newTab;
}
""",
    Output('active-tab', 'data'),
    [Input(f'tab-btn-{val}', 'n_clicks') for val, *_ in TABS_CONFIG],
    State('active-tab', 'data'),
    prevent_initial_call=False,
)


# ── Single render callback ────────────────────────────────────────
# Parameters flow DIRECTLY into this callback — no store intermediary.
# _mock_data(ticker) is seeded by ticker hash so always deterministic.
# Every slider/dropdown/radio change immediately triggers a fresh render.
@app.callback(
    Output('content', 'children'),
    Input('active-tab', 'data'),
    Input('ticker',     'value'),
    Input('strike-pct', 'value'),
    Input('maturity',   'value'),
    Input('opt-type',   'value'),
    Input('exercise',   'value'),
    Input('rf',         'value'),
)
def render_tab(tab, ticker, k_pct, months, opt_type, exercise, rf):
    # ── Normalise inputs ──────────────────────────────────────────
    tab      = tab      or 'data'
    ticker   = ticker   or 'DIS'
    k_pct    = k_pct    if k_pct    is not None else 100
    months   = months   if months   is not None else 6
    opt_type = opt_type or 'call'
    exercise = exercise or 'european'
    rf       = rf       if rf       is not None else 5.25

    r   = rf / 100
    T   = months / 12
    K_pct = float(k_pct)
    acc = TICKER_COLORS.get(ticker, C['teal'])

    # ── Load price data directly — no store/JSON round-trip ───────
    df    = _mock_data(ticker)
    S     = float(df['Close'].iloc[-1])
    sigma = float(df['HV_21'].iloc[-1])
    K     = S * (K_pct / 100)

    # ── Route to the correct phase ────────────────────────────────
    try:
        if tab == 'data':
            content = render_market(df, S, sigma, ticker, acc)
        elif tab == 'binom':
            content = render_binomial(S, K, T, r, sigma, opt_type, exercise, ticker, acc)
        elif tab == 'bs':
            content = render_bs(S, K, T, r, sigma, opt_type, exercise, ticker, acc)
        elif tab == 'garch':
            content = render_garch(df, S, K, T, r, opt_type, ticker, acc)
        elif tab == 'compare':
            content = render_comparison(ticker, r, opt_type, T, K_pct)
        else:
            content = html.Div()
    except Exception as exc:
        import traceback
        content = html.Div([
            html.Div(f'Error rendering phase: {exc}',
                     style={'color': C['coral'], 'padding': '20px', 'fontFamily': MONO}),
            html.Pre(traceback.format_exc(),
                     style={'color': C['muted'], 'fontSize': '11px', 'padding': '0 20px'}),
        ])

    return html.Div(content, className='content-panel', key=f'{tab}-{ticker}-{K_pct:.2f}-{T:.4f}-{r:.5f}-{opt_type}-{exercise}')


app = dash.Dash(__name__)
server = app.server

# app.layout = ...
# callbacks = ...

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8090))
    debug = os.environ.get('DASH_DEBUG', 'False').lower() in {'1', 'true', 'yes'}
    app.run(host='0.0.0.0', port=port, debug=debug)
