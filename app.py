

import streamlit as st
import json, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone

from model import fetch_btc_bars, GBMPredictor, winkler_score

st.set_page_config(page_title="BTC Predictor", page_icon="₿", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Clash+Display:wght@400;500;600;700&display=swap');

html,body,[class*="css"],.stApp { 
    background: radial-gradient(circle at top right, #0a0d14, #030406 60%, #000000) !important;
    color: #C9D1D9 !important; 
    font-family: 'Outfit', sans-serif !important; 
}
#MainMenu,footer,header { visibility:hidden; }
.main .block-container { padding: 2rem 3rem; max-width: 1500px; }
::-webkit-scrollbar{width:6px} 
::-webkit-scrollbar-track{background:transparent} 
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:10px}

/* Glassmorphism utility */
.glass-panel {
    background: rgba(13, 17, 23, 0.45);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
}
.glass-panel:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
    border-color: rgba(255, 255, 255, 0.1);
}

.site-header{
    display:flex;align-items:flex-end;justify-content:space-between;
    padding: 16px 20px;
    margin-bottom:32px;
    position: sticky; top: 0px; z-index: 1000;
    background: rgba(3, 4, 6, 0.7);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 0 0 16px 16px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.5);
}
.site-title{font-family:'Bebas Neue',sans-serif;font-size:3.2rem;letter-spacing:.06em;line-height:1;color:#ffffff; text-shadow: 0 0 20px rgba(255,255,255,0.2);}
.site-title span{color:#F7931A; text-shadow: 0 0 20px rgba(247,147,26,0.4);}
.site-meta{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:#8B949E;letter-spacing:.12em;margin-top:6px;text-transform:uppercase}
.live-pill{display:flex;align-items:center;gap:8px;background:rgba(63,185,80,0.1);border:1px solid rgba(63,185,80,0.3);border-radius:30px;padding:8px 16px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#3FB950;letter-spacing:.15em; box-shadow: 0 0 15px rgba(63,185,80,0.2);}
.live-dot{width:7px;height:7px;border-radius:50%;background:#3FB950;box-shadow:0 0 10px #3FB950;animation:blink 1.8s infinite}
@keyframes blink{0%,100%{opacity:1;box-shadow:0 0 12px #3FB950}50%{opacity:.3;box-shadow:none}}

.sec-label{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.25em;text-transform:uppercase;color:#8B949E;padding-bottom:12px;border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:20px; display:flex; align-items:center;}
.sec-label::before { content:''; display:inline-block; width:6px; height:6px; background:#F7931A; border-radius:50%; margin-right:12px; box-shadow: 0 0 8px #F7931A; }

.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin-bottom:32px}
.stat-card{background: rgba(13, 17, 23, 0.45); backdrop-filter: blur(16px); border:1px solid rgba(255,255,255,0.05);border-radius:16px;padding:24px;position:relative; overflow:hidden; transition: all 0.3s ease; box-shadow: 0 8px 32px 0 rgba(0,0,0,0.2);}
.stat-card:hover{transform: translateY(-3px); box-shadow: 0 12px 40px 0 rgba(0,0,0,0.4); border-color:rgba(255,255,255,0.15);}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg, transparent, #F7931A, transparent); opacity:0.8;}
.stat-card.green::before{background:linear-gradient(90deg, transparent, #3FB950, transparent);}
.stat-card.blue::before{background:linear-gradient(90deg, transparent, #58A6FF, transparent);}
.stat-card.gray::before{background:linear-gradient(90deg, transparent, #8B949E, transparent);}

.stat-label{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.15em;color:#8B949E;text-transform:uppercase;margin-bottom:10px; font-weight:500;}
.stat-value{font-family:'Bebas Neue',sans-serif;font-size:2.8rem;letter-spacing:.05em;line-height:1;color:#ffffff; text-shadow: 0 2px 10px rgba(255,255,255,0.1);}
.stat-value.orange{color:#F7931A; text-shadow: 0 2px 15px rgba(247,147,26,0.3);}
.stat-value.green{color:#3FB950; text-shadow: 0 2px 15px rgba(63,185,80,0.3);}
.stat-value.red{color:#F85149; text-shadow: 0 2px 15px rgba(248,81,73,0.3);}
.stat-sub{font-family:'IBM Plex Mono',monospace;font-size:10px;color:#6E7681;margin-top:8px;}

.price-panel{border-radius:16px;padding:28px;margin-bottom:20px;}
.price-now-label{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.2em;color:#8B949E;text-transform:uppercase;margin-bottom:8px;}
.price-now-val{font-family:'Bebas Neue',sans-serif;font-size:4.5rem;letter-spacing:.04em;color:#ffffff;line-height:1; text-shadow: 0 0 30px rgba(255,255,255,0.15);}
.price-chg-pos{font-family:'IBM Plex Mono',monospace;font-size:13px;color:#3FB950;margin-top:10px; font-weight:600; letter-spacing:0.05em; text-shadow: 0 0 10px rgba(63,185,80,0.3);}
.price-chg-neg{font-family:'IBM Plex Mono',monospace;font-size:13px;color:#F85149;margin-top:10px; font-weight:600; letter-spacing:0.05em; text-shadow: 0 0 10px rgba(248,81,73,0.3);}

.range-panel{border-radius:16px;padding:28px;margin-bottom:20px;text-align:center;}
.range-title{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.25em;color:#8B949E;text-transform:uppercase;margin-bottom:20px;}
.range-lo{font-family:'Bebas Neue',sans-serif;font-size:3rem;letter-spacing:.05em;color:#3FB950; text-shadow: 0 0 20px rgba(63,185,80,0.4);}
.range-arrow{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;color:#F7931A;margin:0 16px; opacity:0.7;}
.range-hi{font-family:'Bebas Neue',sans-serif;font-size:3rem;letter-spacing:.05em;color:#F85149; text-shadow: 0 0 20px rgba(248,81,73,0.4);}
.range-meta{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8B949E;margin-top:16px;line-height:2;}
.range-meta b{color:#C9D1D9;font-weight:600}

.badge{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.15em;text-transform:uppercase;padding:6px 16px;border-radius:6px;margin-top:16px; font-weight:600;}
.badge-bull{background:rgba(63,185,80,.15);color:#3FB950;border:1px solid rgba(63,185,80,.4); box-shadow: 0 0 15px rgba(63,185,80,0.2);}
.badge-bear{background:rgba(248,81,73,.15);color:#F85149;border:1px solid rgba(248,81,73,.4); box-shadow: 0 0 15px rgba(248,81,73,0.2);}
.badge-neutral{background:rgba(139,148,158,.15);color:#C9D1D9;border:1px solid rgba(139,148,158,.3);}

.params-box{border-radius:16px;padding:24px;font-family:'IBM Plex Mono',monospace;font-size:11px;line-height:2.2;color:#8B949E;}
.pk{color:#6E7681;} .pv{color:#E6EDF3; font-weight:500;}

.mini-metric{border-radius:12px;padding:16px 20px;text-align:center;}
.mini-label{font-family:'IBM Plex Mono',monospace;font-size:9.5px;letter-spacing:.15em;color:#8B949E;text-transform:uppercase;margin-bottom:8px;}
.mini-val{font-family:'Bebas Neue',sans-serif;font-size:2rem;letter-spacing:.05em;color:#ffffff;}
.mini-val.green{color:#3FB950; text-shadow: 0 0 10px rgba(63,185,80,0.3);} 
.mini-val.red{color:#F85149; text-shadow: 0 0 10px rgba(248,81,73,0.3);}

/* Table overriding */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05); }

</style>
""", unsafe_allow_html=True)


HISTORY_FILE     = "prediction_history.jsonl"
BACKTEST_SUMMARY = "backtest_summary.json"
CONFIDENCE       = 0.975
VOL_WINDOW       = 24

def fp(p): return f"${p:,.0f}"

def plotly_base():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="IBM Plex Mono", size=10, color="#8B949E"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)", tickcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.03)", tickcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)", zeroline=False,
                   tickprefix="$", side="right"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1, font=dict(size=10, color="#C9D1D9")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(13,17,23,0.9)", font_size=12, font_family="IBM Plex Mono")
    )

def load_history():
    recs = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            for line in f:
                s = line.strip()
                if s:
                    try: recs.append(json.loads(s))
                    except: pass
    return recs

def save_prediction(rec):
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")

def is_duplicate(rec, hist):
    return bool(hist and hist[-1].get("bar_time") == rec.get("bar_time"))


@st.cache_data(ttl=55, show_spinner=False)
def get_live_data():
    return fetch_btc_bars(limit=505)

with st.spinner("Fetching Live Market Data..."):
    try:
        df = get_live_data()
    except Exception as e:
        st.error(f"Binance fetch failed: {e}")
        st.stop()

closes = df["close"].values
predictor = GBMPredictor(vol_window=VOL_WINDOW, n_sims=10_000, confidence=CONFIDENCE)
try:
    pred = predictor.predict(closes, horizon=1)
except Exception as e:
    st.error(f"Model error: {e}")
    st.stop()

current_price = pred["current_price"]
lower, upper  = pred["lower"], pred["upper"]
width         = pred["width"]
mu, sigma, dof = pred["mu"], pred["sigma"], pred["dof"]
last_bar_time  = df["open_time"].iloc[-1].isoformat()
now_str        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
pct_change     = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) > 1 else 0


history = load_history()
new_rec = {
    "bar_time": last_bar_time, "lower": round(lower,2), "upper": round(upper,2),
    "current_price": round(current_price,2), "sigma": round(sigma,8),
    "mu": round(mu,10), "dof": round(dof,2), "confidence": CONFIDENCE,
    "width": round(width,2), "predicted_at": now_str, "actual": None,
}
if history:
    changed = False
    for rec in history:
        if rec.get("actual") is None:
            try:
                bar_idx = df[df["open_time"] == pd.Timestamp(rec.get("bar_time"))].index
                if len(bar_idx) > 0:
                    nxt = bar_idx[0] + 1
                    if nxt < len(df):
                        rec["actual"] = round(float(df["close"].iloc[nxt]), 2)
                        changed = True
            except: pass
    if changed:
        with open(HISTORY_FILE, "w") as f:
            for rec in history: f.write(json.dumps(rec) + "\n")

if not is_duplicate(new_rec, history):
    save_prediction(new_rec)
    history = load_history()

bt = None
if os.path.exists(BACKTEST_SUMMARY):
    with open(BACKTEST_SUMMARY) as f:
        bt = json.load(f)




st.markdown(f"""
<div class="site-header">
  <div>
    <div class="site-title">₿ BTC <span>Range</span> Predictor</div>
    <div class="site-meta">Live Prediction Dashboard</div>
  </div>
  <div class="live-pill"><div class="live-dot"></div>LIVE &nbsp;·&nbsp; {now_str}</div>
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="sec-label">30-Day Backtest Performance</div>', unsafe_allow_html=True)
if bt:
    cov = bt.get("coverage_95", 0)
    mw  = bt.get("mean_width", 0)
    wk  = bt.get("mean_winkler_95", 0)
    n   = bt.get("n_predictions", 0)
    cov_cls = "green" if abs(cov-0.95)<0.025 else ("orange" if abs(cov-0.95)<0.05 else "red")
    cov_bar = "green" if abs(cov-0.95)<0.025 else ""
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card {cov_bar}">
        <div class="stat-label">Coverage · Target 0.95</div>
        <div class="stat-value {cov_cls}">{cov:.4f}</div>
        <div class="stat-sub">{"✓ calibrated" if abs(cov-0.95)<0.025 else "⚠ off-target"}</div>
      </div>
      <div class="stat-card blue">
        <div class="stat-label">Mean Range Width</div>
        <div class="stat-value">{fp(mw)}</div>
        <div class="stat-sub">avg 95% interval</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Mean Winkler Score</div>
        <div class="stat-value">{wk:,.0f}</div>
        <div class="stat-sub">lower = better forecaster</div>
      </div>
      <div class="stat-card gray">
        <div class="stat-label">Backtest Bars</div>
        <div class="stat-value">{n:,}</div>
        <div class="stat-sub">Predictive Engine</div>
      </div>
    </div>""", unsafe_allow_html=True)
else:
    st.info("Run `python backtest.py` to populate metrics.")


st.markdown('<div class="sec-label">Live Prediction — Next 1-Hour Bar</div>', unsafe_allow_html=True)
col_l, col_r = st.columns([3, 2], gap="large")

with col_l:
    chg_cls = "price-chg-pos" if pct_change >= 0 else "price-chg-neg"
    arrow   = "▲" if pct_change >= 0 else "▼"
    sig_txt = "BULLISH DRIFT" if mu>0.0001 else ("BEARISH DRIFT" if mu<-0.0001 else "NEUTRAL")
    sig_cls = "badge-bull" if mu>0.0001 else ("badge-bear" if mu<-0.0001 else "badge-neutral")
    mid     = (lower + upper) / 2
    vol_ann = sigma * (8760**0.5) * 100

    st.markdown(f"""
    <div class="price-panel glass-panel">
      <div class="price-now-label">BTC / USDT · Current Price</div>
      <div class="price-now-val">{fp(current_price)}</div>
      <div class="{chg_cls}">{arrow} {abs(pct_change):.3f}% from previous bar</div>
    </div>

    <div class="range-panel glass-panel">
      <div class="range-title">95% Confidence Interval — Next Hour</div>
      <span class="range-lo">{fp(lower)}</span>
      <span class="range-arrow">→</span>
      <span class="range-hi">{fp(upper)}</span>
      <div class="range-meta">
        <b>Width</b> {fp(width)} &nbsp;&nbsp; <b>Midpoint</b> {fp(mid)}
      </div>
      <span class="badge {sig_cls}">{sig_txt}</span>
    </div>

    <div class="params-box glass-panel">
      <span style="font-size:10px;letter-spacing:.25em;color:#8B949E;text-transform:uppercase;font-weight:600;display:block;margin-bottom:12px;">Market Metrics</span>
      <span class="pk">Hourly Drift &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="pv">{mu*100:+.3f}%</span><br>
      <span class="pk">Hourly Volatility &nbsp;&nbsp;</span><span class="pv">{sigma*100:.3f}%</span><br>
      <span class="pk">Annual Volatility &nbsp;&nbsp;</span><span class="pv">{vol_ann:.1f}%</span><br>
      <span class="pk">Last Update &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="pv">{df["open_time"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")}</span>
    </div>
    """, unsafe_allow_html=True)

with col_r:
    rng   = np.random.default_rng(42)
    std_f = (dof/(dof-2))**0.5 if dof>2 else 1.0
    z_sim = rng.standard_t(dof, size=8000) / std_f
    sims  = current_price * np.exp(mu - 0.5*sigma**2 + sigma*z_sim)

    fig_d = go.Figure()
    fig_d.add_trace(go.Histogram(x=sims, nbinsx=90,
        marker_color="rgba(247,147,26,0.6)",
        marker_line_color="rgba(247,147,26,1)", marker_line_width=1,
        showlegend=False))
    

    fig_d.add_vrect(x0=lower, x1=upper, fillcolor="rgba(63,185,80,0.1)",
        line_color="rgba(63,185,80,0.5)", line_width=1.5, annotation_text="95% CI", annotation_position="top left", annotation_font_size=10, annotation_font_color="#3FB950")
    
    fig_d.add_vline(x=current_price, line_color="#F7931A", line_width=2,
        annotation_text="NOW", annotation_font_color="#F7931A",
        annotation_font_size=10, annotation_position="top right")
    
    fig_d.add_vline(x=lower, line_color="#3FB950", line_width=1.5, line_dash="dot")
    fig_d.add_vline(x=upper, line_color="#F85149", line_width=1.5, line_dash="dot")
    
    pb = plotly_base()
    y_ax = pb["yaxis"].copy()
    y_ax.update(showticklabels=False, tickprefix="")
    pb.update(height=260,
        xaxis=dict(**pb["xaxis"], title="Simulated Price ($)", tickformat=",d"),
        yaxis=y_ax,
        title=dict(text="MONTE CARLO DISTRIBUTION", font=dict(size=11, color="#C9D1D9", family="IBM Plex Mono"), x=0.5))
    fig_d.update_layout(**pb)
    

    st.markdown('<div class="glass-panel" style="border-radius:16px; padding: 16px; margin-bottom: 20px;">', unsafe_allow_html=True)
    st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


    vol_norm  = min(sigma / 0.006, 1.0)
    vol_label = "LOW VOLATILITY" if sigma<0.0015 else ("MEDIUM VOLATILITY" if sigma<0.003 else "HIGH VOLATILITY")
    vol_color = "#3FB950" if sigma<0.0015 else ("#F7931A" if sigma<0.003 else "#F85149")
    st.markdown(f"""
    <div class="params-box glass-panel" style="margin-top:10px;">
      <span style="font-size:10px;letter-spacing:.25em;color:#8B949E;text-transform:uppercase;font-weight:600;">Volatility Regime</span><br><br>
      <div style="display:flex;align-items:center;gap:12px;">
        <div style="flex:1;background:rgba(255,255,255,0.05);border-radius:6px;height:8px;overflow:hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);">
          <div style="width:{int(vol_norm*100)}%;height:100%;background:{vol_color};border-radius:6px; box-shadow: 0 0 10px {vol_color}; transition: width 1s ease;"></div>
        </div>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:{vol_color}; font-weight:600;">{vol_label}</span>
      </div>
      <br><span class="pk">Current Volatility </span><span class="pv" style="font-size:13px;">{sigma*100:.3f}%</span>
    </div>""", unsafe_allow_html=True)


st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div class="sec-label">Market Action & Prediction Ribbon</div>', unsafe_allow_html=True)

plot_df = df.tail(50).copy()
last_ts = df["open_time"].iloc[-1]
next_ts = last_ts + pd.Timedelta(hours=1)

fig_c = go.Figure()


fig_c.add_trace(go.Candlestick(
    x=plot_df["open_time"], open=plot_df["open"], high=plot_df["high"],
    low=plot_df["low"], close=plot_df["close"], name="BTCUSDT",
    increasing_line_color="#3FB950", decreasing_line_color="#F85149",
    increasing_fillcolor="rgba(63,185,80,0.9)", decreasing_fillcolor="rgba(248,81,73,0.9)",
))


fig_c.add_trace(go.Scatter(
    x=[last_ts,next_ts,next_ts,last_ts], y=[upper,upper,lower,lower],
    fill="toself", fillcolor="rgba(247,147,26,0.1)",
    line=dict(color="rgba(247,147,26,0.2)"), name="95% CI Band", hoverinfo="skip"))

fig_c.add_trace(go.Scatter(x=[last_ts,next_ts], y=[upper,upper],
    line=dict(color="#F85149", width=2, dash="dot"), mode="lines", name=f"Upper Bound"))
fig_c.add_trace(go.Scatter(x=[last_ts,next_ts], y=[lower,lower],
    line=dict(color="#3FB950", width=2, dash="dot"), mode="lines", name=f"Lower Bound"))
fig_c.add_trace(go.Scatter(x=[last_ts,next_ts], y=[(lower+upper)/2]*2,
    line=dict(color="#F7931A", width=1.5, dash="dash"), mode="lines", name=f"Midpoint"))

lc = plotly_base()
lc.update(height=500,
    xaxis=dict(**lc["xaxis"], rangeslider=dict(visible=False), type="date", tickformat="%b %d\\n%H:%M", showgrid=True),
    yaxis=dict(**lc["yaxis"], showgrid=True),
    legend=dict(orientation="h", y=1.05, x=1, xanchor="right", font=dict(size=11, color="#C9D1D9"), bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=40, t=20, b=20)
)
fig_c.update_layout(**lc)

st.markdown('<div class="glass-panel" style="border-radius:16px; padding: 20px;">', unsafe_allow_html=True)
st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
st.markdown('</div>', unsafe_allow_html=True)



st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec-label">Live Evaluation Log</div>', unsafe_allow_html=True)
history = load_history()

if not history:
    st.markdown('<div class="params-box glass-panel" style="text-align:center;padding:32px;">No live history logged yet. Stays active as long as app runs.</div>', unsafe_allow_html=True)
else:
    completed = [r for r in history if r.get("actual") is not None]
    if completed:
        hits     = [r for r in completed if r["lower"] <= r["actual"] <= r["upper"]]
        live_cov = len(hits) / len(completed)
        live_wk  = np.mean([winkler_score(r["lower"], r["upper"], r["actual"], 0.05) for r in completed])
        live_w   = np.mean([r["upper"] - r["lower"] for r in completed])
        cov_cls2 = "green" if abs(live_cov-0.95)<0.03 else "red"

        c1, c2, c3, c4 = st.columns(4, gap="large")
        for col, lbl, val, cls in [
            (c1, "Live Coverage",  f"{live_cov:.3f}", cov_cls2),
            (c2, "Hits / Total",   f"{len(hits)} / {len(completed)}", ""),
            (c3, "Live Avg Width", fp(live_w), ""),
            (c4, "Live Winkler",   f"{live_wk:,.0f}", ""),
        ]:
            with col:
                st.markdown(f'<div class="mini-metric glass-panel"><div class="mini-label">{lbl}</div><div class="mini-val {cls}">{val}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)


        hdf = pd.DataFrame(completed[-60:])
        hdf["bar_time"] = pd.to_datetime(hdf["bar_time"])
        hdf["hit"] = hdf.apply(lambda r: r["lower"] <= r["actual"] <= r["upper"], axis=1)

        fig_h = go.Figure()
        

        fig_h.add_trace(go.Scatter(
            x=list(hdf["bar_time"])+list(hdf["bar_time"])[::-1],
            y=list(hdf["upper"])+list(hdf["lower"])[::-1],
            fill="toself", fillcolor="rgba(247,147,26,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI Ribbon", hoverinfo="skip"))
            
        fig_h.add_trace(go.Scatter(x=hdf["bar_time"], y=hdf["upper"],
            line=dict(color="#F85149",width=1.5,dash="dot"), mode="lines", showlegend=False))
        fig_h.add_trace(go.Scatter(x=hdf["bar_time"], y=hdf["lower"],
            line=dict(color="#3FB950",width=1.5,dash="dot"), mode="lines", showlegend=False))

        hits_df   = hdf[hdf["hit"]]
        misses_df = hdf[~hdf["hit"]]
        if len(hits_df):
            fig_h.add_trace(go.Scatter(x=hits_df["bar_time"], y=hits_df["actual"],
                mode="markers", marker=dict(color="#3FB950",size=8,symbol="circle", line=dict(color="#ffffff", width=1)), name="Hit"))
        if len(misses_df):
            fig_h.add_trace(go.Scatter(x=misses_df["bar_time"], y=misses_df["actual"],
                mode="markers", marker=dict(color="#F85149",size=10,symbol="x", line=dict(color="#ffffff", width=1)), name="Miss"))

        lh = plotly_base()
        lh.update(height=300,
            xaxis=dict(**lh["xaxis"], tickformat="%m-%d %H:%M", showgrid=True),
            yaxis=dict(**lh["yaxis"], showgrid=True),
            legend=dict(orientation="h", y=1.1, font=dict(size=11,color="#C9D1D9"), bgcolor="rgba(0,0,0,0)"))
        fig_h.update_layout(**lh)
        
        st.markdown('<div class="glass-panel" style="border-radius:16px; padding: 20px;">', unsafe_allow_html=True)
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


    st.markdown('<div class="sec-label" style="margin-top:24px;">Recent Predictions Log</div>', unsafe_allow_html=True)
    rows = []
    for r in reversed(history[-30:]):
        actual = r.get("actual")
        lo, hi = r["lower"], r["upper"]
        if actual is not None:
            hit    = lo <= actual <= hi
            status = "HIT ✓" if hit else "MISS ✗"
            act_s  = f"${actual:,.0f}"
        else:
            status = "pending ⏳"
            act_s  = "—"
        rows.append({
            "Time (UTC)": pd.Timestamp(r["bar_time"]).strftime("%m-%d %H:%M"),
            "Lower":      f"${lo:,.0f}",
            "Upper":      f"${hi:,.0f}",
            "Width":      f"${hi-lo:,.0f}",
            "Actual":     act_s,
            "Result":     status,
            "Logged":     r.get("predicted_at","")[-12:-4] if r.get("predicted_at") else "",
        })

    tbl = pd.DataFrame(rows)

    def color_result(val):
        if "HIT"  in str(val): return "color: #3FB950; font-weight: 700"
        if "MISS" in str(val): return "color: #F85149; font-weight: 700"
        if "pending" in str(val): return "color: #F7931A; font-weight: 600"
        return "color: #8B949E"

    styled = (
        tbl.style
        .map(color_result, subset=["Result"])
        .set_properties(**{"font-family": "IBM Plex Mono, monospace", "font-size": "12px", "background-color": "transparent", "color": "#E6EDF3"})
        .set_table_styles([
            {"selector":"th","props":[("font-family","IBM Plex Mono, monospace"),("font-size","11px"),("letter-spacing","0.1em"),("text-transform","uppercase"),("color","#8B949E"),("border-bottom","1px solid rgba(255,255,255,0.1)"), ("background-color", "rgba(0,0,0,0)")]},
            {"selector":"td","props":[("border-bottom","1px solid rgba(255,255,255,0.03)")]},
            {"selector":"tr:hover","props":[("background-color","rgba(255,255,255,0.03)")]}
        ])
        .hide(axis="index")
    )
    
    st.markdown('<div class="glass-panel" style="border-radius:16px; padding: 1px;">', unsafe_allow_html=True)
    st.dataframe(styled, use_container_width=True, height=min(len(rows)*40+40, 450))
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown(f"""
<div style="margin-top:40px;padding-top:24px;border-top:1px solid rgba(255,255,255,0.05);text-align:center;
font-family:'IBM Plex Mono',monospace;font-size:10px;color:#6E7681;letter-spacing:.15em;line-height:2.4;">
  &copy; {datetime.now(timezone.utc).year} BTC Predictor &nbsp;·&nbsp; {now_str}
</div>
""", unsafe_allow_html=True)