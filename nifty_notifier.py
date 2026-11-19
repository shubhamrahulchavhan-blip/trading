#!/usr/bin/env python3
"""
NIFTY intraday signal notifier (free alternative to TradingView alerts)  -  v2

A Python port of the NIFTY Intraday Signal Engine (Pine strategy v2). It polls 5-minute
NIFTY data, evaluates the same gates / setups / weighted score on CLOSED candles only,
and sends an email when a BUY CE / BUY PE signal or a trade event occurs
(breakeven-stop move, target, stop-loss, opposite signal, time square-off).

v2 rules ported from the Pine strategy
- Breakeven shift: after +1R (closed bar) the stop moves to entry + 0.1R, from the NEXT bar.
  You get an email "MOVE STOP TO BREAKEVEN" - move your stop manually.
- Stop trading for the day after 2 consecutive losing trades.
- No new entries inside the square-off window; the session's last bar always flattens.
- Yahoo download retries (cloud IPs are sometimes rate-limited).

Notes
- It is a port, so small differences from TradingView are possible.
- Data comes from Yahoo Finance (^NSEI). Index data has no volume, so the volume filter is
  off and VWAP is a session average price (same fallback as the Pine script). Yahoo can be
  delayed or have gaps - check the printed "latest bar" time against your broker/chart.
- Option premiums are NOT fetched or invented. Messages give the strike and SPOT levels.
- Educational tool, not investment advice.

Setup:  pip install yfinance pandas numpy
Run:    python nifty_notifier.py --test      (sends a test email)
        python nifty_notifier.py --once      (one check, used by GitHub Actions)
        python nifty_notifier.py             (loops during market hours)
        python nifty_notifier.py --selftest  (synthetic data, no network, no email)
"""
import argparse, json, os, smtplib, sys, time
from datetime import datetime, timedelta
from email.message import EmailMessage

import numpy as np
import pandas as pd

# ───────────────────────────── CONFIG (edit here or use env vars) ─────────────────────────────
CFG = dict(
    symbol="^NSEI",
    strike_interval=50,
    strike_mode="ATM",          # ATM, ITM 1, ITM 2, OTM 1
    min_score=75, strong_score=85, adx_min=20.0,
    sl_atr_mult=1.2, rr=1.5, min_risk_pts=10.0,
    max_trades=3, max_loss_pts=75.0,
    no_trade_open_min=15, no_entry_close_min=30, square_off_min=15, max_hold=0,
    exit_on_opposite=True,
    spike_mult=2.5, ext_mult=2.0, brk_look=3, pb_tol=0.25,
    piv_left=5, piv_right=5, orb_min=15, chop_crosses=3,
    cd_same=12, cd_opp=6,
    # v2 additions
    use_be=True, be_trig_r=1.0, be_lock_r=0.1,   # breakeven shift
    max_cons_loss=2,                              # stop for the day after N consecutive losses (0 = off)
    bar_minutes=5,
)
EMAIL_FROM = os.getenv("EMAIL_FROM", "")               # e.g. your Gmail address
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")  # Gmail "App password", not your login password
EMAIL_TO = os.getenv("EMAIL_TO", "")
SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notifier_state.json")
TZ = "Asia/Kolkata"
OPEN_MIN, CLOSE_MIN = 555, 930  # 09:15, 15:30


# ───────────────────────────── indicator helpers ─────────────────────────────
def rma(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(c, n=14):
    d = c.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    rs = rma(up, n) / rma(dn, n).replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

def dmi_adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(true_range(h, l, c), n)
    dip = 100 * rma(pd.Series(plus, index=h.index), n) / tr
    dim = 100 * rma(pd.Series(minus, index=h.index), n) / tr
    dx = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    return dip.fillna(0), dim.fillna(0), rma(dx.fillna(0), n)

def supertrend(h, l, c, factor=3.0, n=10):
    """Returns direction: -1 bullish, +1 bearish (same convention as Pine ta.supertrend)."""
    atr = rma(true_range(h, l, c), n).values
    hl2 = ((h + l) / 2).values
    cv = c.values
    ub, lb = hl2 + factor * atr, hl2 - factor * atr
    fub, flb = ub.copy(), lb.copy()
    d = np.ones(len(cv))
    for i in range(1, len(cv)):
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or cv[i - 1] < flb[i - 1]) else flb[i - 1]
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or cv[i - 1] > fub[i - 1]) else fub[i - 1]
        if d[i - 1] == 1 and cv[i] > fub[i - 1]:
            d[i] = -1
        elif d[i - 1] == -1 and cv[i] < flb[i - 1]:
            d[i] = 1
        else:
            d[i] = d[i - 1]
    return pd.Series(d, index=c.index)


# ───────────────────────────── data ─────────────────────────────
def fetch_5m(retries=3):
    import yfinance as yf
    df = None
    for attempt in range(retries):
        try:
            df = yf.download(CFG["symbol"], period="10d", interval="5m", progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                break
        except Exception as ex:
            print(f"[warn] yahoo attempt {attempt + 1}/{retries} failed: {ex}")
        time.sleep(5 * (attempt + 1))
    if df is None or df.empty:
        raise RuntimeError("No data returned from Yahoo Finance")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    idx = df.index
    df.index = idx.tz_localize("UTC").tz_convert(TZ) if idx.tz is None else idx.tz_convert(TZ)
    # drop the still-forming candle
    now = datetime.now(df.index.tz)
    if len(df) and df.index[-1] + timedelta(minutes=CFG["bar_minutes"]) > now:
        df = df.iloc[:-1]
    return df


# ───────────────────────────── engine ─────────────────────────────
def build(df):
    c = CFG
    d = df.copy()
    O, H, L, C, V = d["open"], d["high"], d["low"], d["close"], d["volume"]
    d["date"] = d.index.date
    d["mod"] = d.index.hour * 60 + d.index.minute
    d["e9"], d["e21"], d["e50"] = ema(C, 9), ema(C, 21), ema(C, 50)
    d["rsi"] = rsi(C)
    macd = ema(C, 12) - ema(C, 26)
    sig = ema(macd, 9)
    d["macd"], d["macds"], d["macdh"] = macd, sig, macd - sig
    d["st"] = supertrend(H, L, C)
    d["dip"], d["dim"], d["adx"] = dmi_adx(H, L, C)
    d["atr"] = rma(true_range(H, L, C), 14)
    mid = C.rolling(20).mean()
    sd = C.rolling(20).std(ddof=0)
    d["bbw"] = (4 * sd / mid)
    d["bbwavg"] = d["bbw"].rolling(20).mean()
    volsma = V.rolling(20).mean()
    d["hasvol"] = (volsma.fillna(0) > 0)
    hlc3 = (H + L + C) / 3
    g = d["date"]
    pv = (hlc3 * V).groupby(g).cumsum()
    vv = V.groupby(g).cumsum()
    avg = hlc3.groupby(g).expanding().mean().reset_index(level=0, drop=True)
    d["vwap"] = np.where(vv > 0, pv / vv.replace(0, np.nan), avg)
    d["volsma"] = volsma

    # higher timeframe: previous COMPLETED 15m bar (matches Pine [1] + lookahead_off)
    h15 = C.resample("15min", origin="start_day", offset="9h15min", label="left", closed="left").last().dropna()
    h15df = pd.DataFrame({"c": h15, "e21": ema(h15, 21), "e50": ema(h15, 50)}).shift(1)
    h15df = h15df.reindex(d.index, method="ffill")
    d["htfc"], d["htfe21"], d["htfe50"] = h15df["c"], h15df["e21"], h15df["e50"]

    # previous day levels
    daily = d.groupby("date").agg(h=("high", "max"), l=("low", "min"), c=("close", "last"))
    prev = daily.shift(1)
    d["pdh"], d["pdl"], d["pdc"] = d["date"].map(prev["h"]), d["date"].map(prev["l"]), d["date"].map(prev["c"])
    p = (d["pdh"] + d["pdl"] + d["pdc"]) / 3
    d["r1"], d["s1"] = 2 * p - d["pdl"], 2 * p - d["pdh"]

    # confirmed swing highs/lows (adopted only after right-hand bars)
    pl, pr = c["piv_left"], c["piv_right"]
    sh = H.rolling(pl + pr + 1).max().shift(0)
    swh = np.where(H.shift(pr) == sh, H.shift(pr), np.nan)
    sl = L.rolling(pl + pr + 1).min()
    swl = np.where(L.shift(pr) == sl, L.shift(pr), np.nan)
    d["swh"] = pd.Series(swh, index=d.index).ffill()
    d["swl"] = pd.Series(swl, index=d.index).ffill()
    return d


def score_and_signals(d):
    c = CFG
    n = len(d)
    A = {k: d[k].values for k in ["open", "high", "low", "close", "e9", "e21", "e50", "rsi", "macd", "macds", "macdh",
                                  "st", "dip", "dim", "adx", "atr", "vwap", "htfc", "htfe21", "htfe50", "pdh", "pdl",
                                  "r1", "s1", "swh", "swl", "bbw", "bbwavg", "volume", "volsma", "mod"]}
    use_vol = bool(d["hasvol"].iloc[-1])
    date = d["date"].values

    # opening range per day
    orh, orl = np.full(n, np.nan), np.full(n, np.nan)
    cur_date, h_, l_ = None, np.nan, np.nan
    for i in range(n):
        if date[i] != cur_date:
            cur_date, h_, l_ = date[i], np.nan, np.nan
        m = A["mod"][i]
        if OPEN_MIN <= m < OPEN_MIN + c["orb_min"]:
            h_ = A["high"][i] if np.isnan(h_) else max(h_, A["high"][i])
            l_ = A["low"][i] if np.isnan(l_) else min(l_, A["low"][i])
        if m >= OPEN_MIN + c["orb_min"]:
            orh[i], orl[i] = h_, l_

    cl, op, hi, lo, atr, vw = A["close"], A["open"], A["high"], A["low"], A["atr"], A["vwap"]
    rng = hi - lo
    cpos = np.where(rng > 0, (cl - lo) / np.where(rng > 0, rng, 1), 0.5)
    spike = rng > atr * c["spike_mult"]
    above = cl > vw
    crossf = np.zeros(n)
    crossf[1:] = (above[1:] != above[:-1]).astype(float)
    chop = (pd.Series(crossf).rolling(10).sum().fillna(0).values >= c["chop_crosses"])
    sideways = (A["adx"] < c["adx_min"]) | ((A["adx"] < c["adx_min"] + 5) & (A["bbw"] < A["bbwavg"]))
    vol_ok = (A["volume"] >= A["volsma"]) if use_vol else np.ones(n, bool)

    def xup(a, b):
        r = np.zeros(n, bool)
        r[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1]) & ~np.isnan(b[1:]) & ~np.isnan(b[:-1])
        return r

    def xdn(a, b):
        r = np.zeros(n, bool)
        r[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1]) & ~np.isnan(b[1:]) & ~np.isnan(b[:-1])
        return r

    orb_ready = ~np.isnan(orh)
    brk_bull = xup(cl, A["swh"]) | xup(cl, A["pdh"]) | (orb_ready & xup(cl, orh))
    brk_bear = xdn(cl, A["swl"]) | xdn(cl, A["pdl"]) | (orb_ready & xdn(cl, orl))
    bull_q = (cl > op) & (cpos >= 0.6) & ~spike & ((cl - A["e21"]) <= c["ext_mult"] * atr)
    bear_q = (cl < op) & (cpos <= 0.4) & ~spike & ((A["e21"] - cl) <= c["ext_mult"] * atr)
    bbq, brq = brk_bull & bull_q, brk_bear & bear_q

    def recent(flag):
        out = np.zeros(n, bool)
        last = -10**6
        for i in range(n):
            if flag[i]:
                last = i
            out[i] = (i - last) <= c["brk_look"]
        return out
    brk_bull_r, brk_bear_r = recent(bbq), recent(brq)
    pb_bull = (A["e9"] > A["e21"]) & (A["e21"] > A["e50"]) & (lo <= A["e21"] + c["pb_tol"] * atr) & (cl > A["e9"]) & bull_q
    pb_bear = (A["e9"] < A["e21"]) & (A["e21"] < A["e50"]) & (hi >= A["e21"] - c["pb_tol"] * atr) & (cl < A["e9"]) & bear_q
    setup_bull, setup_bear = brk_bull_r | pb_bull, brk_bear_r | pb_bear

    def room(up):
        out = np.ones(n, bool)
        for i in range(n):
            if up:
                lv = [x for x in (A["swh"][i], A["pdh"][i], orh[i], A["r1"][i]) if not np.isnan(x) and x > cl[i]]
                out[i] = (not lv) or (min(lv) - cl[i]) >= atr[i]
            else:
                lv = [x for x in (A["swl"][i], A["pdl"][i], orl[i], A["s1"][i]) if not np.isnan(x) and x < cl[i]]
                out[i] = (not lv) or (cl[i] - max(lv)) >= atr[i]
        return out
    room_bull, room_bear = room(True), room(False)

    rsi_, mh = A["rsi"], A["macdh"]
    mh_prev = np.concatenate([[np.nan], mh[:-1]])
    adx, dip, dim = A["adx"], A["dip"], A["dim"]
    hc, h21, h50 = A["htfc"], A["htfe21"], A["htfe50"]
    sb = np.zeros(n); se = np.zeros(n)
    sb += np.where(A["st"] < 0, 10, 0); se += np.where(A["st"] > 0, 10, 0)
    sb += np.where(cl > vw, 10, 0);     se += np.where(cl < vw, 10, 0)
    sb += np.where((A["e9"] > A["e21"]) & (A["e21"] > A["e50"]), 15, np.where(A["e9"] > A["e21"], 8, 0))
    se += np.where((A["e9"] < A["e21"]) & (A["e21"] < A["e50"]), 15, np.where(A["e9"] < A["e21"], 8, 0))
    sb += np.where((rsi_ >= 55) & (rsi_ <= 75), 10, np.where((rsi_ > 50) & (rsi_ < 55), 5, np.where(rsi_ > 75, 3, 0)))
    se += np.where((rsi_ <= 45) & (rsi_ >= 25), 10, np.where((rsi_ < 50) & (rsi_ > 45), 5, np.where(rsi_ < 25, 3, 0)))
    sb += np.where((A["macd"] > A["macds"]) & (mh > mh_prev), 10, np.where(A["macd"] > A["macds"], 6, 0))
    se += np.where((A["macd"] < A["macds"]) & (mh < mh_prev), 10, np.where(A["macd"] < A["macds"], 6, 0))
    sb += np.where((adx >= 25) & (dip > dim), 10, np.where((adx >= c["adx_min"]) & (dip > dim), 6, 0))
    se += np.where((adx >= 25) & (dim > dip), 10, np.where((adx >= c["adx_min"]) & (dim > dip), 6, 0))
    if use_vol:
        sb += np.where(vol_ok, 5, 0); se += np.where(vol_ok, 5, 0)
    sb += np.where(brk_bull_r, 10, np.where(pb_bull, 7, 0)); se += np.where(brk_bear_r, 10, np.where(pb_bear, 7, 0))
    sb += np.where((hc > h21) & (h21 > h50), 15, np.where(hc > h21, 8, 0))
    se += np.where((hc < h21) & (h21 < h50), 15, np.where(hc < h21, 8, 0))
    sb += np.where(room_bull, 5, 0); se += np.where(room_bear, 5, 0)
    maxw = 100.0 if use_vol else 95.0
    score_b = np.minimum(100, np.round(sb / maxw * 100)).astype(int)
    score_e = np.minimum(100, np.round(se / maxw * 100)).astype(int)

    m = A["mod"]
    # v2: no new entries inside the square-off window either
    entry_cut = CLOSE_MIN - max(c["no_entry_close_min"], c["square_off_min"])
    trade_win = (m >= OPEN_MIN + c["no_trade_open_min"]) & (m < entry_cut)
    core_b = (cl > vw) & (A["e9"] > A["e21"]) & (A["st"] < 0) & (adx >= c["adx_min"]) & (dip > dim) & (hc > h21)
    core_e = (cl < vw) & (A["e9"] < A["e21"]) & (A["st"] > 0) & (adx >= c["adx_min"]) & (dim > dip) & (hc < h21)
    ok = ~sideways & ~chop & vol_ok & ~spike & ~np.isnan(hc) & ~np.isnan(A["pdh"])
    buy = trade_win & ok & core_b & setup_bull & (score_b >= c["min_score"]) & (score_b > score_e)
    sell = trade_win & ok & core_e & setup_bear & (score_e >= c["min_score"]) & (score_e > score_b)
    return buy, sell, score_b, score_e, A, date


def run_state(d):
    """Replays the Pine v2 trade state machine; returns a list of events with bar index."""
    c = CFG
    buy, sell, sb, se, A, date = score_and_signals(d)
    n = len(d)
    ts = d.index
    atm = lambda px: int(round(px / c["strike_interval"]) * c["strike_interval"])
    steps = {"ATM": 0, "ITM 1": 1, "ITM 2": 2, "OTM 1": -1}[c["strike_mode"]]
    sess_end = (A["mod"] + c["bar_minutes"]) >= CLOSE_MIN       # the session's last bar always flattens
    events = []
    pos = 0; entry = sl = tg = np.nan; ebar = 0; strike = 0; tscore = 0
    risk_pts = np.nan; be_done = False; cons_loss = 0
    trades = 0; day_pnl = 0.0; last_entry = -10**6; last_dir = 0; cur_date = None
    for i in range(n):
        if date[i] != cur_date:
            cur_date = date[i]; trades = 0; day_pnl = 0.0; last_entry = -10**6; last_dir = 0; pos = 0; cons_loss = 0
        flat_start = pos == 0
        exited = False

        # ---- exits (from the bar AFTER entry; the session's last bar always flattens) ----
        if pos != 0 and (i > ebar or sess_end[i]):
            same = i == ebar
            if pos == 1:
                hit_sl = (not same) and A["low"][i] <= sl
                hit_tg = (not same) and A["high"][i] >= tg
            else:
                hit_sl = (not same) and A["high"][i] >= sl
                hit_tg = (not same) and A["low"][i] <= tg
            opp = (not same) and ((pos == 1 and sell[i]) or (pos == -1 and buy[i]))
            tm = bool(sess_end[i]) or A["mod"][i] >= CLOSE_MIN - c["square_off_min"] or (c["max_hold"] > 0 and i - ebar >= c["max_hold"])
            kind = None
            if hit_sl:
                kind, px = ("BREAKEVEN STOP" if be_done else "STOP-LOSS HIT"), sl
            elif hit_tg:
                kind, px = "TARGET HIT", tg
            elif c["exit_on_opposite"] and opp:
                kind, px = "EXIT: opposite signal", A["close"][i]
            elif tm:
                kind, px = "EXIT: time / square-off", A["close"][i]
            if kind:
                pnl = (px - entry) * pos
                day_pnl += pnl
                cons_loss = cons_loss + 1 if pnl < 0 else 0
                events.append(dict(i=i, ts=ts[i], kind=kind, dir=pos, strike=strike, pnl=pnl, spot=A["close"][i]))
                pos = 0; exited = True

        # ---- breakeven shift: once +be_trig_r is reached on a closed bar, stop moves from the NEXT bar ----
        if c["use_be"] and pos != 0 and not be_done and i > ebar and not exited:
            if pos == 1:
                reached = A["high"][i] >= entry + risk_pts * c["be_trig_r"]
            else:
                reached = A["low"][i] <= entry - risk_pts * c["be_trig_r"]
            if reached:
                sl = entry + pos * risk_pts * c["be_lock_r"]
                be_done = True
                events.append(dict(i=i, ts=ts[i], kind="MOVE STOP TO BREAKEVEN", dir=pos, strike=strike,
                                   entry=entry, sl=sl, tg=tg, spot=A["close"][i]))

        # ---- entries ----
        can = (flat_start and not exited and trades < c["max_trades"] and day_pnl > -c["max_loss_pts"]
               and (c["max_cons_loss"] == 0 or cons_loss < c["max_cons_loss"]))
        cd_b = (i - last_entry) >= (c["cd_same"] if last_dir == 1 else c["cd_opp"])
        cd_e = (i - last_entry) >= (c["cd_same"] if last_dir == -1 else c["cd_opp"])
        enter = 0
        if can and buy[i] and cd_b: enter = 1
        elif can and sell[i] and cd_e: enter = -1
        if enter:
            risk = max(A["atr"][i] * c["sl_atr_mult"], c["min_risk_pts"])
            pos = enter; entry = A["close"][i]; ebar = i
            risk_pts = risk; be_done = False
            sl = entry - risk * enter; tg = entry + risk * c["rr"] * enter
            strike = atm(entry) - steps * c["strike_interval"] if enter == 1 else atm(entry) + steps * c["strike_interval"]
            tscore = int(sb[i] if enter == 1 else se[i])
            trades += 1; last_entry = i; last_dir = enter
            events.append(dict(i=i, ts=ts[i], kind="ENTRY", dir=enter, strike=strike, entry=entry, tg=tg, sl=sl, score=tscore,
                               spot=entry))
    return events


# ───────────────────────────── notifications ─────────────────────────────
def grade(s):
    return "VERY STRONG" if s >= 85 else "STRONG" if s >= 75 else "MODERATE" if s >= 65 else "NO TRADE"

def format_event(e):
    t = e["ts"].strftime("%d-%b %H:%M")
    side = "CE" if e["dir"] == 1 else "PE"
    if e["kind"] == "ENTRY":
        return (f"BUY NIFTY {e['strike']} {side}\n"
                f"Spot entry: {e['entry']:.1f}\nTarget (spot): {e['tg']:.1f}\nStop-loss (spot): {e['sl']:.1f}\n"
                f"Score: {e['score']}/100 ({grade(e['score'])})\nCandle close: {t} IST\n"
                f"Option premium: check your broker (not auto-fetched).\nScore is a confluence rating, not a win probability.")
    if e["kind"] == "MOVE STOP TO BREAKEVEN":
        return (f"MOVE STOP TO BREAKEVEN\nNIFTY {e['strike']} {side}\n"
                f"New spot stop-loss: {e['sl']:.1f} (entry {e['entry']:.1f})\nTarget unchanged: {e['tg']:.1f}\n"
                f"Spot now: {e['spot']:.1f}\nCandle close: {t} IST\nMove your stop manually in your broker.")
    return f"{e['kind']}\nNIFTY {e['strike']} {side}\nSpot: {e['spot']:.1f}\nP&L: {e['pnl']:+.1f} spot pts\nCandle close: {t} IST"


def format_html(e):
    """Email-safe HTML (tables + inline styles, works in Gmail/Outlook/mobile)."""
    t = e["ts"].strftime("%a %d %b, %H:%M")
    side = "CE" if e["dir"] == 1 else "PE"
    word = "CALL" if e["dir"] == 1 else "PUT"
    FONT = "font-family:-apple-system,'Segoe UI',Roboto,Arial,sans-serif;"
    if e["kind"] == "ENTRY":
        col = "#0f9d58" if e["dir"] == 1 else "#d93025"
        risk, reward = abs(e["entry"] - e["sl"]), abs(e["tg"] - e["entry"])
        rr = reward / risk if risk else 0
        sc = e["score"]
        bar = "#0f9d58" if sc >= 85 else "#f9ab00" if sc >= 75 else "#80868b"
        def row(label, val, color="#202124", bold=True):
            return (f'<tr><td style="padding:12px 16px;border-bottom:1px solid #eee;color:#5f6368;font-size:14px;{FONT}">{label}</td>'
                    f'<td align="right" style="padding:12px 16px;border-bottom:1px solid #eee;font-size:17px;color:{color};'
                    f'font-weight:{700 if bold else 400};{FONT}">{val}</td></tr>')
        body = (row("Entry (spot)", f"{e['entry']:,.1f}") +
                row("Target (spot)", f"{e['tg']:,.1f}", "#0f9d58") +
                row("Stop-loss (spot)", f"{e['sl']:,.1f}", "#d93025") +
                row("Risk : Reward", f"1 : {rr:.2f}") +
                row("Risk / Reward (points)", f"{risk:.1f} / {reward:.1f}", "#202124", False))
        return f"""<!doctype html><html><body style="margin:0;padding:0;background:#f1f3f4;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f3f4;padding:24px 8px;"><tr><td align="center">
<table width="480" cellpadding="0" cellspacing="0" style="max-width:480px;width:100%;background:#ffffff;border-radius:14px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.08);">
<tr><td style="background:{col};padding:22px 24px;{FONT}">
  <div style="color:#f1f3f4;font-size:12px;letter-spacing:1.5px;font-weight:600;">NEW SIGNAL &middot; BUY {word}</div>
  <div style="color:#ffffff;font-size:28px;font-weight:800;margin-top:6px;">NIFTY {e['strike']} {side}</div>
  <div style="color:#ffffff;font-size:13px;margin-top:6px;">Candle close {t} IST</div>
</td></tr>
<tr><td style="padding:18px 24px 6px 24px;{FONT}">
  <div style="font-size:12px;color:#5f6368;letter-spacing:1px;font-weight:600;">CONFLUENCE SCORE</div>
  <table width="100%" cellpadding="0" cellspacing="0" style="margin-top:8px;"><tr>
    <td style="font-size:26px;font-weight:800;color:#202124;{FONT}" width="110">{sc}<span style="font-size:14px;color:#5f6368;font-weight:600;">/100</span></td>
    <td><div style="background:#e8eaed;border-radius:6px;height:10px;"><div style="background:{bar};width:{sc}%;height:10px;border-radius:6px;"></div></div></td>
    <td align="right" width="110" style="font-size:12px;font-weight:700;color:{bar};{FONT}">{grade(sc)}</td>
  </tr></table>
</td></tr>
<tr><td style="padding:12px 24px 0 24px;"><table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #eee;border-radius:10px;overflow:hidden;background:#ffffff;">{body}</table></td></tr>
<tr><td style="padding:16px 24px 6px 24px;{FONT}font-size:13px;color:#3c4043;line-height:1.5;">
  <b>Check before you trade:</b> look up the live <b>{e['strike']} {side}</b> premium in your broker. Levels above are NIFTY <b>spot</b>, not option prices.
</td></tr>
<tr><td style="padding:14px 24px 22px 24px;{FONT}font-size:11px;color:#80868b;line-height:1.5;border-top:1px solid #f1f3f4;">
  The score is a confluence rating, not a probability of winning. Educational tool, not investment advice. Options can lose their full premium quickly.
</td></tr></table></td></tr></table></body></html>"""
    if e["kind"] == "MOVE STOP TO BREAKEVEN":
        return f"""<!doctype html><html><body style="margin:0;padding:0;background:#f1f3f4;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f3f4;padding:24px 8px;"><tr><td align="center">
<table width="480" cellpadding="0" cellspacing="0" style="max-width:480px;width:100%;background:#ffffff;border-radius:14px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.08);">
<tr><td style="background:#1a73e8;padding:22px 24px;{FONT}">
  <div style="color:#f1f3f4;font-size:12px;letter-spacing:1.5px;font-weight:600;">TRADE UPDATE &middot; ACTION NEEDED</div>
  <div style="color:#ffffff;font-size:24px;font-weight:800;margin-top:6px;">Move stop to breakeven</div>
  <div style="color:#ffffff;font-size:13px;margin-top:6px;">NIFTY {e['strike']} {side} &middot; {t} IST</div>
</td></tr>
<tr><td style="padding:20px 24px;{FONT}">
  <table width="100%" cellpadding="0" cellspacing="0">
    <tr><td style="color:#5f6368;font-size:14px;padding:8px 0;">New stop-loss (spot)</td><td align="right" style="font-size:20px;font-weight:800;color:#d93025;">{e['sl']:,.1f}</td></tr>
    <tr><td style="color:#5f6368;font-size:14px;padding:8px 0;border-top:1px solid #eee;">Entry (spot)</td><td align="right" style="font-size:16px;font-weight:700;color:#202124;border-top:1px solid #eee;">{e['entry']:,.1f}</td></tr>
    <tr><td style="color:#5f6368;font-size:14px;padding:8px 0;border-top:1px solid #eee;">Target (spot), unchanged</td><td align="right" style="font-size:16px;font-weight:700;color:#0f9d58;border-top:1px solid #eee;">{e['tg']:,.1f}</td></tr>
    <tr><td style="color:#5f6368;font-size:14px;padding:8px 0;border-top:1px solid #eee;">Spot now</td><td align="right" style="font-size:16px;font-weight:700;color:#202124;border-top:1px solid #eee;">{e['spot']:,.1f}</td></tr>
  </table>
</td></tr>
<tr><td style="padding:14px 24px 22px 24px;{FONT}font-size:11px;color:#80868b;line-height:1.5;border-top:1px solid #f1f3f4;">
  Move the stop manually in your broker. Levels are NIFTY spot, not option prices. Educational tool, not investment advice.
</td></tr></table></td></tr></table></body></html>"""
    # exits
    k = e["kind"]
    col = "#0f9d58" if k == "TARGET HIT" else "#d93025" if k == "STOP-LOSS HIT" else "#f9ab00"
    pcol = "#0f9d58" if e["pnl"] >= 0 else "#d93025"
    return f"""<!doctype html><html><body style="margin:0;padding:0;background:#f1f3f4;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f3f4;padding:24px 8px;"><tr><td align="center">
<table width="480" cellpadding="0" cellspacing="0" style="max-width:480px;width:100%;background:#ffffff;border-radius:14px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.08);">
<tr><td style="background:{col};padding:22px 24px;{FONT}">
  <div style="color:#f1f3f4;font-size:12px;letter-spacing:1.5px;font-weight:600;">TRADE UPDATE</div>
  <div style="color:#ffffff;font-size:26px;font-weight:800;margin-top:6px;">{k}</div>
  <div style="color:#ffffff;font-size:13px;margin-top:6px;">NIFTY {e['strike']} {side} &middot; {t} IST</div>
</td></tr>
<tr><td style="padding:20px 24px;{FONT}">
  <table width="100%" cellpadding="0" cellspacing="0">
    <tr><td style="color:#5f6368;font-size:14px;padding:8px 0;">Spot at exit</td><td align="right" style="font-size:17px;font-weight:700;color:#202124;">{e['spot']:,.1f}</td></tr>
    <tr><td style="color:#5f6368;font-size:14px;padding:8px 0;border-top:1px solid #eee;">Result (spot points)</td><td align="right" style="font-size:20px;font-weight:800;color:{pcol};border-top:1px solid #eee;">{e['pnl']:+.1f}</td></tr>
  </table>
</td></tr>
<tr><td style="padding:14px 24px 22px 24px;{FONT}font-size:11px;color:#80868b;line-height:1.5;border-top:1px solid #f1f3f4;">
  Result is in NIFTY spot points and excludes costs and option-premium effects. Educational tool, not investment advice.
</td></tr></table></td></tr></table></body></html>"""

def send_email(subject, msg, html=None):
    if not (EMAIL_FROM and EMAIL_APP_PASSWORD and EMAIL_TO):
        return False
    m = EmailMessage()
    m["From"], m["To"], m["Subject"] = EMAIL_FROM, EMAIL_TO, subject
    m.set_content(msg)
    if html:
        m.add_alternative(html, subtype="html")
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as s:
        s.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        s.send_message(m)
    return True

def notify(subject, msg, html=None):
    ok = False
    try:
        ok = send_email(subject, msg, html)
    except Exception as ex:
        print(f"[warn] email failed: {ex}")
    print(f"[{pd.Timestamp.now(tz=TZ):%H:%M:%S} IST] {subject} -> {'email sent' if ok else 'NOT SENT (set EMAIL_FROM, EMAIL_APP_PASSWORD, EMAIL_TO)'}")
    print(msg, "\n")


# ───────────────────────────── main loop ─────────────────────────────
def load_state():
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {}

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"))

def check_once(df=None):
    df = fetch_5m() if df is None else df
    if len(df) < 120:
        print("Not enough bars yet:", len(df)); return
    d = build(df)
    last_ts = d.index[-1]
    print(f"[{pd.Timestamp.now(tz=TZ):%H:%M:%S} IST] latest closed bar: {last_ts:%d-%b %H:%M} close {d['close'].iloc[-1]:.1f}")
    events = run_state(d)
    st = load_state()
    sent = st.get("sent", [])
    now = datetime.now(d.index.tz)
    for e in events:
        # catch events from the last 3 bars (a delayed cloud run may skip a candle),
        # but never anything older than 20 minutes, so a lost state file cannot cause old alerts
        if e["i"] < len(d) - 3 or (now - e["ts"]) > timedelta(minutes=20):
            continue
        key = f"{e['ts'].isoformat()}|{e['kind']}"
        if key in sent:
            continue
        subj = ("BUY NIFTY %d %s" % (e["strike"], "CE" if e["dir"] == 1 else "PE")) if e["kind"] == "ENTRY" else e["kind"]
        notify(subj, format_event(e), format_html(e))
        sent.append(key)
        st["sent"] = sent[-50:]
        save_state(st)

def market_open_now():
    now = datetime.now(pd.Timestamp.now(tz=TZ).tz)
    return now.weekday() < 5 and OPEN_MIN <= now.hour * 60 + now.minute <= CLOSE_MIN + 10

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="send a test notification and exit")
    ap.add_argument("--once", action="store_true", help="check once and exit")
    ap.add_argument("--selftest", action="store_true", help="run engine on synthetic data (no network, no sending)")
    a = ap.parse_args()
    if a.test:
        notify("NIFTY notifier test", "Test message from nifty_notifier.py. If you can read this, notifications work.")
        return
    if a.selftest:
        selftest(); return
    if a.once:
        if not market_open_now():
            print(f"[{pd.Timestamp.now(tz=TZ):%a %d-%b %H:%M:%S} IST] Market closed (Mon-Fri 09:15-15:40 IST) - nothing to do."); return
        check_once(); return
    print("Running. Ctrl+C to stop. Checks every 5-minute candle close during market hours (Mon-Fri 09:15-15:30 IST).")
    while True:
        try:
            if market_open_now():
                check_once()
        except Exception as ex:
            print("[error]", ex)
        now = datetime.now()
        wait = 300 - (now.minute % 5) * 60 - now.second + 8   # a few seconds after each 5m close
        time.sleep(max(15, wait))

def selftest():
    rng = np.random.default_rng(7)
    idx = []
    for day in pd.bdate_range("2026-11-02", periods=8, tz=TZ):
        idx += list(pd.date_range(day + pd.Timedelta(hours=9, minutes=15), day + pd.Timedelta(hours=15, minutes=25), freq="5min"))
    idx = pd.DatetimeIndex(idx)
    steps = rng.normal(0.4, 9, len(idx))
    close = 24500 + np.cumsum(steps)
    openp = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(openp, close) + rng.uniform(0, 6, len(idx))
    low = np.minimum(openp, close) - rng.uniform(0, 6, len(idx))
    df = pd.DataFrame(dict(open=openp, high=high, low=low, close=close, volume=0.0), index=idx)
    d = build(df)
    ev = run_state(d)
    print(f"selftest OK: {len(df)} bars, {len(ev)} events")
    for e in ev[:6]:
        print(format_event(e), "\n---")

if __name__ == "__main__":
    main()
