#!/usr/bin/env python3
"""
NIFTY intraday signal notifier (free alternative to TradingView alerts).

A Python port of nifty_intraday_signals.pine. It polls 5-minute NIFTY data, evaluates the
same gates / setups / weighted score on CLOSED candles only, and sends Telegram and/or email
when a BUY CE / BUY PE signal or an exit (target, stop-loss, opposite, time) occurs.

Notes
- It is a port, so small differences from the TradingView indicator are possible.
- Data comes from Yahoo Finance (^NSEI). Index data has no volume, so the volume filter is
  off and VWAP is a session average price (same fallback as the Pine script). Yahoo can be
  delayed or have gaps - check the printed "latest bar" time against your broker/chart.
- Option premiums are NOT fetched or invented. Messages give the strike and SPOT levels.
- Educational tool, not investment advice.

Setup:  pip install yfinance pandas numpy requests
Run:    python nifty_notifier.py --test      (sends a test message)
        python nifty_notifier.py             (runs during market hours)
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
)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")       # from @BotFather
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")   # your chat id
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
def fetch_5m():
    import yfinance as yf
    df = yf.download(CFG["symbol"], period="10d", interval="5m", progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("No data returned from Yahoo Finance")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    idx = df.index
    df.index = idx.tz_localize("UTC").tz_convert(TZ) if idx.tz is None else idx.tz_convert(TZ)
    # drop the still-forming candle
    now = datetime.now(df.index.tz)
    if len(df) and df.index[-1] + timedelta(minutes=5) > now:
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
    trade_win = (m >= OPEN_MIN + c["no_trade_open_min"]) & (m < CLOSE_MIN - c["no_entry_close_min"])
    core_b = (cl > vw) & (A["e9"] > A["e21"]) & (A["st"] < 0) & (adx >= c["adx_min"]) & (dip > dim) & (hc > h21)
    core_e = (cl < vw) & (A["e9"] < A["e21"]) & (A["st"] > 0) & (adx >= c["adx_min"]) & (dim > dip) & (hc < h21)
    ok = ~sideways & ~chop & vol_ok & ~spike & ~np.isnan(hc) & ~np.isnan(A["pdh"])
    buy = trade_win & ok & core_b & setup_bull & (score_b >= c["min_score"]) & (score_b > score_e)
    sell = trade_win & ok & core_e & setup_bear & (score_e >= c["min_score"]) & (score_e > score_b)
    return buy, sell, score_b, score_e, A, date


def run_state(d):
    """Replays the Pine trade state machine; returns a list of events with bar index."""
    c = CFG
    buy, sell, sb, se, A, date = score_and_signals(d)
    n = len(d)
    ts = d.index
    atm = lambda px: int(round(px / c["strike_interval"]) * c["strike_interval"])
    steps = {"ATM": 0, "ITM 1": 1, "ITM 2": 2, "OTM 1": -1}[c["strike_mode"]]
    events = []
    pos = 0; entry = sl = tg = np.nan; ebar = 0; strike = 0; tscore = 0
    trades = 0; day_pnl = 0.0; last_entry = -10**6; last_dir = 0; cur_date = None
    for i in range(n):
        if date[i] != cur_date:
            cur_date = date[i]; trades = 0; day_pnl = 0.0; last_entry = -10**6; last_dir = 0; pos = 0
        flat_start = pos == 0
        exited = False
        if pos != 0 and i > ebar:
            hit_sl = A["low"][i] <= sl if pos == 1 else A["high"][i] >= sl
            hit_tg = A["high"][i] >= tg if pos == 1 else A["low"][i] <= tg
            opp = (pos == 1 and sell[i]) or (pos == -1 and buy[i])
            tm = A["mod"][i] >= CLOSE_MIN - c["square_off_min"] or (c["max_hold"] > 0 and i - ebar >= c["max_hold"])
            kind = None
            if hit_sl: kind, px = "STOP-LOSS HIT", sl
            elif hit_tg: kind, px = "TARGET HIT", tg
            elif c["exit_on_opposite"] and opp: kind, px = "EXIT: opposite signal", A["close"][i]
            elif tm: kind, px = "EXIT: time / square-off", A["close"][i]
            if kind:
                pnl = (px - entry) * pos
                day_pnl += pnl
                events.append(dict(i=i, ts=ts[i], kind=kind, dir=pos, strike=strike, pnl=pnl, spot=A["close"][i]))
                pos = 0; exited = True
        can = flat_start and not exited and trades < c["max_trades"] and day_pnl > -c["max_loss_pts"]
        cd_b = (i - last_entry) >= (c["cd_same"] if last_dir == 1 else c["cd_opp"])
        cd_e = (i - last_entry) >= (c["cd_same"] if last_dir == -1 else c["cd_opp"])
        enter = 0
        if can and buy[i] and cd_b: enter = 1
        elif can and sell[i] and cd_e: enter = -1
        if enter:
            risk = max(A["atr"][i] * c["sl_atr_mult"], c["min_risk_pts"])
            pos = enter; entry = A["close"][i]; ebar = i
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
    return f"{e['kind']}\nNIFTY {e['strike']} {side}\nSpot: {e['spot']:.1f}\nP&L: {e['pnl']:+.1f} spot pts\nCandle close: {t} IST"

def send_telegram(msg):
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        return False
    import requests
    r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=20)
    r.raise_for_status()
    return True

def send_email(subject, msg):
    if not (EMAIL_FROM and EMAIL_APP_PASSWORD and EMAIL_TO):
        return False
    m = EmailMessage()
    m["From"], m["To"], m["Subject"] = EMAIL_FROM, EMAIL_TO, subject
    m.set_content(msg)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as s:
        s.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        s.send_message(m)
    return True

def notify(subject, msg):
    sent = []
    for name, fn in (("telegram", lambda: send_telegram(msg)), ("email", lambda: send_email(subject, msg))):
        try:
            if fn(): sent.append(name)
        except Exception as ex:
            print(f"[warn] {name} failed: {ex}")
    print(f"[{datetime.now():%H:%M:%S}] {subject} -> sent via: {', '.join(sent) or 'NOTHING (configure Telegram/email)'}")
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
    print(f"[{datetime.now():%H:%M:%S}] latest closed bar: {last_ts:%d-%b %H:%M} close {d['close'].iloc[-1]:.1f}")
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
        notify(subj, format_event(e))
        sent.append(key)
        st["sent"] = sent[-50:]
        save_state(st)

def market_open_now():
    now = datetime.now(pd.Timestamp.now(tz=TZ).tz)
    return now.weekday() < 5 and OPEN_MIN <= now.hour * 60 + now.minute <= CLOSE_MIN + 1

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
            print("Market closed (Mon-Fri 09:15-15:30 IST) - nothing to do."); return
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
