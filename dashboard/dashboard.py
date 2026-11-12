"""
Streamlit Dashboard — TRIAD Multi-Modal Threat Monitor

Tabs:
1. Live Monitor  — verdict, threat gauge, source cards, MITRE ATT&CK stages,
                   threat timeline, Blind Spot Replay, SHAP explanation,
                   baseline-vs-now, live events, incident report download
2. What-If Lab   — drive the trained model with sliders and presets
3. Research      — 5-way experiment results, confusion matrices,
                   feature importance, CICIDS2017 baseline

Run:  streamlit run dashboard/dashboard.py   (Flask app must be running)
"""

import os
import sys
import json
import math
import html
from collections import deque
from datetime import datetime

import requests
import pandas as pd
import altair as alt
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from app.feature_extractor import (
    FEATURE_COLUMNS, AUTH_FEATURES, NETWORK_FEATURES, SYSTEM_FEATURES, INTERNAL_PREFIXES, LOG_DIR
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'multimodal_results')
CICIDS_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'cicids_results')
FLASK_URL = 'http://127.0.0.1:5000'

HISTORY_POINTS = 120          # seconds of timeline
EPISODE_COOLDOWN = 20         # seconds of BENIGN before an episode closes

SAFE, WARN, DANGER = '#2dd4bf', '#fbbf24', '#ff3b5c'
SOURCE_COLOR = {'auth': '#a78bfa', 'network': '#38bdf8', 'system': '#fb923c'}
SOURCE_NAME = {'auth': 'Authentication', 'network': 'Network', 'system': 'System'}
MODALITY_OF = {**{f: 'auth' for f in AUTH_FEATURES},
               **{f: 'network' for f in NETWORK_FEATURES},
               **{f: 'system' for f in SYSTEM_FEATURES}}
VERDICT = {'BENIGN': ('SECURE', SAFE), 'SUSPICIOUS': ('SUSPICIOUS', WARN), 'ATTACK': ('ATTACK', DANGER)}

FEATURE_LABEL = {
    'failed_login_count': 'Failed logins', 'successful_login': 'Successful login',
    'failure_rate': 'Login failure rate', 'time_between_attempts': 'Time between attempts',
    'login_frequency': 'Login attempts / min', 'request_count': 'Request count',
    'requests_per_second': 'Requests / s', 'unique_endpoints': 'Unique endpoints',
    'bytes_sent': 'Bytes sent', 'bytes_received': 'Bytes received', 'error_count': 'HTTP errors',
    'cpu_usage': 'CPU usage', 'memory_usage': 'Memory usage', 'process_count': 'Process count',
    'new_process_count': 'New processes',
}

# MITRE ATT&CK techniques the simulated attacks map to
STAGES = [
    ('brute', 'T1110', 'Brute Force', 'auth', 'Repeated password guessing'),
    ('valid', 'T1078', 'Valid Accounts', 'auth', 'Login succeeds after guessing'),
    ('scan', 'T1595.003', 'Wordlist Scanning', 'network', 'Rapid endpoint probing'),
    ('hijack', 'T1496', 'Resource Hijacking', 'system', 'CPU spike + new processes'),
]

# Normal-user ceilings (standard benign profile of the training data)
BASELINES = [
    ('failed_login_count', 'Failed logins', '', 1, '{:.0f}'),
    ('login_frequency', 'Login attempts', '/min', 1.2, '{:.1f}'),
    ('requests_per_second', 'Request rate', 'req/s', 0.4, '{:.2f}'),
    ('unique_endpoints', 'Endpoints hit', '', 3, '{:.0f}'),
    ('cpu_usage', 'Host CPU (peak)', '%', 32, '{:.0f}'),
    ('new_process_count', 'New processes', '', 0, '{:.0f}'),
]

# What-If presets (values inside the ranges of the training profiles)
PRESETS = {
    'Normal user': dict(failed_login_count=0, successful_login=1, failure_rate=0.0, time_between_attempts=30.0,
                        login_frequency=0.6, request_count=9, requests_per_second=0.2, unique_endpoints=2,
                        bytes_sent=1900, bytes_received=4300, error_count=0, cpu_usage=22.0, memory_usage=42.0,
                        process_count=138, new_process_count=0),
    'Power user (benign)': dict(failed_login_count=0, successful_login=1, failure_rate=0.0, time_between_attempts=40.0,
                                login_frequency=0.4, request_count=48, requests_per_second=3.8, unique_endpoints=5,
                                bytes_sent=14700, bytes_received=29000, error_count=1, cpu_usage=27.0,
                                memory_usage=46.0, process_count=142, new_process_count=0),
    'Windows Update (benign)': dict(failed_login_count=0, successful_login=1, failure_rate=0.0,
                                    time_between_attempts=40.0, login_frequency=0.3, request_count=8,
                                    requests_per_second=0.2, unique_endpoints=2, bytes_sent=1700,
                                    bytes_received=3800, error_count=0, cpu_usage=76.0, memory_usage=77.0,
                                    process_count=185, new_process_count=2),
    'Brute force only': dict(failed_login_count=20, successful_login=0, failure_rate=1.0, time_between_attempts=0.6,
                             login_frequency=30.0, request_count=22, requests_per_second=3.0, unique_endpoints=1,
                             bytes_sent=4400, bytes_received=2500, error_count=0, cpu_usage=22.0, memory_usage=42.0,
                             process_count=138, new_process_count=0),
    'Stolen credentials': dict(failed_login_count=0, successful_login=1, failure_rate=0.0, time_between_attempts=35.0,
                               login_frequency=0.4, request_count=60, requests_per_second=7.0, unique_endpoints=7,
                               bytes_sent=19000, bytes_received=2200, error_count=7, cpu_usage=70.0,
                               memory_usage=73.0, process_count=178, new_process_count=3),
    'Full kill chain': dict(failed_login_count=20, successful_login=1, failure_rate=0.95, time_between_attempts=0.7,
                            login_frequency=32.0, request_count=66, requests_per_second=10.0, unique_endpoints=7,
                            bytes_sent=18500, bytes_received=2600, error_count=10, cpu_usage=77.0,
                            memory_usage=77.0, process_count=185, new_process_count=4),
}

# (feature, label, min, max, step) — sliders shown in the What-If Lab
SLIDERS = {
    'auth': [('failed_login_count', 0, 30, 1), ('login_frequency', 0.0, 50.0, 0.5),
             ('time_between_attempts', 0.0, 60.0, 0.5)],
    'network': [('request_count', 0, 100, 1), ('requests_per_second', 0.0, 16.0, 0.1),
                ('unique_endpoints', 1, 10, 1), ('error_count', 0, 20, 1)],
    'system': [('cpu_usage', 0.0, 100.0, 1.0), ('memory_usage', 20.0, 100.0, 1.0),
               ('process_count', 100, 240, 1), ('new_process_count', 0, 8, 1)],
}


# ═══════════════════════════════════════
# PAGE + STYLE
# ═══════════════════════════════════════

st.set_page_config(page_title="TRIAD · Threat Monitor", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root { --line: rgba(148,178,214,.13); --t2: #9fb0c4; --t3: #64748b; --mono: 'JetBrains Mono', Consolas, monospace; }
html, body, [class*="css"], .stApp { font-family: 'Inter', 'Segoe UI', sans-serif; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.2rem; max-width: 1700px; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, var(--glow, rgba(45,212,191,.10)), transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, rgba(59,91,219,.10), transparent 60%), #05080d;
         transition: background 1s; }
.mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }

.brand { display:flex; align-items:center; gap:14px; margin-bottom: 4px; }
.brand-name { font-weight:800; letter-spacing:.32em; font-size:20px; color:#e6edf6; }
.brand-sub { font-size:12px; color:var(--t3); letter-spacing:.04em; }
.brand-dot { width:12px; height:12px; border-radius:50%; background: var(--accent, #2dd4bf); box-shadow: 0 0 14px var(--accent, #2dd4bf); }

.panel { background: linear-gradient(160deg, rgba(22,32,46,.72), rgba(14,21,31,.55));
         backdrop-filter: blur(20px) saturate(150%); -webkit-backdrop-filter: blur(20px) saturate(150%);
         border: 1px solid var(--line); border-radius: 18px; padding: 18px 20px; margin-bottom: 14px;
         box-shadow: 0 20px 50px -30px rgba(0,0,0,.8), inset 0 1px 0 rgba(255,255,255,.04); }
.ph { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px; gap:10px; }
.ph h4 { white-space:nowrap; margin:0; padding:0; font-size:12.5px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:var(--t2); }
.ph span { font-size:11.5px; color:var(--t3); }

.hero { display:flex; align-items:center; gap:26px; min-height: 226px; }
.gauge { position:relative; flex-shrink:0; }
.gauge svg { width:100%; height:100%; transform: rotate(-90deg); }
.gauge circle { fill:none; stroke-width:13; }
.g-track { stroke: rgba(148,178,214,.09); }
.g-arc { stroke-linecap: round; transition: stroke-dasharray .8s cubic-bezier(.2,.8,.2,1), stroke .5s; }
.g-center { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.g-val { font: 700 52px var(--mono); letter-spacing:-.04em; line-height:1; color:#e6edf6; }
.g-val small { font-size:20px; color:var(--t2); }
.g-lbl { max-width:110px; font-size:10px; color:var(--t3); text-transform:uppercase; letter-spacing:.12em; margin-top:8px; text-align:center; }
.eyebrow { font-size:11px; color:var(--t3); letter-spacing:.06em; text-transform:uppercase; }
.verdict { font-size:46px; font-weight:800; letter-spacing:.02em; line-height:1.05; margin:6px 0 10px; }
.reason { color:var(--t2); font-size:14px; line-height:1.55; max-width:52ch; }
.stats { display:flex; gap:26px; margin-top:14px; padding-top:12px; border-top:1px solid var(--line); }
.stats div { display:flex; flex-direction:column; gap:2px; }
.stats .k { font-size:10.5px; color:var(--t3); text-transform:uppercase; letter-spacing:.08em; }
.stats .v { font: 600 16px var(--mono); color:#e6edf6; }

.chain { display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; }
.stage { position:relative; padding:13px 14px; border-radius:14px; border:1px solid var(--line); background: rgba(255,255,255,.02); }
.stage .tid { font: 700 11px var(--mono); color:var(--t3); }
.stage .sn { font-size:14px; font-weight:700; margin:6px 0 3px; color:var(--t2); }
.stage .sm { font-size:11.5px; color:var(--t3); }
.stage .dot { position:absolute; top:14px; right:14px; width:9px; height:9px; border-radius:50%; background: rgba(148,178,214,.25); }
.stage.lit .tid, .stage.lit .sn { color:#e6edf6; }

.fusion { display:flex; align-items:center; gap:10px; margin-top:16px; flex-wrap:wrap; }
.fnode { font-size:12px; font-weight:600; padding:8px 12px; border-radius:10px; border:1px solid var(--line); color:var(--t2); background: rgba(255,255,255,.02); }
.farrow { color: var(--t3); font-size: 14px; }
.fcore { font-size:12px; font-weight:700; padding:8px 12px; border-radius:10px; border:1px solid rgba(148,178,214,.35); color:#e6edf6; }
.fout { font-size:14px; font-weight:800; letter-spacing:.08em; padding:8px 16px; border-radius:10px; }

.src { position:relative; overflow:hidden; }
.src::before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background: var(--c); opacity:.5; }
.src-h { display:flex; justify-content:space-between; align-items:center; }
.src-n { font-size:12.5px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:var(--t2); }
.chip { font: 700 10.5px var(--mono); letter-spacing:.1em; padding:4px 9px; border-radius:999px; color:var(--t3); border:1px solid rgba(148,178,214,.25); }
.src-score { display:flex; align-items:baseline; gap:8px; margin:8px 0 10px; }
.src-score .mono { font-size:44px; font-weight:700; letter-spacing:-.03em; color:#e6edf6; }
.src-score small { font-size:10.5px; color:var(--t3); text-transform:uppercase; letter-spacing:.1em; }
.facts { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.facts div { display:flex; flex-direction:column; gap:2px; }
.facts .big { font: 700 22px var(--mono); color:#e6edf6; }
.facts .hot { font-size:30px; color: var(--c); text-shadow: 0 0 18px var(--c); }
.facts .lbl { font-size:11.5px; color:var(--t3); }
.spark { width:100%; height:42px; margin-top:10px; display:block; }
.src.elevated { border-color: rgba(251,191,36,.38); }
.src.elevated .chip { color:#fbbf24; border-color: rgba(251,191,36,.5); background: rgba(251,191,36,.08); }
.src.hostile { border-color: var(--c); box-shadow: 0 0 46px -12px var(--c), inset 0 1px 0 rgba(255,255,255,.05); }
.src.hostile::before { opacity:1; box-shadow: 0 0 18px var(--c); }
.src.hostile .chip { color:#fff; background:#ff3b5c; border-color:#ff3b5c; animation: blink 1s infinite; }
.src.hostile .src-score .mono { color: var(--c); }
@keyframes blink { 50% { opacity:.55; } }

.replay { display:flex; flex-direction:column; gap:7px; }
.rrow { display:grid; grid-template-columns: 104px 1fr 104px; align-items:center; gap:10px; }
.pips { display:flex; gap:4px; }
.pip { font: 700 10px var(--mono); width:30px; text-align:center; padding:3px 0; border-radius:6px; color:var(--t3); border:1px dashed rgba(148,178,214,.25); }
.bar { height:8px; border-radius:4px; background: rgba(148,178,214,.08); position:relative; overflow:hidden; }
.bar i { position:absolute; left:0; top:0; bottom:0; border-radius:4px; }
.bar b { position:absolute; left:50%; top:0; bottom:0; width:1px; background: rgba(255,255,255,.4); }
.res { font: 700 11px var(--mono); text-align:right; }
.res span { color:#9fb0c4; font-weight:500; margin-right:6px; }
.rrow.all { padding-top:8px; margin-top:2px; border-top:1px solid var(--line); }
.foot { font-size:12px; color:var(--t3); margin-top:12px; line-height:1.5; }
.foot b { color:#e6edf6; }

.dev { display:flex; flex-direction:column; gap:11px; }
.drow { display:grid; grid-template-columns: 128px 1fr 92px; align-items:center; gap:12px; }
.dn { font-size:13px; color:var(--t2); }
.dtrack { position:relative; height:10px; border-radius:5px; background: rgba(148,178,214,.07); }
.dnorm { position:absolute; left:0; top:0; bottom:0; width:25%; border-radius:5px; background: rgba(45,212,191,.15); border-right:1px dashed rgba(45,212,191,.6); }
.dfill { position:absolute; left:0; top:2px; bottom:2px; border-radius:4px; transition: width .7s; }
.dv { font: 700 13px var(--mono); text-align:right; color:var(--t2); }
.dv em { display:block; font-style:normal; font-size:10.5px; color:var(--t3); }
.drow.over .dv { color:#e6edf6; } .drow.over .dv em { color:#ff3b5c; }

.shap-row { display:grid; grid-template-columns: 138px 1fr 54px; align-items:center; gap:10px; margin-bottom:7px; }
.shap-n { font-size:12.5px; color:var(--t2); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.shap-track { position:relative; height:10px; }
.shap-track::before { content:""; position:absolute; left:50%; top:-3px; bottom:-3px; width:1px; background: rgba(255,255,255,.3); }
.shap-track i { position:absolute; top:0; bottom:0; border-radius:3px; }
.shap-v { font: 700 11.5px var(--mono); text-align:right; }
.shap-sep { height:1px; background: var(--line); margin: 10px 0 12px; }

.feed { display:flex; flex-direction:column; gap:4px; }
.ev { display:grid; grid-template-columns: 70px 86px 1fr; gap:10px; align-items:center; font: 500 12px var(--mono);
      padding:6px 10px; border-radius:8px; color:var(--t2); border-left:2px solid transparent; background: rgba(255,255,255,.015); }
.ev .t { color:var(--t3); } .ev .tag { font-weight:700; font-size:10.5px; letter-spacing:.06em; }
.ev.fail { border-left-color:#a78bfa; } .ev.fail .tag { color:#ff3b5c; }
.ev.ok .tag { color:#2dd4bf; }
.ev.breach { border-left-color:#ff3b5c; background: rgba(255,59,92,.08); } .ev.breach .tag { color:#ff3b5c; }
.ev.net { border-left-color:#38bdf8; } .ev.net .tag { color:#38bdf8; }
.ev.err .tag { color:#fbbf24; }
.empty { color: var(--t3); font-size: 13px; padding: 8px 2px; }

.banner { text-align:center; padding:14px 20px; border-radius:16px; margin-bottom:14px;
          background: rgba(60,6,16,.75); border:1px solid rgba(255,59,92,.7); box-shadow: 0 0 60px -10px rgba(255,59,92,.7);
          animation: flash 1.2s ease-out; }
.banner .bt { font-weight:800; font-size:22px; letter-spacing:.2em; color:#fff; }
.banner .bs { font: 500 13px var(--mono); color:#ffb3c0; margin-top:4px; }
@keyframes flash { 0% { transform: scale(.97); box-shadow: 0 0 140px 20px rgba(255,59,92,.9); } }

.kpi { padding:16px 18px; }
.kpi .k { font-size:11px; color:var(--t3); text-transform:uppercase; letter-spacing:.08em; }
.kpi .v { font: 700 34px var(--mono); color:#e6edf6; margin-top:6px; letter-spacing:-.02em; }
.kpi .d { font-size:12.5px; color:var(--t2); margin-top:4px; }
.kpi.hl { border-color: rgba(45,212,191,.45); box-shadow: 0 0 40px -14px rgba(45,212,191,.6); }
.kpi.hl .v { color:#2dd4bf; }

table.cmp { width:100%; border-collapse:collapse; font-size:13px; }
table.cmp th { text-align:left; font-size:10.5px; color:var(--t3); text-transform:uppercase; letter-spacing:.08em; padding:8px 10px; border-bottom:1px solid var(--line); }
table.cmp td { padding:10px; border-bottom:1px solid rgba(148,178,214,.06); color:var(--t2); font-family: var(--mono); }
table.cmp td:first-child { font-family: 'Inter', sans-serif; color:#e6edf6; font-weight:600; }
table.cmp tr.best td { background: rgba(45,212,191,.07); color:#e6edf6; }
table.cmp tr.best td:first-child { box-shadow: inset 3px 0 0 #2dd4bf; }
.cellbar { display:inline-block; height:6px; border-radius:3px; background:#38bdf8; margin-left:8px; vertical-align:middle; opacity:.7; }

.cm { display:grid; grid-template-columns: 1fr 1fr; gap:4px; margin-top:8px; }
.cm div { border-radius:8px; padding:12px 6px; text-align:center; font: 700 20px var(--mono); }
.cm small { display:block; font: 500 9.5px 'Inter', sans-serif; letter-spacing:.06em; text-transform:uppercase; opacity:.75; margin-top:2px; }
.cm-title { font-size:12.5px; font-weight:700; color:#e6edf6; }
.cm-sub { font-size:11px; color:var(--t3); }

div[data-testid="stTabs"] button p { font-size: 14px; font-weight: 600; }
.st-key-timeline_box { background: linear-gradient(160deg, rgba(22,32,46,.72), rgba(14,21,31,.55)) !important;
    border: 1px solid var(--line) !important; border-radius: 18px !important; padding: 18px 16px 6px !important;
    box-shadow: 0 20px 50px -30px rgba(0,0,0,.8); }
.st-key-incident_box { border: 1px solid rgba(255,59,92,.45) !important; border-radius: 18px !important;
    background: rgba(60,6,16,.35) !important; padding: 14px 18px !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# HTML BUILDERS
# ═══════════════════════════════════════

def esc(s):
    return html.escape(str(s))


def level_of(score):
    return 'hostile' if score >= 50 else 'elevated' if score >= 25 else 'clear'


def gauge(score, color, size=196, label='threat score'):
    r = 80
    circ = 2 * math.pi * r
    arc = circ * max(0.0, min(score, 100.0)) / 100
    return (f'<div class="gauge" style="width:{size}px;height:{size}px">'
            f'<svg viewBox="0 0 200 200"><circle class="g-track" cx="100" cy="100" r="{r}"/>'
            f'<circle class="g-arc" cx="100" cy="100" r="{r}" style="stroke:{color};stroke-dasharray:{arc:.1f} {circ:.1f};'
            f'opacity:{1 if arc >= 1 else 0};'
            f'filter:drop-shadow(0 0 8px {color})"/></svg>'
            f'<div class="g-center"><div class="g-val">{score:.0f}<small>%</small></div>'
            f'<div class="g-lbl">{label}</div></div></div>')


def spark(values, color):
    if len(values) < 2:
        return '<svg class="spark"></svg>'
    w, h = 300, 40
    step = w / (HISTORY_POINTS - 1)
    start = HISTORY_POINTS - len(values)
    pts = ' '.join(f'{(start + i) * step:.1f},{h - 2 - v / 100 * (h - 4):.1f}' for i, v in enumerate(values))
    first_x = start * step
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<polygon points="{first_x:.1f},{h} {pts} {w},{h}" fill="{color}" opacity=".12"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>')


def reason_text(verdict, data):
    f = data.get('features', {})
    parts = []
    if data.get('auth_score', 0) >= 50:
        parts.append(f"{f.get('failed_login_count', 0)} failed logins")
    if data.get('network_score', 0) >= 50:
        parts.append(f"{f.get('requests_per_second', 0):.1f} req/s across {f.get('unique_endpoints', 0)} endpoints")
    if data.get('system_score', 0) >= 50:
        parts.append(f"CPU {f.get('cpu_usage', 0):.0f}% with {f.get('new_process_count', 0)} new processes")
    if verdict == 'ATTACK':
        return 'Fusion confirmed an attack: ' + (' + '.join(parts) if parts else 'correlated multi-source evidence') + '.'
    if verdict == 'SUSPICIOUS':
        lead = ', '.join(parts) if parts else 'some signals are elevated'
        return (f"Single-source alarm: {lead}. The fused model scores only {data.get('threat_score', 0):.0f}% "
                f"(needs 50%), so the attack is not confirmed.")
    return 'No hostile activity across authentication, network or host telemetry.'


def score_color(score):
    """Gauge colour follows the model's probability; the verdict word carries the verdict colour."""
    return DANGER if score >= 50 else WARN if score >= 40 else SAFE


def hero_html(verdict, score, reason, stats):
    word, color = VERDICT[verdict]
    stat_html = ''.join(f'<div><span class="k">{esc(k)}</span><span class="v">{esc(v)}</span></div>' for k, v in stats)
    return (f'<div class="panel hero" style="border-color:{color}55;box-shadow:0 0 60px -22px {color}">'
            f'{gauge(score, score_color(score))}<div><div class="eyebrow">Fused verdict · 15 features · 3 sources</div>'
            f'<div class="verdict" style="color:{color};text-shadow:0 0 30px {color}44">{word}</div>'
            f'<div class="reason">{esc(reason)}</div><div class="stats">{stat_html}</div></div></div>')


def chain_html(lit, verdict, scores):
    cells = ''
    for key, tid, name, src, desc in STAGES:
        c = SOURCE_COLOR[src]
        if key in lit:
            cells += (f'<div class="stage lit" style="border-color:{c};background:{c}1f;box-shadow:0 0 28px -8px {c}">'
                      f'<div class="dot" style="background:{c};box-shadow:0 0 12px {c}"></div>'
                      f'<div class="tid">{tid}</div><div class="sn">{name}</div>'
                      f'<div class="sm" style="color:{c}">seen {esc(lit[key])}</div></div>')
        else:
            cells += (f'<div class="stage"><div class="dot"></div><div class="tid">{tid}</div>'
                      f'<div class="sn">{name}</div><div class="sm">{desc}</div></div>')
    nodes = ''
    for src in ('auth', 'network', 'system'):
        c = SOURCE_COLOR[src]
        on = scores.get(src, 0) >= 50
        style = f'border-color:{c};color:#e6edf6;background:{c}22;box-shadow:0 0 18px -6px {c}' if on else ''
        nodes += f'<span class="fnode" style="{style}">{SOURCE_NAME[src]}</span>'
    word, color = VERDICT[verdict]
    return (f'<div class="panel"><div class="ph"><h4>Attack kill chain · MITRE ATT&amp;CK</h4>'
            f'<span>Techniques observed in this episode</span></div><div class="chain">{cells}</div>'
            f'<div class="fusion">{nodes}<span class="farrow">⟶</span><span class="fcore">Feature fusion · RF</span>'
            f'<span class="farrow">⟶</span><span class="fout" style="color:{color};border:1px solid {color};'
            f'background:{color}1c">{word}</span></div></div>')


def source_html(src, score, facts, history):
    lvl = level_of(score)
    c = SOURCE_COLOR[src]
    fact_html = ''.join(f'<div><span class="big{" hot" if hot else ""}">{esc(v)}</span><span class="lbl">{esc(l)}</span></div>'
                        for v, l, hot in facts)
    return (f'<div class="panel src {lvl}" style="--c:{c}"><div class="src-h"><span class="src-n">{SOURCE_NAME[src]}</span>'
            f'<span class="chip">{lvl.upper() if lvl != "clear" else "CLEAR"}</span></div>'
            f'<div class="src-score"><span class="mono">{score:.0f}</span><small>indicator score</small></div>'
            f'<div class="facts">{fact_html}</div>{spark(history, c)}</div>')


def replay_html(replay, title_note):
    if not replay:
        return '<div class="panel"><div class="ph"><h4>Blind spot replay</h4></div><div class="empty">Unavailable.</div></div>'
    rows = ''
    for r in replay:
        srcs = r['sources']
        pips = ''.join(
            f'<span class="pip" style="background:{SOURCE_COLOR[s]};color:#0b0f17;border:1px solid {SOURCE_COLOR[s]}">{s[0].upper()}</span>'
            if s in srcs else f'<span class="pip">{s[0].upper()}</span>' for s in ('auth', 'network', 'system'))
        p = r['probability'] * 100
        caught = r['detected']
        col = DANGER if caught else '#64748b'
        glow = f';box-shadow:0 0 10px {DANGER}' if caught else ''
        rows += (f'<div class="rrow{" all" if len(srcs) == 3 else ""}"><div class="pips">{pips}</div>'
                 f'<div class="bar"><i style="width:{p:.0f}%;background:{col}{glow}"></i><b></b></div>'
                 f'<div class="res" style="color:{col}"><span>{p:.0f}%</span>{"CAUGHT" if caught else "MISSED"}</div></div>')
    singles = [r for r in replay if len(r['sources']) == 1 and r['detected']]
    fused = replay[-1]['detected']
    if fused and not singles:
        foot = '<b>No single source catches this — only the fused model does.</b>'
    elif fused:
        foot = 'Caught by: ' + ', '.join(SOURCE_NAME[r['sources'][0]] for r in singles) + ' alone, and by fusion.'
    else:
        foot = 'Fused model below the decision line (50%).'
    return (f'<div class="panel"><div class="ph"><h4>Blind spot replay</h4><span>{esc(title_note)}</span></div>'
            f'<div class="replay">{rows}</div><div class="foot">{foot}<br>Switched-off sources are replaced '
            f'with a typical benign session. White tick = decision line.</div></div>')


def shap_html(explained, title='Why this verdict · SHAP'):
    if explained is None:
        return (f'<div class="panel"><div class="ph"><h4>{title}</h4></div>'
                f'<div class="empty">Install SHAP to enable: pip install shap</div></div>')
    base, contrib = explained
    groups = {s: sum(v for f, v in contrib.items() if MODALITY_OF[f] == s) for s in SOURCE_COLOR}
    scale = max([abs(v) for v in groups.values()] + [abs(v) for v in contrib.values()] + [0.05])

    def row(name, v, color=None):
        w = abs(v) / scale * 50
        col = color or (DANGER if v > 0 else SAFE)
        left = 50 if v >= 0 else 50 - w
        return (f'<div class="shap-row"><span class="shap-n">{esc(name)}</span><div class="shap-track">'
                f'<i style="left:{left:.1f}%;width:{w:.1f}%;background:{col}"></i></div>'
                f'<span class="shap-v" style="color:{DANGER if v > 0 else SAFE}">{v * 100:+.0f}</span></div>')

    body = ''.join(row(SOURCE_NAME[s], v, SOURCE_COLOR[s]) for s, v in groups.items())
    body += '<div class="shap-sep"></div>'
    top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:6]
    body += ''.join(row(FEATURE_LABEL[f], v) for f, v in top)
    return (f'<div class="panel"><div class="ph"><h4>{title}</h4><span>+ attack · − benign</span></div>'
            f'{body}<div class="foot">Starts from the model average of {base * 100:.0f}%. Top: per source · bottom: strongest features.</div></div>')


def baseline_html(f):
    rows = ''
    for key, name, unit, ceil, fmt in BASELINES:
        v = float(f.get(key, 0) or 0)
        ref = ceil if ceil > 0 else 1
        over = v > ceil
        ratio = v / ref
        width = 25 + 75 * min(math.log(ratio) / math.log(30), 1) if over and ratio > 1 else min(ratio * 25, 100)
        if over:
            note = (f'{ratio:.1f}× normal' if ratio < 10 else f'{ratio:.0f}× normal') if ceil > 0 else 'normal is 0'
        else:
            note = f'normal ≤ {fmt.format(ceil)}'
        col = DANGER if over else SAFE
        glow = f';box-shadow:0 0 12px {DANGER}' if over else ''
        rows += (f'<div class="drow{" over" if over else ""}"><span class="dn">{name}</span>'
                 f'<div class="dtrack"><div class="dnorm"></div><div class="dfill" style="width:{width:.1f}%;background:{col}{glow}"></div></div>'
                 f'<span class="dv">{fmt.format(v)}{(" " + unit) if unit else ""}<em>{note}</em></span></div>')
    return (f'<div class="panel"><div class="ph"><h4>Baseline vs now</h4><span>Shaded = normal-user range</span></div>'
            f'<div class="dev">{rows}</div></div>')


def feed_html(events):
    if not events:
        body = '<div class="empty">Waiting for activity on localhost:5000…</div>'
    else:
        body = ''
        fails = 0
        rendered = []
        for e in events:
            clock = datetime.fromtimestamp(e['t']).strftime('%H:%M:%S')
            if e['kind'] == 'auth':
                if e['event'] == 'LOGIN_FAILED':
                    fails += 1
                    rendered.append(('fail', clock, 'AUTH FAIL', f"user={e['user']}"))
                elif e['event'] == 'LOGIN_SUCCESS':
                    if fails >= 3:
                        rendered.append(('breach', clock, 'COMPROMISE', f"user={e['user']} after {fails} failures"))
                    else:
                        rendered.append(('ok', clock, 'LOGIN', f"user={e['user']}"))
                else:
                    rendered.append(('ok', clock, e['event'], f"user={e['user']}"))
            else:
                cls = 'err' if e['status'].isdigit() and int(e['status']) >= 400 else 'net'
                rendered.append((cls, clock, e['method'], f"{e['path']} → {e['status']}"))
        for cls, clock, tag, text in reversed(rendered[-10:]):
            body += f'<div class="ev {cls}"><span class="t">{clock}</span><span class="tag">{esc(tag)}</span><span>{esc(text)}</span></div>'
    return (f'<div class="panel"><div class="ph"><h4>Live events</h4><span>{len(events)} in the last 60 s</span></div>'
            f'<div class="feed">{body}</div></div>')


# ═══════════════════════════════════════
# DATA
# ═══════════════════════════════════════

def _tail(path, n=400):
    if not os.path.exists(path):
        return []
    with open(path, 'r') as fh:
        return list(deque(fh, maxlen=n))


def recent_events(window=60):
    """Authentication + access events from the last `window` seconds (console traffic excluded)."""
    cutoff = datetime.now().timestamp() - window
    events = []
    for kind, name in (('auth', 'auth.log'), ('access', 'access.log')):
        for line in _tail(os.path.join(LOG_DIR, name)):
            parts = [p.strip() for p in line.strip().split(' | ')]
            if len(parts) < 3:
                continue
            try:
                t = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S.%f').timestamp()
            except ValueError:
                continue
            if t < cutoff:
                continue
            if kind == 'auth':
                events.append({'t': t, 'kind': 'auth', 'event': parts[1], 'user': parts[2].replace('user=', '')})
            elif len(parts) >= 4 and not parts[2].startswith(INTERNAL_PREFIXES) and parts[2] != '/login':
                events.append({'t': t, 'kind': 'access', 'method': parts[1], 'path': parts[2], 'status': parts[3]})
    events.sort(key=lambda e: e['t'])
    return events


def active_stages(f, events):
    """Which MITRE techniques the current window shows evidence of."""
    probes = sum(1 for e in events if e['kind'] == 'access')
    found = set()
    if f.get('failed_login_count', 0) >= 5:
        found.add('brute')
    if f.get('successful_login', 0) and (f.get('failed_login_count', 0) >= 3 or probes >= 15):
        found.add('valid')
    if probes >= 15:
        found.add('scan')
    if f.get('cpu_usage', 0) >= 60 and f.get('new_process_count', 0) >= 2:
        found.add('hijack')
    return found


@st.cache_data
def load_metrics():
    path = os.path.join(RESULTS_DIR, 'metrics.json')
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return json.load(fh)


@st.cache_data
def load_cicids():
    path = os.path.join(CICIDS_RESULTS_DIR, 'metrics.json')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def as_shap(raw):
    """[base, {feature: contribution}] from the API -> (base, dict) or None."""
    return (raw[0], raw[1]) if raw else None


@st.cache_data(ttl=600)
def load_importances():
    try:
        return requests.get(f'{FLASK_URL}/api/model-info', timeout=5).json().get('feature_importances', {})
    except Exception:
        return {}


def score_scenario(x):
    try:
        return requests.post(f'{FLASK_URL}/api/whatif', json=x, timeout=5).json()
    except Exception:
        return None


# ═══════════════════════════════════════
# INCIDENT REPORT
# ═══════════════════════════════════════

def incident_report(ep):
    d = ep['snapshot']
    f = d.get('features', {})
    word, color = VERDICT['ATTACK']
    stage_rows = ''.join(
        f"<tr><td>{tid}</td><td>{name}</td><td>{SOURCE_NAME[src]}</td><td>{esc(ep['stages'].get(key, '—'))}</td></tr>"
        for key, tid, name, src, _ in STAGES)
    feat_rows = ''.join(
        f"<tr><td>{FEATURE_LABEL[c]}</td><td>{SOURCE_NAME[MODALITY_OF[c]]}</td><td>{f.get(c, 0)}</td></tr>"
        for c in FEATURE_COLUMNS)
    replay_rows = ''.join(
        f"<tr><td>{' + '.join(SOURCE_NAME[s] for s in r['sources'])}</td><td>{r['probability'] * 100:.1f}%</td>"
        f"<td style='color:{DANGER if r['detected'] else '#64748b'};font-weight:700'>{'CAUGHT' if r['detected'] else 'MISSED'}</td></tr>"
        for r in d.get('blind_spot', []))
    shap_rows = ''
    if ep.get('shap'):
        base, contrib = ep['shap']
        for c, v in sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:8]:
            shap_rows += f"<tr><td>{FEATURE_LABEL[c]}</td><td>{SOURCE_NAME[MODALITY_OF[c]]}</td><td>{v * 100:+.1f} pts</td></tr>"
    first_alert = f"{ep['first_alert_s']:.1f} s after the first hostile event" if ep.get('first_alert_s') is not None else '—'
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Incident {ep['id']}</title>
<style>
body {{ font-family: 'Segoe UI', Inter, sans-serif; background:#0b1018; color:#e6edf6; margin:0; padding:40px; }}
.wrap {{ max-width: 920px; margin: 0 auto; }}
h1 {{ margin:0 0 4px; letter-spacing:.06em; }} h2 {{ font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:#9fb0c4; margin:30px 0 10px; }}
.tag {{ display:inline-block; padding:6px 14px; border-radius:8px; background:{color}22; border:1px solid {color}; color:{color}; font-weight:800; letter-spacing:.1em; }}
.grid {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin-top:20px; }}
.card {{ background:#121a26; border:1px solid #1f2b3b; border-radius:12px; padding:14px; }}
.card .k {{ font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.08em; }} .card .v {{ font: 700 22px Consolas, monospace; margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }} td, th {{ text-align:left; padding:8px 10px; border-bottom:1px solid #1f2b3b; }}
th {{ color:#64748b; font-size:11px; text-transform:uppercase; letter-spacing:.08em; }}
.muted {{ color:#64748b; font-size:12px; }}
@media print {{ body {{ background:#fff; color:#111; }} .card {{ background:#f5f7fa; border-color:#ddd; }} td, th {{ border-color:#ddd; }} }}
</style></head><body><div class="wrap">
<div class="muted">TRIAD · Multi-Modal Threat Detection · Incident report</div>
<h1>Incident {ep['id']}</h1><span class="tag">{word} CONFIRMED</span>
<div class="grid">
<div class="card"><div class="k">Confirmed at</div><div class="v">{esc(ep['confirmed_clock'])}</div></div>
<div class="card"><div class="k">Peak threat</div><div class="v">{ep['peak']:.1f}%</div></div>
<div class="card"><div class="k">First alert</div><div class="v" style="font-size:15px">{esc(first_alert)}</div></div>
<div class="card"><div class="k">Sources hostile</div><div class="v" style="font-size:15px">{esc(', '.join(ep['hostile_sources']) or '—')}</div></div>
</div>
<h2>Summary</h2><p>{esc(reason_text('ATTACK', d))}</p>
<h2>Kill chain · MITRE ATT&amp;CK</h2><table><tr><th>Technique</th><th>Name</th><th>Source</th><th>First seen</th></tr>{stage_rows}</table>
<h2>Indicator scores</h2><table><tr><th>Authentication</th><th>Network</th><th>System</th><th>Fused (RF)</th></tr>
<tr><td>{d.get('auth_score', 0):.0f}</td><td>{d.get('network_score', 0):.0f}</td><td>{d.get('system_score', 0):.0f}</td><td>{d.get('threat_score', 0):.1f}%</td></tr></table>
<h2>Blind spot replay — would a single source have caught it?</h2><table><tr><th>Sources visible</th><th>Attack probability</th><th>Result</th></tr>{replay_rows}</table>
<h2>Why — top SHAP contributions</h2><table><tr><th>Feature</th><th>Source</th><th>Contribution</th></tr>{shap_rows or '<tr><td colspan=3>SHAP not available</td></tr>'}</table>
<h2>Telemetry at peak (60 s window)</h2><table><tr><th>Feature</th><th>Source</th><th>Value</th></tr>{feat_rows}</table>
<p class="muted" style="margin-top:30px">Model: Random Forest (100 trees) trained offline on 15 fused features (5 authentication, 6 network, 4 system).
Incident date {datetime.fromtimestamp(ep['start']).strftime('%Y-%m-%d')}.</p>
</div></body></html>"""


# ═══════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════

ss = st.session_state
ss.setdefault('history', [])
ss.setdefault('episode', None)
ss.setdefault('last_incident', None)
ss.setdefault('prev_verdict', 'BENIGN')
for _feat, _val in PRESETS['Normal user'].items():
    ss.setdefault(f'wf_{_feat}', _val)
ss.setdefault('wf_login_ok', True)


def reset_live_state():
    ss.history = []
    ss.episode = None
    ss.last_incident = None
    ss.prev_verdict = 'BENIGN'


def track_episode(data, events, now):
    """Latch stages, peak and the incident snapshot for the current non-benign episode."""
    verdict = data.get('verdict', data.get('label', 'BENIGN'))
    ep = ss.episode
    if ep and verdict == 'BENIGN' and now - ep['last_active'] > EPISODE_COOLDOWN:
        ss.episode = ep = None
    if verdict == 'BENIGN' and ep is None:
        return None
    if ep is None:
        fails = [e['t'] for e in events if e['kind'] == 'auth' and e['event'] == 'LOGIN_FAILED' and e['t'] >= now - 20]
        probes = [e['t'] for e in events if e['kind'] == 'access' and e['t'] >= now - 15]
        first_hostile = min(fails) if fails else (min(probes) if probes else None)
        ep = ss.episode = {
            'id': datetime.now().strftime('INC-%H%M%S'), 'start': now, 'last_active': now,
            'first_alert_s': (now - first_hostile) if first_hostile else None,
            'peak': 0.0, 'stages': {}, 'confirmed_clock': None, 'snapshot': None, 'shap': None,
            'hostile_sources': [],
        }
    if verdict != 'BENIGN':
        ep['last_active'] = now
    for key in active_stages(data.get('features', {}), events):
        ep['stages'].setdefault(key, datetime.fromtimestamp(now).strftime('%H:%M:%S'))
    score = data.get('threat_score', 0)
    if verdict == 'ATTACK' and ep['confirmed_clock'] is None:
        ep['confirmed_clock'] = datetime.fromtimestamp(now).strftime('%H:%M:%S')
    if verdict == 'ATTACK' and (ep['snapshot'] is None or score > ep['peak']):
        ep['peak'] = score
        ep['snapshot'] = data
        ep['shap'] = as_shap(data.get('shap'))
        ep['hostile_sources'] = [SOURCE_NAME[s] for s in SOURCE_COLOR if data.get(f'{s}_score', 0) >= 50]
        ss.last_incident = ep
    elif ep['confirmed_clock'] is None:
        ep['peak'] = max(ep['peak'], score)
    return ep


# ═══════════════════════════════════════
# HEADER
# ═══════════════════════════════════════

head_l, head_r1, head_r2 = st.columns([6, 1, 1])
with head_l:
    st.markdown('<div class="brand"><span class="brand-dot"></span><div><div class="brand-name">TRIAD</div>'
                '<div class="brand-sub">Multi-Modal Cyber Threat Detection · Authentication + Network + System</div></div></div>',
                unsafe_allow_html=True)
with head_r1:
    if st.button('Recalibrate', width='stretch', help='Re-measure this machine\'s idle memory / process baseline'):
        try:
            requests.post(f'{FLASK_URL}/api/calibrate', timeout=3)
            st.toast('Host baseline recalibrated')
        except Exception:
            st.toast('Flask app not reachable')
with head_r2:
    if st.button('Reset logs', width='stretch'):
        try:
            requests.post(f'{FLASK_URL}/api/clear-logs', timeout=3)
            reset_live_state()
            st.toast('Logs cleared — clean slate')
        except Exception:
            st.toast('Flask app not reachable')

tab_live, tab_lab, tab_research = st.tabs(['Live Monitor', 'What-If Lab', 'Research'])


# ═══════════════════════════════════════
# TAB 1: LIVE MONITOR
# ═══════════════════════════════════════

@st.fragment(run_every='1s')
def live_monitor():
    now = datetime.now().timestamp()
    try:
        data = requests.get(f'{FLASK_URL}/api/threat', timeout=3).json()
    except Exception:
        st.markdown('<div class="panel"><div class="ph"><h4>Flask app not reachable</h4></div>'
                    '<div class="empty">Start it with: python app/app.py</div></div>', unsafe_allow_html=True)
        return

    verdict = data.get('verdict', data.get('label', 'BENIGN'))
    score = float(data.get('threat_score', 0))
    f = data.get('features', {})
    scores = {'auth': data.get('auth_score', 0), 'network': data.get('network_score', 0), 'system': data.get('system_score', 0)}
    events = recent_events()

    ss.history = (ss.history + [{'t': now, 'fused': score, **scores, 'verdict': verdict}])[-HISTORY_POINTS:]
    ep = track_episode(data, events, now)

    _, color = VERDICT[verdict]
    st.markdown(f'<style>.stApp {{ --glow: {color}26; --accent: {color}; }}</style>', unsafe_allow_html=True)

    if verdict == 'ATTACK' and ss.prev_verdict != 'ATTACK':
        st.toast('ATTACK CONFIRMED — correlated evidence across sources', icon='🚨')
    ss.prev_verdict = verdict
    if verdict == 'ATTACK' and ep:
        lead = f"first alert {ep['first_alert_s']:.1f} s after first hostile event · " if ep.get('first_alert_s') is not None else ''
        st.markdown(f'<div class="banner"><div class="bt">ATTACK CONFIRMED</div>'
                    f'<div class="bs">{lead}confirmed {ep["confirmed_clock"]}</div></div>', unsafe_allow_html=True)

    # Row 1 — verdict + kill chain
    c1, c2 = st.columns([5, 7])
    with c1:
        stats = [('Window', '60 s'),
                 ('Episode peak', f"{ep['peak']:.0f}%" if ep else '—'),
                 ('First alert', f"+{ep['first_alert_s']:.1f} s" if ep and ep.get('first_alert_s') is not None else '—')]
        st.markdown(hero_html(verdict, score, reason_text(verdict, data), stats), unsafe_allow_html=True)
    with c2:
        st.markdown(chain_html(ep['stages'] if ep else {}, verdict, scores), unsafe_allow_html=True)

    # Row 2 — source cards
    hist = ss.history
    cards = {
        'auth': [(f"{f.get('failed_login_count', 0)}", 'failed logins', f.get('failed_login_count', 0) >= 5),
                 (f"{f.get('login_frequency', 0):.0f}", 'attempts / min', f.get('login_frequency', 0) >= 10)],
        'network': [(f"{f.get('requests_per_second', 0):.1f}", 'requests / s', f.get('requests_per_second', 0) >= 1.5),
                    (f"{f.get('unique_endpoints', 0)}", 'endpoints hit', f.get('unique_endpoints', 0) >= 4)],
        'system': [(f"{f.get('cpu_usage', 0):.0f}%", 'peak CPU', f.get('cpu_usage', 0) >= 60),
                   (f"{f.get('new_process_count', 0)}", 'new processes', f.get('new_process_count', 0) >= 2)],
    }
    for col, src in zip(st.columns(3), ('auth', 'network', 'system')):
        with col:
            st.markdown(source_html(src, scores[src], cards[src], [h[src] for h in hist]), unsafe_allow_html=True)

    # Row 3 — timeline + blind spot replay
    c1, c2 = st.columns([8, 4])
    with c1:
        box = st.container(border=True, key='timeline_box')
        box.markdown('<div class="ph" style="margin-bottom:0"><h4>Threat timeline</h4><span>last 2 minutes · dashed lines = '
                     'suspicious (40%) and attack (50%) thresholds</span></div>', unsafe_allow_html=True)
        df = pd.DataFrame(hist)
        df['ago'] = df['t'] - now  # seconds before now (plain numbers render on every Streamlit version)
        long = df.melt(id_vars=['ago'], value_vars=['fused', 'auth', 'network', 'system'], var_name='series', value_name='score')
        long['series'] = long['series'].map({'fused': 'Fused (RF)', 'auth': 'Authentication', 'network': 'Network', 'system': 'System'})
        palette = alt.Scale(domain=['Fused (RF)', 'Authentication', 'Network', 'System'],
                            range=['#e6edf6', SOURCE_COLOR['auth'], SOURCE_COLOR['network'], SOURCE_COLOR['system']])
        lines = alt.Chart(long).mark_line(interpolate='monotone').encode(
            x=alt.X('ago:Q', title=None, scale=alt.Scale(domain=[-HISTORY_POINTS, 0], nice=False),
                    axis=alt.Axis(labelColor='#64748b', grid=False, domainColor='#1f2b3b', tickCount=8,
                                  labelExpr="datum.value == 0 ? 'now' : datum.value + ' s'")),
            y=alt.Y('score:Q', title=None, scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(labelColor='#64748b', gridColor='#141d29', domain=False, tickCount=5)),
            color=alt.Color('series:N', scale=palette, legend=alt.Legend(orient='top', labelColor='#9fb0c4', title=None)),
            strokeWidth=alt.condition(alt.datum.series == 'Fused (RF)', alt.value(3.2), alt.value(1.6)),
        )
        # Everything is drawn from the one `long` table (multi-table layers break on older Streamlit)
        base = alt.Chart(long)
        area = base.transform_filter(alt.datum.series == 'Fused (RF)').mark_area(
            opacity=0.18, interpolate='monotone', color=color).encode(x='ago:Q', y='score:Q')
        one_row = base.transform_aggregate(n='count()')
        rules = (one_row.mark_rule(strokeDash=[5, 5], opacity=.7, color=WARN).encode(y=alt.datum(40)) +
                 one_row.mark_rule(strokeDash=[5, 5], opacity=.7, color=DANGER).encode(y=alt.datum(50)))
        chart = (area + rules + lines).properties(height=300).configure(background='transparent').configure_view(strokeWidth=0)
        box.altair_chart(chart, use_container_width=True, theme=None)
    with c2:
        snap = ep['snapshot'] if ep and ep.get('snapshot') else None
        if snap:
            st.markdown(replay_html(snap.get('blind_spot', []), f"captured at peak · {snap.get('threat_score', 0):.0f}%"),
                        unsafe_allow_html=True)
        else:
            st.markdown(replay_html(data.get('blind_spot', []), 'live · same moment, sources switched off'),
                        unsafe_allow_html=True)

    # Row 4 — SHAP, baseline vs now, events
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(shap_html(as_shap(data.get('shap'))), unsafe_allow_html=True)
    with c2:
        st.markdown(baseline_html(f), unsafe_allow_html=True)
    with c3:
        st.markdown(feed_html(events), unsafe_allow_html=True)

    # Incident report — the most recent confirmed attack, kept until the logs are reset
    inc = ss.last_incident
    if inc and inc.get('snapshot'):
        box = st.container(border=True, key='incident_box')
        r1, r2 = box.columns([4, 1], vertical_alignment='center')
        r1.markdown(f"<div class='ph' style='margin:0'><h4>Incident {inc['id']}</h4><span>confirmed {inc['confirmed_clock']} · "
                    f"peak {inc['peak']:.0f}% · sources: {esc(', '.join(inc['hostile_sources']) or '—')} · "
                    f"stages {len(inc['stages'])}/4</span></div>", unsafe_allow_html=True)
        r2.download_button('Download incident report', incident_report(inc), file_name=f"{inc['id']}.html",
                           mime='text/html', width='stretch', type='primary', on_click='ignore', key=f"dl_{inc['id']}")


with tab_live:
    live_monitor()



# ═══════════════════════════════════════
# TAB 2: WHAT-IF LAB
# ═══════════════════════════════════════

def apply_preset(name):
    for feat, val in PRESETS[name].items():
        ss[f'wf_{feat}'] = val
    ss['wf_login_ok'] = bool(PRESETS[name]['successful_login'])


with tab_lab:
    st.markdown('<div class="ph"><h4>What-If Lab</h4><span>Move the sliders — the trained Random Forest re-scores instantly. '
                'Values are in the model\'s own (training) units.</span></div>', unsafe_allow_html=True)
    preset_cols = st.columns(len(PRESETS))
    for col, name in zip(preset_cols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), width='stretch')

    ss['wf_successful_login'] = int(ss['wf_login_ok'])
    x = {feat: ss[f'wf_{feat}'] for feat in FEATURE_COLUMNS}
    attempts = x['failed_login_count'] + x['successful_login']
    x['failure_rate'] = round(x['failed_login_count'] / attempts, 4) if attempts else 0.0

    result = score_scenario(x)
    if result is None:
        st.warning('Flask app not reachable — start it with: python app/app.py')
        st.stop()
    p = result['probability'] * 100
    replay = result['blind_spot']
    word, color = VERDICT[result['verdict']]

    out1, out2, out3 = st.columns([5, 4, 4])
    with out1:
        st.markdown(f'<div class="panel hero" style="min-height:0;border-color:{color}55;box-shadow:0 0 60px -22px {color}">'
                    f'{gauge(p, score_color(p), size=170, label="attack probability")}<div>'
                    f'<div class="eyebrow">Model output for this scenario</div>'
                    f'<div class="verdict" style="color:{color}">{word}</div>'
                    f'<div class="reason">Decision line is 50%. Compare "Brute force only" with "Full kill chain", '
                    f'or try "Windows Update": high CPU alone is not an attack.</div></div></div>',
                    unsafe_allow_html=True)
    with out2:
        st.markdown(replay_html(replay, 'this scenario'), unsafe_allow_html=True)
    with out3:
        st.markdown(shap_html(as_shap(result.get('shap'))), unsafe_allow_html=True)

    for col, (src, sliders) in zip(st.columns(3, gap='large'), SLIDERS.items()):
        with col:
            st.markdown(f"<div class='src-n' style='margin:6px 0 4px;color:{SOURCE_COLOR[src]}'>{SOURCE_NAME[src]}</div>",
                        unsafe_allow_html=True)
            if src == 'auth':
                st.checkbox('Login eventually succeeded', key='wf_login_ok')
            for feat, lo, hi, step in sliders:
                st.slider(FEATURE_LABEL[feat], lo, hi, step=step, key=f'wf_{feat}')


# ═══════════════════════════════════════
# TAB 3: RESEARCH
# ═══════════════════════════════════════

NICE = {'A_Network': 'Network only', 'B_Authentication': 'Authentication only', 'C_System': 'System only',
        'D_Network_Auth': 'Network + Authentication', 'E_All_Multimodal': 'All three (multimodal)'}

with tab_research:
    metrics = load_metrics()
    if not metrics:
        st.info('Run `python experiments/multimodal_experiment.py` to generate results.')
    else:
        by = {m['name']: m for m in metrics}
        e = by['E_All_Multimodal']
        singles = [by[k] for k in ('A_Network', 'B_Authentication', 'C_System')]
        best_single = max(singles, key=lambda m: m['f1'])
        worst_single = min(singles, key=lambda m: m['f1'])

        k1, k2, k3, k4 = st.columns(4)
        kpis = [
            (k1, 'Multimodal F1', f"{e['f1']:.3f}", 'All 15 features, fused', True),
            (k2, 'Multimodal accuracy', f"{e['accuracy'] * 100:.1f}%", f"{e['confusion_matrix'][0][1] + e['confusion_matrix'][1][0]} errors in 600 test sessions", False),
            (k3, f"vs {NICE[best_single['name']]}", f"+{(e['f1'] - best_single['f1']) * 100:.1f} pts", 'F1 gain over the best single source', False),
            (k4, f"vs {NICE[worst_single['name']]}", f"+{(e['f1'] - worst_single['f1']) * 100:.1f} pts", 'F1 gain over the weakest single source', False),
        ]
        for col, k, v, d, hl in kpis:
            col.markdown(f'<div class="panel kpi{" hl" if hl else ""}"><div class="k">{esc(k)}</div>'
                         f'<div class="v">{v}</div><div class="d">{esc(d)}</div></div>', unsafe_allow_html=True)

        rows = ''
        for m in metrics:
            cls = ' class="best"' if m['name'] == 'E_All_Multimodal' else ''
            rows += (f"<tr{cls}><td>{NICE[m['name']]}</td><td>{m['n_features']}</td>"
                     f"<td>{m['accuracy']:.3f}</td><td>{m['precision']:.3f}</td><td>{m['recall']:.3f}</td>"
                     f"<td>{m['f1']:.3f}<span class='cellbar' style='width:{max((m['f1'] - 0.8) * 400, 4):.0f}px'></span></td>"
                     f"<td>{m['roc_auc']:.3f}</td><td>{m['train_time']:.2f}s</td></tr>")
        st.markdown(f'<div class="panel"><div class="ph"><h4>Five-way comparison</h4><span>Same 80/20 split · '
                    f'Random Forest (100 trees) · bar = F1 above 0.80</span></div><table class="cmp"><tr><th>Configuration</th>'
                    f'<th>Features</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th>'
                    f'<th>Train time</th></tr>{rows}</table></div>', unsafe_allow_html=True)

        st.markdown('<div class="ph" style="margin-top:8px"><h4>Confusion matrices</h4><span>600 held-out sessions each</span></div>',
                    unsafe_allow_html=True)
        for col, m in zip(st.columns(5), metrics):
            (tn, fp), (fn, tp) = m['confusion_matrix']
            col.markdown(
                f"<div class='panel' style='padding:14px'><div class='cm-title'>{NICE[m['name']]}</div>"
                f"<div class='cm-sub'>{fp + fn} errors</div><div class='cm'>"
                f"<div style='background:rgba(45,212,191,.16);color:#2dd4bf'>{tn}<small>true benign</small></div>"
                f"<div style='background:rgba(255,59,92,{min(.08 + fp / 60, .5):.2f});color:#ff8a9e'>{fp}<small>false alarm</small></div>"
                f"<div style='background:rgba(255,59,92,{min(.08 + fn / 60, .5):.2f});color:#ff8a9e'>{fn}<small>missed attack</small></div>"
                f"<div style='background:rgba(45,212,191,.16);color:#2dd4bf'>{tp}<small>caught attack</small></div>"
                f"</div></div>", unsafe_allow_html=True)

        importances = load_importances()
        if importances:
            imp = pd.DataFrame({'feature': [FEATURE_LABEL[c] for c in FEATURE_COLUMNS],
                                'source': [SOURCE_NAME[MODALITY_OF[c]] for c in FEATURE_COLUMNS],
                                'importance': [importances.get(c, 0) for c in FEATURE_COLUMNS]})
            totals = imp.groupby('source')['importance'].sum()
            c1, c2 = st.columns([8, 4])
            with c1:
                st.markdown('<div class="ph" style="margin-top:8px"><h4>Feature importance · deployed model</h4>'
                            '<span>Gini importance of the multimodal Random Forest</span></div>', unsafe_allow_html=True)
                bars = alt.Chart(imp).mark_bar(cornerRadiusEnd=4).encode(
                    x=alt.X('importance:Q', title=None, axis=alt.Axis(labelColor='#64748b', gridColor='#141d29', format='.0%')),
                    y=alt.Y('feature:N', sort='-x', title=None, axis=alt.Axis(labelColor='#9fb0c4', labelLimit=200)),
                    color=alt.Color('source:N', scale=alt.Scale(domain=list(SOURCE_NAME.values()), range=list(SOURCE_COLOR.values())),
                                    legend=alt.Legend(orient='top', labelColor='#9fb0c4', title=None)),
                    tooltip=['feature', 'source', alt.Tooltip('importance:Q', format='.1%')],
                ).properties(height=420).configure(background='transparent').configure_view(strokeWidth=0)
                st.altair_chart(bars, use_container_width=True, theme=None)
            with c2:
                st.markdown('<div class="ph" style="margin-top:8px"><h4>Importance by source</h4><span>sums to 100%</span></div>',
                            unsafe_allow_html=True)
                for src in ('auth', 'network', 'system'):
                    share = totals.get(SOURCE_NAME[src], 0)
                    st.markdown(f'<div class="panel kpi" style="--c:{SOURCE_COLOR[src]}"><div class="k">{SOURCE_NAME[src]}</div>'
                                f'<div class="v" style="color:{SOURCE_COLOR[src]}">{share * 100:.0f}%</div>'
                                f'<div class="d">{sum(1 for c in FEATURE_COLUMNS if MODALITY_OF[c] == src)} features</div></div>',
                                unsafe_allow_html=True)

        with st.expander('Experiment figures — ROC curves, ablation, computational overhead'):
            figs = [('ROC curves', 'roc_curves_overlay.png'), ('Ablation study', 'ablation_study.png'),
                    ('Computational overhead', 'computational_overhead.png'), ('Comparison chart', 'comparison_chart.png')]
            for col, (title, fname) in zip(st.columns(2) * 2, figs):
                path = os.path.join(RESULTS_DIR, fname)
                if os.path.exists(path):
                    col.image(path, caption=title, use_container_width=True)

    st.markdown('<div class="ph" style="margin-top:18px"><h4>CICIDS2017 network baseline</h4>'
                '<span>Network-flow Random Forest on CICIDS2017-format flows</span></div>', unsafe_allow_html=True)
    cic = load_cicids()
    if cic:
        for col, (k, key) in zip(st.columns(5), [('Accuracy', 'accuracy'), ('Precision', 'precision'), ('Recall', 'recall'),
                                                 ('F1', 'f1'), ('ROC-AUC', 'roc_auc')]):
            col.markdown(f'<div class="panel kpi"><div class="k">{k}</div><div class="v">{float(cic.get(key, 0)):.3f}</div>'
                         f'<div class="d">{cic.get("n_features", "")} flow features</div></div>', unsafe_allow_html=True)
        with st.expander('CICIDS2017 figures'):
            for col, fname in zip(st.columns(3), ['confusion_matrix.png', 'roc_curve.png', 'feature_importance.png']):
                path = os.path.join(CICIDS_RESULTS_DIR, fname)
                if os.path.exists(path):
                    col.image(path, use_container_width=True)
    else:
        st.info('Run `python experiments/cicids_experiment.py` to generate CICIDS2017 results.')
