"""
Flask Login Application — Multi-Modal Cyber Threat Detection
Phase 1: Login web app with telemetry logging
Phase 2: Telemetry collection integrated
Phase 8: Live detection endpoint
"""

import os
import sys
import time
import threading
import json
import psutil
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, session, jsonify

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.feature_extractor import INTERNAL_PREFIXES, FEATURE_COLUMNS
from app import detector

app = Flask(__name__)
app.secret_key = 'cyber-threat-detection-demo-key-2024'

# ── Demo credentials ──
DEMO_USERS = {
    'admin': 'admin123',
    'user1': 'password1',
    'analyst': 'security2024'
}

# ── Log directory setup ──
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

AUTH_LOG = os.path.join(LOG_DIR, 'auth.log')
ACCESS_LOG = os.path.join(LOG_DIR, 'access.log')
SYSTEM_LOG = os.path.join(LOG_DIR, 'system.log')


# ══════════════════════════════════════
# TELEMETRY LOGGING (Phase 2)
# ══════════════════════════════════════

def log_auth_event(event_type, username, ip):
    """Log authentication events to auth.log"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    line = f"{timestamp} | {event_type} | user={username} | ip={ip}\n"
    with open(AUTH_LOG, 'a') as f:
        f.write(line)


def log_access_event(method, path, status_code, response_size, ip):
    """Log HTTP access events to access.log"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    line = f"{timestamp} | {method} | {path} | {status_code} | {response_size} bytes | ip={ip}\n"
    with open(ACCESS_LOG, 'a') as f:
        f.write(line)


_cpu_lock = threading.Lock()
_last_cpu_times = [psutil.cpu_times()]


def cpu_since_last_snapshot():
    """Host CPU % averaged since the previous snapshot (shared across request threads)."""
    with _cpu_lock:
        now = psutil.cpu_times()
        prev, _last_cpu_times[0] = _last_cpu_times[0], now
    busy = lambda t: sum(t) - t.idle - getattr(t, 'iowait', 0)
    total = sum(now) - sum(prev)
    if total <= 0:
        return 0.0
    return round(min(max((busy(now) - busy(prev)) / total * 100, 0.0), 100.0), 1)


def log_system_event():
    """Log system metrics to system.log"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    cpu = cpu_since_last_snapshot()  # average since the previous snapshot — no spikes, no blocking
    mem = psutil.virtual_memory().percent
    procs = len(psutil.pids())
    line = f"{timestamp} | cpu={cpu} | mem={mem} | procs={procs} | new_procs=0\n"
    with open(SYSTEM_LOG, 'a') as f:
        f.write(line)


@app.after_request
def after_request_logging(response):
    """Log every HTTP request and system state after each request."""
    ip = request.remote_addr or '127.0.0.1'
    if request.path.startswith(INTERNAL_PREFIXES):
        # Console traffic is not user activity — keep it out of the access log,
        # but still record a host snapshot (system telemetry heartbeat).
        log_system_event()
        return response
    log_access_event(
        method=request.method,
        path=request.path,
        status_code=response.status_code,
        response_size=response.content_length or 0,
        ip=ip
    )
    log_system_event()
    return response


# ══════════════════════════════════════
# ROUTES (Phase 1)
# ══════════════════════════════════════

@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        ip = request.remote_addr or '127.0.0.1'

        if username in DEMO_USERS and DEMO_USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            log_auth_event('LOGIN_SUCCESS', username, ip)
            return redirect(url_for('dashboard'))
        else:
            log_auth_event('LOGIN_FAILED', username, ip)
            error = 'Invalid credentials. Try again.'

    return render_template('login.html', error=error)


@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session.get('username'))


@app.route('/admin')
def admin():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('admin.html', username=session.get('username'))


@app.route('/api/data')
def api_data():
    """API endpoint returning system info as JSON."""
    if not session.get('logged_in'):
        return jsonify({'error': 'unauthorized'}), 401
    return jsonify({
        'status': 'ok',
        'user': session.get('username'),
        'timestamp': datetime.now().isoformat(),
        'data': {
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent,
            'processes': len(psutil.pids())
        }
    })


@app.route('/logout')
def logout():
    username = session.get('username', 'unknown')
    ip = request.remote_addr or '127.0.0.1'
    log_auth_event('LOGOUT', username, ip)
    session.clear()
    return redirect(url_for('login'))


# ══════════════════════════════════════
# THREAT API (Phase 8 — used by dashboard)
# ══════════════════════════════════════

@app.route('/api/threat')
def api_threat():
    """Returns current threat assessment from the detector."""
    try:
        from app.detector import get_threat_status
        result = get_threat_status()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'threat_score': 0,
            'label': 'BENIGN',
            'auth_score': 0,
            'network_score': 0,
            'system_score': 0,
            'error': str(e)
        })


# ══════════════════════════════════════
# HOST CALIBRATION
# ══════════════════════════════════════

def calibrate_host():
    """Record this machine's idle memory / process count as its baseline."""
    detector.set_calibration(psutil.virtual_memory().percent, len(psutil.pids()))
    return detector.get_calibration()


calibrate_host()


@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    """Re-measure the idle baseline of this machine."""
    return jsonify(calibrate_host())


@app.route('/api/whatif', methods=['POST'])
def api_whatif():
    """Score a hand-made feature vector (model units) — used by the dashboard's What-If Lab."""
    x = request.get_json(force=True)
    return jsonify(detector.score_scenario(x))


@app.route('/api/model-info')
def api_model_info():
    """Feature importances of the deployed model — used by the dashboard's Research tab."""
    model = detector.load_model()
    if model is None:
        return jsonify({'feature_importances': {}})
    return jsonify({'feature_importances': {c: float(v) for c, v in zip(FEATURE_COLUMNS, model.feature_importances_)}})


# ══════════════════════════════════════
# CLEAR LOGS ENDPOINT (for demo resets)
# ══════════════════════════════════════

@app.route('/api/clear-logs', methods=['POST'])
def clear_logs():
    """Clear all log files for a fresh demo."""
    for log_file in [AUTH_LOG, ACCESS_LOG, SYSTEM_LOG]:
        if os.path.exists(log_file):
            open(log_file, 'w').close()
    return jsonify({'status': 'logs cleared'})


if __name__ == '__main__':
    print("=" * 50)
    print("  MULTI-MODAL THREAT DETECTION — LOGIN APP")
    print("  http://127.0.0.1:5000")
    print("=" * 50)
    print(f"  Demo users: {', '.join(DEMO_USERS.keys())}")
    print("=" * 50)
    app.run(debug=True, port=5000)
