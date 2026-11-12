"""
Attack Simulator — Phase 9 (Dynamic & Randomized)
Controlled, safe, local-only attack simulator targeting localhost:5000 ONLY.

Features dynamic randomization on every execution:
- Random password attempts from realistic dictionary
- Randomized timing delays (jitter)
- Randomized endpoint scan paths and query parameters
- Randomized system resource spikes
- Multiple attack profiles:
    --mode attack      : Full kill-chain (brute force -> compromise -> scan -> system anomaly)
    --mode brute_force : Authentication-heavy attack
    --mode scanner     : Rapid network endpoint enumeration
    --mode normal      : Realistic human user browsing
"""

import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import random
import requests
import psutil

BASE_URL = 'http://127.0.0.1:5000'
SESSION = requests.Session()

# Expanded dictionary of common credential attacks
PASSWORD_POOL = [
    'password', '123456', 'admin', 'letmein', 'welcome',
    'monkey', 'dragon', 'master', 'qwerty', 'abc123',
    'password1', 'iloveyou', 'trustno1', 'sunshine', 'princess',
    'football', 'charlie', 'shadow', 'michael', 'login',
    'admin1234', 'root2024', 'pass123', 'toor', 'supersecret',
    'access123', 'system@1', 'company2024', 'changeme', 'guest'
]

USERNAME_POOL = ['admin', 'root', 'administrator', 'user1', 'service_account']

SCAN_ENDPOINTS = [
    '/dashboard', '/admin', '/api/data',
    '/dashboard', '/api/data', '/admin',
    '/api/data', '/dashboard', '/admin'
]


def normal_mode():
    """Simulate realistic legitimate user behavior with randomized human delays."""
    print("\n" + "=" * 55)
    print("  NORMAL MODE — Simulating Legitimate User Session")
    print("=" * 55)

    # 1. Login phase (occasionally 1 typo, like a real human)
    has_typo = random.random() < 0.4
    if has_typo:
        typo_pwd = random.choice(['admin12', 'Admin123', 'admin_wrong'])
        print(f"\n[1] User typing password (minor typo: '{typo_pwd}')...")
        r = SESSION.post(f'{BASE_URL}/login', data={'username': 'admin', 'password': typo_pwd})
        print(f"    Login attempt: HTTP {r.status_code} (Expected failure)")
        time.sleep(random.uniform(1.5, 3.5))

    print("\n[1] User authenticating with correct credentials...")
    r = SESSION.post(f'{BASE_URL}/login', data={'username': 'admin', 'password': 'admin123'})
    print(f"    Login attempt: HTTP {r.status_code} (Success)")
    time.sleep(random.uniform(1.0, 2.5))

    # 2. Browsing pages with human reading delays
    pages = random.sample(['/dashboard', '/admin', '/api/data', '/dashboard'], k=4)
    print(f"\n[2] Human browsing session ({len(pages)} pages)...")
    for i, page in enumerate(pages, 1):
        delay = round(random.uniform(1.8, 3.8), 2)
        r = SESSION.get(f'{BASE_URL}{page}')
        print(f"    [{i}/{len(pages)}] GET {page} -> HTTP {r.status_code} (User reading for {delay}s)")
        time.sleep(delay)

    # 3. Logout
    print("\n[3] User logging out cleanly...")
    SESSION.get(f'{BASE_URL}/logout')
    print("[OK] Normal session completed cleanly. Threat score remains low.")


def attack_mode(profile='killchain'):
    """Simulate realistic, dynamic attack with randomized entropy."""
    print("\n" + "=" * 55)
    print(f"  [!] ATTACK MODE — Dynamic Attack Profile: [{profile.upper()}]")
    print("  Target: localhost:5000 ONLY (Safe demonstration)")
    print("=" * 55)

    # Phase 1: Authentication Brute Force
    n_attempts = random.randint(12, 16)
    sampled_passwords = random.sample(PASSWORD_POOL, min(n_attempts, len(PASSWORD_POOL)))
    print(f"\n[Phase 1] Launching automated brute-force ({n_attempts} randomized dictionary attempts)...")

    for i, pwd in enumerate(sampled_passwords, 1):
        target_user = random.choice(['admin', 'admin', 'admin', 'root'])
        delay = random.uniform(0.05, 0.15)
        r = SESSION.post(f'{BASE_URL}/login', data={'username': target_user, 'password': pwd})
        print(f"    Attempt {i:02d}/{n_attempts:02d}: user='{target_user}' pwd='{pwd}' -> FAILED (HTTP {r.status_code})")
        time.sleep(delay)

    # Phase 2: Credential Compromise
    print("\n[Phase 2] Attacker discovers valid credential 'admin123'...")
    time.sleep(0.3)
    r = SESSION.post(f'{BASE_URL}/login', data={'username': 'admin', 'password': 'admin123'})
    print(f"    Target Authenticated: SUCCESS (HTTP {r.status_code}) — Session Compromised!")
    time.sleep(0.2)

    if profile in ('killchain', 'scanner'):
        # Phase 3: Rapid Endpoint Scanning & Enumeration
        n_scans = random.randint(20, 28)
        print(f"\n[Phase 3] Rapid endpoint scanning ({n_scans} requests across protected endpoints)...")

        for i in range(1, n_scans + 1):
            target = random.choice(SCAN_ENDPOINTS)
            query_param = f"?scan_id={random.randint(100, 999)}&query=probe_{random.randint(1, 10)}"
            full_path = f"{target}{query_param if random.random() < 0.5 else ''}"
            delay = random.uniform(0.02, 0.08)
            r = SESSION.get(f'{BASE_URL}{full_path}')
            print(f"    Scan [{i:02d}/{n_scans:02d}]: GET {target} -> HTTP {r.status_code}")
            time.sleep(delay)

    if profile in ('killchain', 'system'):
        # Phase 4: Host System Anomaly (Controlled calculation spike)
        print("\n[Phase 4] Simulating host system anomaly (controlled CPU & process spike)...")
        _simulate_system_anomaly()

    print("\n" + "=" * 55)
    print("  [ALERT] Dynamic attack simulation completed!")
    print("  View real-time anomaly detection on Streamlit Dashboard: http://localhost:8501")
    print("=" * 55)


def _simulate_system_anomaly():
    """Trigger controlled, harmless CPU activity and log host anomaly event."""
    duration = random.uniform(1.5, 2.5)
    print(f"    Starting controlled CPU spike ({duration:.1f}s)...")
    start = time.time()
    count = 0
    while time.time() - start < duration:
        count += sum(i * i for i in range(10000))

    # Log metrics to system.log
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    system_log = os.path.join(log_dir, 'system.log')
    os.makedirs(log_dir, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    real_cpu = psutil.cpu_percent(interval=0.3)
    simulated_cpu = max(real_cpu, round(random.uniform(65.0, 88.0), 1))
    mem = psutil.virtual_memory().percent
    procs = len(psutil.pids())
    new_procs = random.randint(2, 5)

    with open(system_log, 'a') as f:
        f.write(f"{timestamp} | cpu={simulated_cpu} | mem={mem} | procs={procs} | new_procs={new_procs}\n")
    print(f"    [OK] System metrics logged: cpu={simulated_cpu}%, new_procs={new_procs}")


def clear_logs():
    """Clear logs for clean demo restart."""
    try:
        SESSION.post(f'{BASE_URL}/api/clear-logs', timeout=2)
        print("[OK] Logs cleared via Flask API.")
    except Exception:
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        for f in ['auth.log', 'access.log', 'system.log']:
            path = os.path.join(log_dir, f)
            if os.path.exists(path):
                open(path, 'w').close()
        print("[OK] Logs cleared locally.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dynamic Cyber Threat Attack Simulator')
    parser.add_argument('--mode', choices=['normal', 'attack'], default='attack',
                        help='Simulation mode: normal or attack')
    parser.add_argument('--profile', choices=['killchain', 'scanner', 'brute_force'], default='killchain',
                        help='Attack profile type')
    parser.add_argument('--clear', action='store_true',
                        help='Clear logs before running')
    args = parser.parse_args()

    if args.clear:
        clear_logs()

    if args.mode == 'normal':
        normal_mode()
    else:
        attack_mode(profile=args.profile)
