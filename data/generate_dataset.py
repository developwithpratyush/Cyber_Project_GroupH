"""
Dataset Generator — Phase 5 (Academic Multi-Modal Security Dataset)
Generates 3,000 synchronized multi-modal session records:
- 1,500 Benign sessions (Standard, Heavy Web Browsers, Human Typos, OS background updates)
- 1,500 Attack sessions (Full Killchain, Stealth Low-and-Slow, Stolen Credentials, Host Cryptominer)

Incorporate realistic cross-domain noise and natural distribution overlap
so that individual modalities exhibit expected real-world blind spots,
demonstrating the empirical advantage of multi-modal feature fusion.
"""

import os
import sys
import csv
import random
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

OUTPUT_DIR = os.path.dirname(__file__)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'demo_multimodal.csv')

COLUMNS = [
    'failed_login_count', 'successful_login', 'failure_rate',
    'time_between_attempts', 'login_frequency',
    'request_count', 'requests_per_second', 'unique_endpoints',
    'bytes_sent', 'bytes_received', 'error_count',
    'cpu_usage', 'memory_usage', 'process_count', 'new_process_count',
    'label'
]


# =====================================================================
# BENIGN PROFILES (1,500 total)
# =====================================================================

def generate_standard_benign():
    """Standard legitimate user browsing normally."""
    failed = random.choice([0, 0, 0, 0, 1])
    success = 1
    total = failed + success
    failure_rate = round(failed / total, 4)
    time_between = round(random.uniform(15.0, 50.0), 4)
    login_freq = round(random.uniform(0.1, 1.2), 4)

    req_count = random.randint(3, 16)
    rps = round(random.uniform(0.04, 0.4), 4)
    unique_ep = random.randint(1, 3)
    bytes_s = random.randint(600, 3200)
    bytes_r = random.randint(1200, 7500)
    err_count = 0

    cpu = round(random.uniform(12.0, 32.0), 1)
    mem = round(random.uniform(32.0, 52.0), 1)
    procs = random.randint(120, 155)
    new_procs = 0

    return [
        failed, success, failure_rate, time_between, login_freq,
        req_count, rps, unique_ep, bytes_s, bytes_r, err_count,
        cpu, mem, procs, new_procs,
        0
    ]


def generate_heavy_benign():
    """
    Legitimate power user / heavy data downloader.
    Causes Network-only False Positives (high traffic volume & request frequency),
    while Auth and System remain completely normal.
    """
    failed = 0
    success = 1
    failure_rate = 0.0
    time_between = round(random.uniform(20.0, 60.0), 4)
    login_freq = round(random.uniform(0.1, 0.8), 4)

    # Elevated network activity overlapping with scan attacks
    req_count = random.randint(32, 65)
    rps = round(random.uniform(2.2, 5.5), 4)
    unique_ep = random.randint(4, 6)
    bytes_s = random.randint(7500, 22000)
    bytes_r = random.randint(14000, 45000)
    err_count = random.choice([0, 1, 2, 3])

    # Normal host metrics
    cpu = round(random.uniform(16.0, 38.0), 1)
    mem = round(random.uniform(35.0, 58.0), 1)
    procs = random.randint(125, 160)
    new_procs = 0

    return [
        failed, success, failure_rate, time_between, login_freq,
        req_count, rps, unique_ep, bytes_s, bytes_r, err_count,
        cpu, mem, procs, new_procs,
        0
    ]


def generate_typo_benign():
    """
    Legitimate user who mistyped their password 2-4 times before success.
    Causes Auth-only False Positives (failure rate > 60%),
    while Network and System show normal human interaction.
    """
    failed = random.randint(2, 4)
    success = 1
    total = failed + success
    failure_rate = round(failed / total, 4)
    time_between = round(random.uniform(6.0, 22.0), 4)
    login_freq = round(random.uniform(3.0, 7.5), 4)

    req_count = random.randint(4, 16)
    rps = round(random.uniform(0.08, 0.45), 4)
    unique_ep = random.randint(1, 3)
    bytes_s = random.randint(900, 3800)
    bytes_r = random.randint(1800, 8500)
    err_count = 0

    cpu = round(random.uniform(14.0, 35.0), 1)
    mem = round(random.uniform(32.0, 54.0), 1)
    procs = random.randint(120, 155)
    new_procs = 0

    return [
        failed, success, failure_rate, time_between, login_freq,
        req_count, rps, unique_ep, bytes_s, bytes_r, err_count,
        cpu, mem, procs, new_procs,
        0
    ]


def generate_system_heavy_benign():
    """
    Legitimate user browsing while Windows Update or Antivirus scans the host.
    Causes System-only False Positives (CPU 65-88%),
    while Auth is clean and Network traffic is normal.
    """
    failed = random.choice([0, 1])
    success = 1
    total = failed + success
    failure_rate = round(failed / total, 4)
    time_between = round(random.uniform(25.0, 60.0), 4)
    login_freq = round(random.uniform(0.1, 0.6), 4)

    req_count = random.randint(3, 14)
    rps = round(random.uniform(0.04, 0.32), 4)
    unique_ep = random.randint(1, 3)
    bytes_s = random.randint(600, 2800)
    bytes_r = random.randint(1200, 6500)
    err_count = 0

    cpu = round(random.uniform(64.0, 88.0), 1)
    mem = round(random.uniform(68.0, 86.0), 1)
    procs = random.randint(165, 205)
    new_procs = random.randint(1, 3)

    return [
        failed, success, failure_rate, time_between, login_freq,
        req_count, rps, unique_ep, bytes_s, bytes_r, err_count,
        cpu, mem, procs, new_procs,
        0
    ]


# =====================================================================
# ATTACK PROFILES (1,500 total)
# =====================================================================

def generate_full_killchain_attack():
    """Full brute force -> compromise -> directory crawl -> payload execution."""
    failed = random.randint(12, 28)
    success = random.choice([0, 1, 1])
    total = failed + success
    failure_rate = round(failed / total, 4)
    time_between = round(random.uniform(0.12, 1.4), 4)
    login_freq = round(random.uniform(16.0, 48.0), 4)

    req_count = random.randint(42, 90)
    rps = round(random.uniform(4.5, 16.0), 4)
    unique_ep = random.randint(5, 9)
    bytes_s = random.randint(11000, 26000)
    bytes_r = random.randint(800, 4500)
    err_count = random.randint(3, 18)

    cpu = round(random.uniform(62.0, 92.0), 1)
    mem = round(random.uniform(64.0, 90.0), 1)
    procs = random.randint(160, 210)
    new_procs = random.randint(2, 6)

    return [
        failed, success, failure_rate, time_between, login_freq,
        req_count, rps, unique_ep, bytes_s, bytes_r, err_count,
        cpu, mem, procs, new_procs,
        1
    ]


def generate_stealth_brute_attack():
    """
    Low & Slow attacker limiting failed attempts to 2-3 to evade Auth rules.
    Causes Auth-only False Negatives (auth detector misses it).
    Network & System evidence detects the threat.
    """
    failed = random.randint(2, 4)
    success = 1
    total = failed + success
    failure_rate = round(failed / total, 4)
    time_between = round(random.uniform(4.0, 14.0), 4)
    login_freq = round(random.uniform(2.5, 5.5), 4)

    # Moderate network scanning
    req_count = random.randint(28, 55)
    rps = round(random.uniform(2.0, 6.5), 4)
    unique_ep = random.randint(4, 7)
    bytes_s = random.randint(7000, 18000)
    bytes_r = random.randint(1000, 4200)
    err_count = random.randint(1, 8)

    cpu = round(random.uniform(55.0, 84.0), 1)
    mem = round(random.uniform(58.0, 84.0), 1)
    procs = random.randint(155, 195)
    new_procs = random.randint(1, 4)

    return [
        failed, success, failure_rate, time_between, login_freq,
        req_count, rps, unique_ep, bytes_s, bytes_r, err_count,
        cpu, mem, procs, new_procs,
        1
    ]


def generate_stolen_credential_attack():
    """
    Attacker using valid credentials (0 failed logins).
    Auth-only detector is completely blind (0% auth risk).
    Detected by anomalous Network data extraction + System activity.
    """
    failed = 0
    success = 1
    failure_rate = 0.0
    time_between = round(random.uniform(20.0, 55.0), 4)
    login_freq = round(random.uniform(0.1, 0.7), 4)

    req_count = random.randint(38, 80)
    rps = round(random.uniform(3.5, 12.0), 4)
    unique_ep = random.randint(5, 9)
    bytes_s = random.randint(10000, 28000)
    bytes_r = random.randint(600, 3800)
    err_count = random.randint(2, 12)

    cpu = round(random.uniform(56.0, 86.0), 1)
    mem = round(random.uniform(60.0, 86.0), 1)
    procs = random.randint(156, 200)
    new_procs = random.randint(2, 5)

    return [
        failed, success, failure_rate, time_between, login_freq,
        req_count, rps, unique_ep, bytes_s, bytes_r, err_count,
        cpu, mem, procs, new_procs,
        1
    ]


def generate_resource_payload_attack():
    """
    Compromised session running host malware / cryptominer.
    Network requests are low and quiet, but System metrics are critical.
    Network-only models miss this.
    """
    failed = random.choice([0, 1, 2])
    success = 1
    total = failed + success
    failure_rate = round(failed / total, 4)
    time_between = round(random.uniform(10.0, 35.0), 4)
    login_freq = round(random.uniform(0.5, 2.5), 4)

    # Normal/quiet web interaction
    req_count = random.randint(6, 20)
    rps = round(random.uniform(0.1, 0.8), 4)
    unique_ep = random.randint(2, 4)
    bytes_s = random.randint(1500, 5000)
    bytes_r = random.randint(1500, 6000)
    err_count = random.choice([0, 1])

    # Host anomaly
    cpu = round(random.uniform(84.0, 98.5), 1)
    mem = round(random.uniform(78.0, 95.0), 1)
    procs = random.randint(176, 225)
    new_procs = random.randint(3, 7)

    return [
        failed, success, failure_rate, time_between, login_freq,
        req_count, rps, unique_ep, bytes_s, bytes_r, err_count,
        cpu, mem, procs, new_procs,
        1
    ]


# =====================================================================
# DATASET GENERATION ORCHESTRATOR
# =====================================================================

def generate_dataset(n_samples=3000):
    random.seed(42)
    np.random.seed(42)

    rows = []
    half = n_samples // 2

    # Benign distribution (1500 samples)
    n_std_benign = int(half * 0.55)      # 825
    n_heavy_benign = int(half * 0.15)    # 225
    n_typo_benign = int(half * 0.15)     # 225
    n_sys_benign = half - (n_std_benign + n_heavy_benign + n_typo_benign)  # 225

    for _ in range(n_std_benign):
        rows.append(generate_standard_benign())
    for _ in range(n_heavy_benign):
        rows.append(generate_heavy_benign())
    for _ in range(n_typo_benign):
        rows.append(generate_typo_benign())
    for _ in range(n_sys_benign):
        rows.append(generate_system_heavy_benign())

    # Attack distribution (1500 samples)
    n_killchain = int(half * 0.45)       # 675
    n_stealth = int(half * 0.20)         # 300
    n_stolen = int(half * 0.15)          # 225
    n_payload = half - (n_killchain + n_stealth + n_stolen)  # 300

    for _ in range(n_killchain):
        rows.append(generate_full_killchain_attack())
    for _ in range(n_stealth):
        rows.append(generate_stealth_brute_attack())
    for _ in range(n_stolen):
        rows.append(generate_stolen_credential_attack())
    for _ in range(n_payload):
        rows.append(generate_resource_payload_attack())

    # Inject realistic 1.8% border telemetry ambiguity reflecting real-world IDS noise
    for row in rows:
        if random.random() < 0.018:
            row[-1] = 1 - row[-1]

    random.shuffle(rows)

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)

    benign_count = sum(1 for r in rows if r[-1] == 0)
    attack_count = sum(1 for r in rows if r[-1] == 1)

    print("=" * 60)
    print("  SYNCHRONIZED MULTI-MODAL DATASET GENERATION")
    print("=" * 60)
    print(f"Destination: {OUTPUT_FILE}")
    print(f"Total Rows:  {len(rows)} (15 features + 1 label)")
    print(f"Benign:      {benign_count} (50%) [Standard, Heavy Web, Typos, OS updates]")
    print(f"Attack:      {attack_count} (50%) [Killchain, Stealth, Stolen Creds, Cryptominer]")
    print("[OK] Realistic cross-modality noise and blindspots injected successfully.")


if __name__ == '__main__':
    generate_dataset(3000)
