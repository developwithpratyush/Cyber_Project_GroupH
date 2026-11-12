"""
Feature Extractor — Phase 3
Converts raw log files into numerical feature vectors for the RF model.

Extracts 15 features across 3 modalities:
- Authentication (5): failed_login_count, successful_login, failure_rate, time_between_attempts, login_frequency
- Network (6): request_count, requests_per_second, unique_endpoints, bytes_sent, bytes_received, error_count
- System (4): cpu_usage, memory_usage, process_count, new_process_count
"""

import os
import re
from datetime import datetime, timedelta

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')

# Requests made by the monitoring console itself are not user/attacker traffic
# and must never be counted as network telemetry.
INTERNAL_PREFIXES = ('/static', '/api/threat', '/api/calibrate', '/api/clear-logs', '/api/whatif',
                     '/api/model-info', '/favicon.ico')

# System telemetry is sampled every second; a host anomaly is judged on the
# peak of a short recent window rather than on whichever line happens to be last.
SYSTEM_WINDOW_SECONDS = 12
MIN_RATE_SPAN_SECONDS = 10.0


def parse_timestamp(ts_str):
    """Parse log timestamp string to datetime."""
    try:
        return datetime.strptime(ts_str.strip(), '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        return None


def read_log_lines(log_file, window_seconds=60):
    """Read log lines within the last N seconds."""
    if not os.path.exists(log_file):
        return []

    cutoff = datetime.now() - timedelta(seconds=window_seconds)
    lines = []

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Extract timestamp (first 23 chars: "2024-01-01 10:00:01.123")
            ts_str = line[:23]
            ts = parse_timestamp(ts_str)
            if ts and ts >= cutoff:
                lines.append((ts, line))

    return lines


def extract_auth_features(window_seconds=60):
    """Extract 5 authentication features from auth.log."""
    auth_log = os.path.join(LOG_DIR, 'auth.log')
    lines = read_log_lines(auth_log, window_seconds)

    failed_count = 0
    success_count = 0
    timestamps = []

    for ts, line in lines:
        timestamps.append(ts)
        if 'LOGIN_FAILED' in line:
            failed_count += 1
        elif 'LOGIN_SUCCESS' in line:
            success_count += 1

    total = failed_count + success_count
    failure_rate = failed_count / total if total > 0 else 0.0
    successful_login = 1 if success_count > 0 else 0

    # Time between attempts (average seconds)
    time_between = 0.0
    if len(timestamps) > 1:
        diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
        time_between = sum(diffs) / len(diffs) if diffs else 0.0

    # Login frequency (attempts per minute)
    login_frequency = (total / window_seconds) * 60 if window_seconds > 0 else 0.0

    return {
        'failed_login_count': failed_count,
        'successful_login': successful_login,
        'failure_rate': round(failure_rate, 4),
        'time_between_attempts': round(time_between, 4),
        'login_frequency': round(login_frequency, 4)
    }


def extract_network_features(window_seconds=60):
    """Extract 6 network/application features from access.log."""
    access_log = os.path.join(LOG_DIR, 'access.log')
    lines = read_log_lines(access_log, window_seconds)

    endpoints = set()
    bytes_sent = 0
    bytes_received = 0
    error_count = 0

    lines = [(ts, line) for ts, line in lines if not _is_internal(line)]
    request_count = len(lines)

    for ts, line in lines:
        parts = line.split(' | ')
        if len(parts) >= 4:
            # Extract endpoint path
            path = parts[2].strip() if len(parts) > 2 else ''
            endpoints.add(path)

            # Extract status code
            try:
                status = int(parts[3].strip())
                if status >= 400:
                    error_count += 1
            except (ValueError, IndexError):
                pass

            # Extract bytes
            if len(parts) > 4:
                bytes_match = re.search(r'(\d+)\s*bytes', parts[4])
                if bytes_match:
                    bytes_received += int(bytes_match.group(1))

        # Estimate bytes_sent (request payload — simplified)
        bytes_sent += 200  # Average request header size

    # Requests per second
    # Rate over at least 10 s, so a login + redirect pair is not read as a burst
    if lines:
        time_span = (lines[-1][0] - lines[0][0]).total_seconds()
        requests_per_second = request_count / max(time_span, MIN_RATE_SPAN_SECONDS)
    else:
        requests_per_second = 0.0

    return {
        'request_count': request_count,
        'requests_per_second': round(requests_per_second, 4),
        'unique_endpoints': len(endpoints),
        'bytes_sent': bytes_sent,
        'bytes_received': bytes_received,
        'error_count': error_count
    }


def extract_system_features(window_seconds=SYSTEM_WINDOW_SECONDS):
    """Extract 4 system features from system.log (peak over the recent window)."""
    system_log = os.path.join(LOG_DIR, 'system.log')
    lines = read_log_lines(system_log, window_seconds)
    if not lines:
        # Quiet host: no snapshot in the window — use the most recent one instead of zeros
        lines = read_log_lines(system_log, 3600)[-1:]

    cpu_usage = 0.0
    memory_usage = 0.0
    process_count = 0
    new_process_count = 0

    for _, line in lines:
        cpu_match = re.search(r'cpu=([\d.]+)', line)
        mem_match = re.search(r'mem=([\d.]+)', line)
        proc_match = re.search(r'procs=(\d+)', line)
        new_proc_match = re.search(r'new_procs=(\d+)', line)

        if cpu_match:
            cpu_usage = max(cpu_usage, float(cpu_match.group(1)))
        if mem_match:
            memory_usage = max(memory_usage, float(mem_match.group(1)))
        if proc_match:
            process_count = max(process_count, int(proc_match.group(1)))
        if new_proc_match:
            new_process_count = max(new_process_count, int(new_proc_match.group(1)))

    return {
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'process_count': process_count,
        'new_process_count': new_process_count
    }


def _is_internal(line):
    parts = line.split(' | ')
    return len(parts) > 2 and parts[2].strip().startswith(INTERNAL_PREFIXES)


def extract_all_features(window_seconds=60):
    """Extract all 15 features across 3 modalities. Returns a flat dict."""
    features = {}
    features.update(extract_auth_features(window_seconds))
    features.update(extract_network_features(window_seconds))
    features.update(extract_system_features(min(window_seconds, SYSTEM_WINDOW_SECONDS)))
    return features


# Feature column names in order (must match model training)
FEATURE_COLUMNS = [
    'failed_login_count', 'successful_login', 'failure_rate',
    'time_between_attempts', 'login_frequency',
    'request_count', 'requests_per_second', 'unique_endpoints',
    'bytes_sent', 'bytes_received', 'error_count',
    'cpu_usage', 'memory_usage', 'process_count', 'new_process_count'
]

AUTH_FEATURES = ['failed_login_count', 'successful_login', 'failure_rate',
                 'time_between_attempts', 'login_frequency']

NETWORK_FEATURES = ['request_count', 'requests_per_second', 'unique_endpoints',
                    'bytes_sent', 'bytes_received', 'error_count']

SYSTEM_FEATURES = ['cpu_usage', 'memory_usage', 'process_count', 'new_process_count']


if __name__ == '__main__':
    # Quick test
    features = extract_all_features(window_seconds=300)
    print("Extracted Features:")
    for k, v in features.items():
        print(f"  {k}: {v}")
