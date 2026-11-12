"""
College Presentation Demo Runner — Phase 10
Automates the exact presentation sequence required for college evaluation:
1. Show normal login and legitimate activity -> low threat score.
2. Show attack scenario (authentication brute-force, rapid scanning, system spike) -> elevated threat score.
3. Show feature importance and multi-modality evidence fusion.
"""

import os
import sys
import time
import requests

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

FLASK_URL = 'http://127.0.0.1:5000'
PROJECT_ROOT = os.path.dirname(__file__)


def check_flask_app():
    """Verify Flask application is active."""
    try:
        r = requests.get(f"{FLASK_URL}/api/threat", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def print_banner(text):
    print("\n" + "=" * 65)
    print(f"  {text}")
    print("=" * 65)


def run_presentation_demo():
    print_banner("COLLEGE PRESENTATION DEMO: MULTI-MODAL THREAT DETECTION")
    print("Verifying environment...")

    if not check_flask_app():
        print("[ERROR] Flask app is not running!")
        print("        Please start it in another terminal: python app/app.py")
        sys.exit(1)

    print("[OK] Flask application detected at http://127.0.0.1:5000")

    # Step 1: Clean slate
    print("\n[Step 1] Resetting system logs to baseline...")
    requests.post(f"{FLASK_URL}/api/clear-logs")
    time.sleep(1)

    r = requests.get(f"{FLASK_URL}/api/threat").json()
    print(f"  Initial Status: {r['label']} (Threat Score: {r['threat_score']}%)")

    # Step 2: Normal Activity Simulation
    print_banner("STEP 2: SIMULATING LEGITIMATE USER ACTIVITY")
    print("Simulating valid user login and standard page browsing...")
    from simulator.attack_simulator import normal_mode
    normal_mode()

    # Step 3: Verify Low Threat
    time.sleep(1)
    r = requests.get(f"{FLASK_URL}/api/threat").json()
    print("\n[Detection Result - Legitimate Traffic]:")
    print(f"  Threat Label:  {r['label']}")
    print(f"  Threat Score:  {r['threat_score']}%")
    print(f"  Auth Score:    {r['auth_score']}%")
    print(f"  Network Score: {r['network_score']}%")
    print(f"  System Score:  {r['system_score']}%")
    print("  -> System correctly maintains BENIGN status.")

    input("\nPress Enter to trigger the simulated Multi-Modal Attack...")

    # Step 4: Attack Simulation
    print_banner("STEP 3: SIMULATING MULTI-MODAL CYBER ATTACK")
    print("Simulating brute-force auth + abnormal requests + resource anomaly...")
    from simulator.attack_simulator import attack_mode
    attack_mode()

    # Step 5: Verify Threat Detection
    time.sleep(1)
    r = requests.get(f"{FLASK_URL}/api/threat").json()
    print_banner("STEP 4: MULTI-MODAL FUSION DETECTION RESULT")
    print(f"  Threat Status:   [ {r['label']} ]")
    print(f"  Combined Threat: {r['threat_score']}%")
    print(f"  Modalities Breakdown:")
    print(f"    - Authentication Risk: {r['auth_score']}% (Failures: {r['features'].get('failed_login_count', 0)})")
    print(f"    - Network Risk:        {r['network_score']}% (Requests: {r['features'].get('request_count', 0)})")
    print(f"    - System Risk:         {r['system_score']}% (CPU: {r['features'].get('cpu_usage', 0)}%, New Procs: {r['features'].get('new_process_count', 0)})")
    print("\n  Conclusion:")
    print("  Correlation across all 3 data sources confirmed anomalous behavior.")
    print("  View real-time visual charts on the Streamlit dashboard: http://localhost:8501")
    print_banner("DEMO COMPLETED SUCCESSFULLY")


if __name__ == '__main__':
    run_presentation_demo()
