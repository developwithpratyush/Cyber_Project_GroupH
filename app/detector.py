"""
Detector — Phase 8
Loads the trained multimodal RF model and classifies live telemetry.
Training happens offline. This module only does inference.

Live additions (no retraining):
- The model is loaded once and kept in memory.
- Host calibration: absolute memory / process counts of the demo laptop are
  re-expressed relative to its own idle baseline, so they land in the same
  range the model was trained on.
- Three-tier verdict: BENIGN / SUSPICIOUS (a single source looks hostile) /
  ATTACK (the fused model confirms).
- Blind-spot replay: the same live moment re-scored with modalities masked
  to a benign baseline, showing what single-source detection would conclude.
"""

import os
import threading

import joblib
import pandas as pd

# Import feature extractor
from app.feature_extractor import extract_all_features, FEATURE_COLUMNS
from app.feature_extractor import AUTH_FEATURES, NETWORK_FEATURES, SYSTEM_FEATURES

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'multimodal_model.pkl')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'demo_multimodal.csv')

ATTACK_THRESHOLD = 0.5       # model decision boundary (predict())
SUSPICIOUS_PROBABILITY = 0.4
SUSPICIOUS_INDICATOR = 50.0  # any single modality indicator above this

# Typical idle host in the training data; the live host is mapped onto it.
TRAINING_IDLE_MEMORY = 42.0
TRAINING_IDLE_PROCESSES = 140

MODALITIES = {
    'auth': AUTH_FEATURES,
    'network': NETWORK_FEATURES,
    'system': SYSTEM_FEATURES,
}

_model = None
_benign_reference = None
_lock = threading.Lock()
_calibration = {'memory': None, 'processes': None}


def load_model():
    """Load the trained RF model from disk (once)."""
    global _model
    with _lock:
        if _model is None and os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
            # Single-row inference is faster without worker dispatch overhead.
            if hasattr(_model, 'n_jobs'):
                _model.n_jobs = 1
    return _model


def benign_reference():
    """Median benign session from the dataset — the 'nothing to see here' values used for masking."""
    global _benign_reference
    if _benign_reference is None:
        df = pd.read_csv(DATA_PATH)
        _benign_reference = {k: float(v) for k, v in df[df['label'] == 0][FEATURE_COLUMNS].median().items()}
    return _benign_reference


def set_calibration(memory_percent, process_count):
    """Record the idle baseline of the machine running the demo."""
    _calibration['memory'] = float(memory_percent)
    _calibration['processes'] = int(process_count)


def get_calibration():
    return dict(_calibration)


def to_model_space(features):
    """Map raw live features into the value range the model was trained on."""
    x = dict(features)
    # A source with no activity in the window looks like a quiet, normal session
    ref = benign_reference()
    if x.get('failed_login_count', 0) == 0 and not x.get('successful_login', 0):
        x.update({c: ref[c] for c in AUTH_FEATURES})
    if x.get('request_count', 0) == 0:
        x.update({c: ref[c] for c in NETWORK_FEATURES})
    if _calibration['memory'] is not None and x.get('memory_usage', 0) > 0:
        x['memory_usage'] = min(max(TRAINING_IDLE_MEMORY + x['memory_usage'] - _calibration['memory'], 0.0), 100.0)
    if _calibration['processes'] is not None and x.get('process_count', 0) > 0:
        x['process_count'] = max(TRAINING_IDLE_PROCESSES + x['process_count'] - _calibration['processes'], 0)
    return x


def score_vectors(rows):
    """Attack probability for a list of model-space feature dicts."""
    model = load_model()
    frame = pd.DataFrame([[r[c] for c in FEATURE_COLUMNS] for r in rows], columns=FEATURE_COLUMNS)
    return [float(p[1]) for p in model.predict_proba(frame)]


def mask_to(x, keep):
    """Keep the listed modalities; replace the others with benign reference values."""
    ref = benign_reference()
    masked = dict(x)
    for name, cols in MODALITIES.items():
        if name not in keep:
            for c in cols:
                masked[c] = ref[c]
    return masked


BLIND_SPOT_COMBOS = [
    ('auth',), ('network',), ('system',),
    ('auth', 'network'), ('network', 'system'), ('auth', 'system'),
    ('auth', 'network', 'system'),
]


def evaluate(model_x):
    """Fused probability plus the blind-spot replay for one model-space vector."""
    rows = [model_x] + [mask_to(model_x, combo) for combo in BLIND_SPOT_COMBOS[:-1]]
    probs = score_vectors(rows)
    fused = probs[0]
    replay = [{'sources': list(combo), 'probability': round(p, 4), 'detected': p >= ATTACK_THRESHOLD}
              for combo, p in zip(BLIND_SPOT_COMBOS, probs[1:] + [fused])]
    return fused, replay


_explainer = None


def explain(model_x):
    """
    SHAP contributions of each feature to the attack probability of one vector.
    Returns (base_value, {feature: contribution}) or None if SHAP is unavailable.
    """
    global _explainer
    try:
        import shap
    except ImportError:
        return None
    if _explainer is None:
        _explainer = shap.TreeExplainer(load_model())
    frame = pd.DataFrame([[model_x[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    values = _explainer.shap_values(frame)
    base = _explainer.expected_value
    # Attack class: list-per-class (older SHAP) or (rows, features, classes) array
    if isinstance(values, list):
        contrib, base = values[1][0], base[1]
    elif getattr(values, 'ndim', 2) == 3:
        contrib, base = values[0, :, 1], base[1]
    else:
        contrib = values[0]
    return float(base), {c: float(v) for c, v in zip(FEATURE_COLUMNS, contrib)}


def explain_json(model_x):
    """SHAP result as [base, {feature: contribution}] (JSON friendly), or None."""
    try:
        result = explain(model_x)
    except Exception:
        return None
    return [result[0], result[1]] if result else None


def score_scenario(x):
    """Probability, verdict, blind-spot replay and SHAP for a model-space feature dict."""
    x = {c: float(x.get(c, 0)) for c in FEATURE_COLUMNS}
    prob, replay = evaluate(x)
    if prob >= ATTACK_THRESHOLD:
        verdict = 'ATTACK'
    elif prob >= SUSPICIOUS_PROBABILITY:
        verdict = 'SUSPICIOUS'
    else:
        verdict = 'BENIGN'
    return {'probability': round(prob, 4), 'verdict': verdict, 'blind_spot': replay, 'shap': explain_json(x)}


def get_threat_status(window_seconds=60):
    """
    Extract features from live logs, run inference, return threat status.
    Returns a dict with threat score, label, and per-modality scores.
    """
    features = extract_all_features(window_seconds)

    if load_model() is None:
        # No model available — use rule-based fallback
        return rule_based_detection(features)

    model_x = to_model_space(features)
    threat_prob, replay = evaluate(model_x)
    prediction = 1 if threat_prob >= ATTACK_THRESHOLD else 0

    # Per-modality indicator scores (rule-based heuristics, for display)
    auth_score = float(compute_modality_score(model_x, AUTH_FEATURES))
    network_score = float(compute_modality_score(model_x, NETWORK_FEATURES))
    system_score = float(compute_modality_score(model_x, SYSTEM_FEATURES))

    if prediction == 1:
        verdict = 'ATTACK'
    elif threat_prob >= SUSPICIOUS_PROBABILITY or max(auth_score, network_score, system_score) >= SUSPICIOUS_INDICATOR:
        verdict = 'SUSPICIOUS'
    else:
        verdict = 'BENIGN'

    return {
        'threat_score': round(float(threat_prob * 100), 1),
        'label': 'ATTACK' if prediction == 1 else 'BENIGN',
        'verdict': verdict,
        'auth_score': round(auth_score, 1),
        'network_score': round(network_score, 1),
        'system_score': round(system_score, 1),
        'features': features,
        'model_features': model_x,
        'blind_spot': replay,
        'shap': explain_json(model_x),
        'prediction': int(prediction)
    }


def compute_modality_score(features, modality_keys):
    """Simple heuristic score for a modality based on feature values."""
    # Normalize known attack-indicative features to 0-100 scale
    scores = []

    for key in modality_keys:
        val = features.get(key, 0)
        if key == 'failed_login_count':
            scores.append(min(val / 20 * 100, 100))
        elif key == 'failure_rate':
            scores.append(val * 100)
        elif key == 'login_frequency':
            scores.append(min(val / 30 * 100, 100))
        elif key == 'request_count':
            scores.append(min(val / 50 * 100, 100))
        elif key == 'requests_per_second':
            scores.append(min(val / 10 * 100, 100))
        elif key == 'error_count':
            scores.append(min(val / 10 * 100, 100))
        elif key == 'cpu_usage':
            scores.append(min(val / 80 * 100, 100) if val > 30 else val)
        elif key == 'memory_usage':
            scores.append(min(val / 80 * 100, 100) if val > 50 else val)
        elif key == 'new_process_count':
            scores.append(min(val / 3 * 100, 100))

    return sum(scores) / len(scores) if scores else 0.0


def rule_based_detection(features):
    """Fallback rule-based detection when no model is available."""
    auth_score = 0
    if features.get('failed_login_count', 0) > 5:
        auth_score = min(features['failed_login_count'] / 20 * 100, 100)

    network_score = 0
    if features.get('request_count', 0) > 20:
        network_score = min(features['request_count'] / 50 * 100, 100)

    system_score = 0
    if features.get('cpu_usage', 0) > 60:
        system_score = min(features['cpu_usage'] / 80 * 100, 100)

    combined = (auth_score * 0.4 + network_score * 0.35 + system_score * 0.25)
    label = 'ATTACK' if combined > 50 else 'BENIGN'

    return {
        'threat_score': round(combined, 1),
        'label': label,
        'verdict': label,
        'auth_score': round(auth_score, 1),
        'network_score': round(network_score, 1),
        'system_score': round(system_score, 1),
        'features': features,
        'model_features': features,
        'blind_spot': [],
        'prediction': 1 if label == 'ATTACK' else 0
    }
