"""
Centralised model-file path management.

Every training script, agent_trader, and recount.py uses these helpers so
that all model artefacts land in MODELS/<TICKER>_<YYYYMMDD>_<name>.ext .
"""
import os
import glob
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'MODELS')


def ensure_model_dir():
    """Create MODELS/ if it doesn't exist, return its path."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    return MODEL_DIR


def get_prefix(csv_file):
    """Return '<TICKER>_<YYYYMMDD>_' from a CSV filename."""
    ticker = os.path.basename(csv_file).split('_')[0]
    date_str = datetime.now().strftime('%Y%m%d')
    return f'{ticker}_{date_str}_'


def model_path(prefix, name):
    """Full path to a model artefact inside MODELS/."""
    return os.path.join(MODEL_DIR, f'{prefix}{name}')


def save_model(prefix, name, obj):
    """joblib.dump *obj* to MODELS/<prefix><name>."""
    import joblib
    ensure_model_dir()
    path = model_path(prefix, name)
    joblib.dump(obj, path)
    return path


def save_text(prefix, name, lines):
    """Write lines (str or list) to MODELS/<prefix><name>."""
    ensure_model_dir()
    path = model_path(prefix, name)
    with open(path, 'w') as f:
        if isinstance(lines, str):
            f.write(lines)
        else:
            f.write('\n'.join(lines))
    return path


def find_latest_prefix(ticker):
    """
    Return (prefix, date_str) for the newest model set for *ticker*.
    Returns (None, None) if no models found.
    """
    pattern = os.path.join(MODEL_DIR, f'{ticker}_*_xgboost_model.pkl')
    matches = glob.glob(pattern)
    if not matches:
        return None, None

    dates = set()
    for m in matches:
        # basename: TICKER_YYYYMMDD_xgboost_model.pkl
        part = os.path.basename(m).split('_')
        if len(part) >= 2 and len(part[1]) == 8 and part[1].isdigit():
            dates.add(part[1])

    if not dates:
        return None, None

    latest = sorted(dates)[-1]
    return f'{ticker}_{latest}_', latest


def cleanup_old_stock_files(ticker, keep_date=None):
    """
    Remove all MODELS/ files for *ticker* except those containing *keep_date*.
    Call this before training so stale pkls don't accumulate.
    """
    removed = 0
    for f in glob.glob(os.path.join(MODEL_DIR, f'{ticker}_*')):
        if keep_date and keep_date in os.path.basename(f):
            continue
        try:
            os.remove(f)
            removed += 1
        except OSError:
            pass
    return removed
