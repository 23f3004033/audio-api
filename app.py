from flask import Flask, request, jsonify
import base64
import numpy as np
import librosa
import io

app = Flask(__name__)

# -----------------------------------
# Standard fallback response (IMPORTANT)
# -----------------------------------
def fallback_response():
    return {
        "rows": 0,
        "columns": [],
        "mean": {},
        "std": {},
        "variance": {},
        "min": {},
        "max": {},
        "median": {},
        "mode": {},
        "range": {},
        "allowed_values": {},
        "value_range": {},
        "correlation": []
    }


# -----------------------------------
# Decode base64 audio safely
# -----------------------------------
def decode_audio(audio_base64):
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_file = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_file, sr=None)
        return y, sr
    except:
        return np.array([0.0]), 1


# -----------------------------------
# Extract features (MFCC)
# -----------------------------------
def extract_features(y, sr):
    try:
        if y is None or len(y) == 0:
            return np.array([0.0])

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc.flatten()
    except:
        return np.array([0.0])


# -----------------------------------
# Safe mode calculation
# -----------------------------------
def safe_mode(arr):
    try:
        if len(arr) == 0:
            return 0.0
        vals, counts = np.unique(arr, return_counts=True)
        return float(vals[np.argmax(counts)])
    except:
        return 0.0


# -----------------------------------
# Compute statistics
# -----------------------------------
def compute_stats(features):
    try:
        features = np.array(features, dtype=float)

        if len(features) == 0:
            features = np.array([0.0])

        min_val = float(np.min(features))
        max_val = float(np.max(features))

        return {
            "rows": int(len(features)),
            "columns": ["value"],
            "mean": {"value": float(np.mean(features))},
            "std": {"value": float(np.std(features))},
            "variance": {"value": float(np.var(features))},
            "min": {"value": min_val},
            "max": {"value": max_val},
            "median": {"value": float(np.median(features))},
            "mode": {"value": safe_mode(features)},
            "range": {"value": float(max_val - min_val)},
            "allowed_values": {"value": []},
            "value_range": {"value": [min_val, max_val]},
            "correlation": []
        }

    except:
        return fallback_response()


# -----------------------------------
# API endpoint (MAIN)
# -----------------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json(force=True)

        if not data or "audio_base64" not in data:
            return jsonify(fallback_response())

        audio_base64 = data.get("audio_base64", "")

        y, sr = decode_audio(audio_base64)
        features = extract_features(y, sr)
        result = compute_stats(features)

        return jsonify(result)

    except:
        return jsonify(fallback_response())


# -----------------------------------
# Optional root route (for browser)
# -----------------------------------
@app.route('/')
def home():
    return "API is running 🚀"


# -----------------------------------
# Run server
# -----------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)