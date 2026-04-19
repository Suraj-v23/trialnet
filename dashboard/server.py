#!/usr/bin/env python3
"""
server.py — Dashboard API Server for TrialNet

Serves training metrics and provides real-time updates via Server-Sent Events.
"""

import json
import os
import sys
import time
from flask import Flask, jsonify, send_from_directory, Response
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/style.css')
def css():
    return send_from_directory('.', 'style.css')


@app.route('/app.js')
def js():
    return send_from_directory('.', 'app.js')


@app.route('/api/history/<mode>')
def get_history(mode):
    """Get training history for a specific mode."""
    filepath = os.path.join(DATA_DIR, f'history_{mode}.json')
    if os.path.exists(filepath):
        with open(filepath) as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'No data found'}), 404


@app.route('/api/comparison')
def get_comparison():
    """Get comparison data across all modes."""
    filepath = os.path.join(DATA_DIR, 'comparison.json')
    if os.path.exists(filepath):
        with open(filepath) as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'No comparison data. Run: python train.py --mode compare'}), 404


@app.route('/api/live/<mode>')
def get_live(mode):
    """Get live training metrics."""
    filepath = os.path.join(DATA_DIR, f'live_metrics_{mode}.json')
    if os.path.exists(filepath):
        with open(filepath) as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'No live data'}), 404


@app.route('/api/available')
def get_available():
    """List available training data."""
    available = {'modes': [], 'has_comparison': False}
    for mode in ['traditional', 'trial', 'hybrid']:
        if os.path.exists(os.path.join(DATA_DIR, f'history_{mode}.json')):
            available['modes'].append(mode)
    available['has_comparison'] = os.path.exists(os.path.join(DATA_DIR, 'comparison.json'))
    return jsonify(available)


@app.route('/api/stream/<mode>')
def stream(mode):
    """Server-Sent Events stream for real-time updates."""
    def event_stream():
        last_modified = 0
        filepath = os.path.join(DATA_DIR, f'live_metrics_{mode}.json')
        while True:
            if os.path.exists(filepath):
                mtime = os.path.getmtime(filepath)
                if mtime > last_modified:
                    last_modified = mtime
                    with open(filepath) as f:
                        data = f.read()
                    yield f"data: {data}\n\n"
            time.sleep(1)

    return Response(event_stream(), mimetype='text/event-stream')


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    print("\n🌐 TrialNet Dashboard Server starting...")
    print("   Open http://localhost:5050 in your browser\n")
    app.run(host='0.0.0.0', port=5050, debug=True)
