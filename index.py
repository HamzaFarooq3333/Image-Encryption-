"""Vercel serverless entry‑point wrapping your Flask app.

Deploy steps:
1. Put **app.py** at the project root (as you have now).
2. Keep *this* file in **api/index.py** – Vercel’s Python
   runtime will treat it as a Serverless Function.
3. Optionally add a `vercel.json` (see docs) that routes every
   request to `/api/index.py` if you want clean URLs.
"""

import sys
from pathlib import Path

# Ensure the project root (where app.py lives) is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import the Flask application object from app.py
from app import app as flask_app

def handler(environ, start_response):
    """WSGI‑compatible entry‑point expected by Vercel."""
    return flask_app(environ, start_response)
