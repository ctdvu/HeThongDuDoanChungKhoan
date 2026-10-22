# -*- coding: utf-8 -*-
"""
Root entrypoint for Render deployment
"""
import os
import sys
import importlib.util

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.join(ROOT_DIR, 'webapp')

if WEBAPP_DIR not in sys.path:
    sys.path.insert(0, WEBAPP_DIR)

# Load webapp/app.py cleanly under unique module name to prevent circular import
spec = importlib.util.spec_from_file_location("webapp_backend", os.path.join(WEBAPP_DIR, "app.py"))
webapp_backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(webapp_backend)

# Expose Flask instance as 'app' for Gunicorn (gunicorn app:app)
app = webapp_backend.app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
