# -*- coding: utf-8 -*-
"""
Root entrypoint for Render deployment
"""
import sys
import os

# Add webapp directory to path
webapp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webapp')
if webapp_dir not in sys.path:
    sys.path.insert(0, webapp_dir)

from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
