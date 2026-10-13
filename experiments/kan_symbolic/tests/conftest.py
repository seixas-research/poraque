# -*- coding: utf-8 -*-
# file: conftest.py
"""Put the archived modules (``../src``) on the path, as top-level modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
