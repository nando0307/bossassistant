from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("NEO4J_URI", "neo4j://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("NVIDIA_API_KEY", "test-key")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
