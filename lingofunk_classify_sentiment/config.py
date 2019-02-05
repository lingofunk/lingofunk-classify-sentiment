import json
import os
from pathlib import Path


ROOT = Path(__file__).parent
CONFIG_PATH = os.path.join(ROOT, "config.json")

with open(CONFIG_PATH) as f:
    config = json.load(f)


def fetch(relpath):
    return os.path.join(ROOT, relpath)
