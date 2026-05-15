"""
    Shared dependencies and convenience exports for ASP scripts.
    Set ASP_HOME before doing anything
"""

import importlib
import sys
from pathlib import Path
import json

# ASP_HOME = "/home/bepennell/research/AstrometricSignalProcessing"

# asp_home_path = Path(ASP_HOME).expanduser().resolve()
# if str(asp_home_path) not in sys.path:
#     sys.path.insert(0, str(asp_home_path))

with open('./config.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
folder_a_path = Path(d["home"])

if str(folder_a_path) not in sys.path:
    sys.path.append(str(folder_a_path))
    
import numpy as np
import pickle
import matplotlib.pyplot as plt
from astropy.table import Table, join, vstack
import pandas as pd
from datetime import datetime, date
from utils.asp import asp_import

# for progress bars
try:
    # for Jupyter
    from tqdm.notebook import tqdm
except ImportError:
    # for terminal
    from tqdm import tqdm

# nonsense to load in the other modules
EXPORTED_MODULES = ["utils.asp", "utils.style", "utils.utils", "utils.massfunction"]

for module_name in EXPORTED_MODULES:
    module = importlib.import_module(module_name)
    names = getattr(module, "__all__", None)
    if names is None:
        names = [name for name in dir(module) if not name.startswith("_")]
    globals().update({name: getattr(module, name) for name in names})