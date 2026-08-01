"""Stable import path for the OmniHand Pro articulation configuration.

The original description directory contains hyphens, so it cannot be imported
with normal dotted Python syntax.  This module keeps the source configuration in
that directory while exposing a regular import path to Isaac Lab tasks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_CONFIG_PATH = Path(__file__).parent / "o12_hand_description-o12_t3" / "omnihand.py"
_SPEC = importlib.util.spec_from_file_location("atec_rl_lab._omnihand_pro_cfg", _CONFIG_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load OmniHand Pro configuration from {_CONFIG_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

OMNIHAND_CFG = _MODULE.OMNIHAND_CFG
OMNIHAND_ACTION_JOINT_NAMES = _MODULE.OMNIHAND_ACTION_JOINT_NAMES
OMNIHAND_JOINT_NAMES = _MODULE.OMNIHAND_JOINT_NAMES

__all__ = ["OMNIHAND_CFG", "OMNIHAND_ACTION_JOINT_NAMES", "OMNIHAND_JOINT_NAMES"]
