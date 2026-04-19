"""Reproduce the +LNN +ODE column in the F1 chart."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import run_suite


if __name__ == "__main__":
    print(run_suite("lnn_ode_pred", "+ LNN + ODE"))
