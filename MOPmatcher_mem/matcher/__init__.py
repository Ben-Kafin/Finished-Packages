# -*- coding: utf-8 -*-
"""
matcher: True Bloch state matching pipeline for VASP DFT outputs.
Auto-detects ISPIN=1, ISPIN=2, and NCL/SOC calculations.

Author: Benjamin Kafin
"""

from .matcher_vacupot import RectangularTrueBlochMatcher, run_match
from .vacupot_plotter import RectAEPAWColorPlotter, PlotConfig
from .classifier import StateBehaviorClassifier
from .spread_plot import plot_spread
from .state_decomposition_plot import plot_state_decomposition

__all__ = [
    "RectangularTrueBlochMatcher",
    "run_match",
    "RectAEPAWColorPlotter",
    "PlotConfig",
    "StateBehaviorClassifier",
    "plot_spread",
    "plot_state_decomposition",
]
