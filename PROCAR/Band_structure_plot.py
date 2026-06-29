# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 12:45:00 2026
@author: Benjamin Kafin

Educational Version: Integrated VASP Fatband Plotter.
This version is fully generic, mapping any provided 'filter_types'
to RGB channels without hard-coded element names.

Spin-mode detection added: a single DOSCAR read now determines the Fermi
energy AND auto-detects the spin configuration (collinear unpolarized,
collinear polarized, or non-collinear / SOC) from the site-projected
column count. The detected mode is stored on self.spin_mode for later use;
the coloring path is unchanged.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#from matplotlib.widgets import CheckButtons
#from collections import defaultdict
from enum import Enum, auto


class SpinMode(Enum):
    COLLINEAR_UNPOL = auto()
    COLLINEAR_POL = auto()
    NCL_SOC = auto()


# Column-count -> SpinMode mapping (unambiguous by VASP format)
#   ISPIN=1:  3 (l-decomposed), 9 (lm-decomposed spd), 16 (lm spdf)
#   ISPIN=2:  6, 18, 32
#   NCL/SOC:  12 (l x 4), 36 (lm-spd x 4), 64 (lm-spdf x 4)
_COL_TO_MODE = {}
for _n in (3, 9, 16):
    _COL_TO_MODE[_n] = SpinMode.COLLINEAR_UNPOL
for _n in (6, 18, 32):
    _COL_TO_MODE[_n] = SpinMode.COLLINEAR_POL
for _n in (12, 36, 64):
    _COL_TO_MODE[_n] = SpinMode.NCL_SOC


class CPUOrbitalBandPlotter:
    """
    Handles ingestion of VASP output files and generates an interactive
    band structure plot using generic N-element normalization.
    """
    def __init__(self, directory=".", filter_types=None):
        self.directory = directory
        self.procar_path = os.path.join(directory, "PROCAR")
        self.poscar_path = os.path.join(directory, "POSCAR")
        self.band_dat_path = os.path.join(directory, "BAND.dat")
        self.klabels_path = os.path.join(directory, "KLABELS")
        self.filter_types = filter_types or []
        self.spin_mode = None
        self.fermi_level = self.parse_doscar(directory)
        self.atom_type_map = self.parse_poscar()

        print("--- Initialization Complete ---")
        print(f"Fermi Level (from DOSCAR): {self.fermi_level:.4f} eV")
        print(f"Detected Spin Mode: {self.spin_mode.name if self.spin_mode is not None else 'UNKNOWN'}")
        print(f"POSCAR Mapping: {len(self.atom_type_map)} ions found")
        print(f"Filters Active: {self.filter_types}")

        self.states = {} # O(1) Lookup: (spin, k_idx, b_idx)
        self.ispin = 1

    def parse_doscar(self, directory):
        """Single DOSCAR read: returns the Fermi energy and sets self.spin_mode
        from the site-projected column count.

        Line 1 token 0 = ion count; line 6 (index 5) holds nedos (token 2) and
        the Fermi energy (token 3). Block 0 is the total DOS; the first
        site-projected block (block 1, preceded by a header line) gives the
        column count used to detect the spin configuration."""
        doscar_path = os.path.join(directory, "DOSCAR")
        if not os.path.exists(doscar_path):
            return 0.0
        with open(doscar_path, "r") as f:
            atomnum = int(f.readline().split()[0])
            for _ in range(4):
                f.readline()
            header = f.readline().split()
            nedos, ef = int(header[2]), float(header[3])

            # Block 0: total DOS (skip its nedos rows).
            for _ in range(nedos):
                f.readline()

            # Block 1: first site-projected atom block, preceded by a header
            # line. Read one data row to count its site-projected columns.
            if atomnum >= 1:
                f.readline()  # atom block header line
                first_row = f.readline().split()
                num_cols = len(first_row) - 1  # drop the leading energy column
                self.spin_mode = _COL_TO_MODE.get(num_cols)

        return ef

    def parse_poscar(self):
        mapping = {}
        if not os.path.exists(self.poscar_path): return mapping
        with open(self.poscar_path) as f:
            lines = [ln.strip() for ln in f]
        elements, counts = lines[5].split(), [int(x) for x in lines[6].split()]
        idx = 1
        for elem, num in zip(elements, counts):
            for _ in range(num):
                mapping[idx] = elem
                idx += 1
        return mapping

    def parse_procar(self):
        """Zero-indexed Magnitude parsing. Skips LORBIT=14 phase blocks."""
        if not os.path.exists(self.procar_path): return
        print("\n>>> Starting Zero-Based PROCAR Parsing...")

        with open(self.procar_path, "r") as f:
            content = f.read()

        kpt1_matches = list(re.finditer(r"k[- ]?point\s+1\b", content, re.I))
        spin_blocks = [(0, content[:kpt1_matches[1].start()])] if len(kpt1_matches) > 1 else [(0, content)]
        if len(kpt1_matches) > 1:
            self.ispin = 2
            spin_blocks.append((1, content[kpt1_matches[1].start():]))

        N_atoms = max(self.atom_type_map.keys()) if self.atom_type_map else 0

        for s_idx, data in spin_blocks:
            lines = data.splitlines()
            i = 0
            while i < len(lines):
                k_match = re.search(r"k[- ]?point\s+(\d+)", lines[i], re.I)
                if not k_match: i += 1; continue
                k_idx = int(k_match.group(1)) - 1
                i += 1

                while i < len(lines) and not re.search(r"k[- ]?point", lines[i], re.I):
                    b_match = re.search(r"band\s+(\d+)", lines[i], re.I)
                    if not b_match: i += 1; continue
                    b_idx = int(b_match.group(1)) - 1
                    i += 1

                    while i < len(lines) and not lines[i].strip().lower().startswith("ion"): i += 1
                    i += 1

                    mags = np.zeros(N_atoms, dtype=np.float32)
                    while i < len(lines):
                        ln = lines[i].strip()
                        if not ln or ln.lower().startswith("tot"):
                            while i < len(lines) and not (re.search(r"band\s+\d+", lines[i], re.I) or
                                                          re.search(r"k[- ]?point", lines[i], re.I)):
                                i += 1
                            break
                        tokens = ln.split()
                        mags[int(tokens[0])-1] = float(tokens[-1])
                        i += 1

                    self.states[(s_idx, k_idx, b_idx)] = mags

        print(f"<<< Finished Parsing. Locked {len(self.states)} zero-indexed states.")

    def parse_band_dat(self):
        band_data = []
        if not os.path.exists(self.band_dat_path): return []
        with open(self.band_dat_path, 'r') as f:
            current_k, current_e = [], []
            for line in f:
                if line.startswith('# Band-Index:'):
                    if current_k: band_data.append({'k': np.array(current_k), 'e': np.array(current_e)})
                    current_k, current_e = [], []
                    continue
                if not line.startswith('#') and line.strip():
                    p = line.split()
                    current_k.append(float(p[0]))
                    current_e.append(float(p[1]) - self.fermi_level)
            if current_k: band_data.append({'k': np.array(current_k), 'e': np.array(current_e)})
        return band_data

    def plot_colored_bands(self):
        """Renders scatter points with full hue normalization for ANY provided elements."""
        print("\nGenerating Generic Normalized Band Plot...")
        band_raw = self.parse_band_dat()
        fig, ax = plt.subplots(figsize=(12, 8))
        if self.ispin == 2: plt.subplots_adjust(right=0.85)

        spin_collections = {0: [], 1: []}

        for s_idx in range(self.ispin):
            for b_idx, band in enumerate(band_raw):
                k_coords, e_coords = band['k'], band['e']
                colors = []

                for k_idx in range(len(k_coords)):
                    mags = self.states.get((s_idx, k_idx, b_idx), np.zeros(2000))

                    # Named (non-None) filter slots only; None slots leave their channel empty.
                    active_types = [t for t in self.filter_types if t]
                    type_weights = {t: 0.0 for t in active_types}
                    for ion_i, elem in self.atom_type_map.items():
                        if elem in type_weights:
                            type_weights[elem] += mags[ion_i-1]

                    # Per-position weights; a None/falsy slot contributes 0 to its channel.
                    weights = [type_weights[t] if t else 0.0 for t in self.filter_types]
                    w_total = sum(weights)

                    if w_total > 0:
                        # Normalize first 3 elements to RGB channels
                        r = weights[0] / w_total
                        g = weights[1] / w_total if len(weights) > 1 else 0.0
                        b = weights[2] / w_total if len(weights) > 2 else 0.0
                        colors.append([r, g, b, 1.0])
                    else:
                        colors.append([0.0, 0.0, 0.0, 1.0])

                sc = ax.scatter(k_coords, e_coords, c=colors, s=15, edgecolors='none', zorder=2)
                spin_collections[s_idx].append(sc)

        # Dynamic Legend based on input filter order
        base_colors = [[1,0,0], [0,1,0], [0,0,1]]
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=e,
                          markerfacecolor=c, markersize=10)
                          for e, c in zip(self.filter_types, base_colors) if e]
        ax.legend(handles=legend_elements, title="Atomic Projections", loc='upper right')

        ax.axhline(0, color='red', ls='-', lw=0.8, alpha=0.5)
        ax.set_ylim(-3, 3); ax.set_ylabel('Energy (eV)')
        ax.set_title(f"Dynamic Normalized Fatbands: {self.filter_types}")
        plt.show()

if __name__ == "__main__":
    # The plotter now purely follows this input order: 0=Red, 1=Green, 2=Blue
    plotter = CPUOrbitalBandPlotter(directory=".", filter_types=['Cs', 'Sn', 'I'])
    plotter.parse_procar()
    plotter.plot_colored_bands()
