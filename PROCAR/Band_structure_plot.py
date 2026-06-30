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
from matplotlib.widgets import CheckButtons
from collections import defaultdict
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


# LORBIT=14 lm column order (9 columns): s, py, pz, px, dxy, dyz, dz2, dxz, x2-y2.
# Index groups for l-grouped (s/p/d) orbital coloring.
_L_GROUP_COLS = {
    's': [0],
    'p': [1, 2, 3],
    'd': [4, 5, 6, 7, 8],
}
# Orbital -> RGB channel for orbital-mode hue (s=Red, p=Green, d=Blue), mirroring
# the element-mode 0=R,1=G,2=B convention.
_ORBITAL_ORDER = ['s', 'p', 'd']


class CPUOrbitalBandPlotter:
    """
    Handles ingestion of VASP output files and generates an interactive
    band structure plot using generic N-element normalization.
    """
    def __init__(self, directory=".", filter_types=None, color_by='element'):
        self.directory = directory
        self.procar_path = os.path.join(directory, "PROCAR")
        self.poscar_path = os.path.join(directory, "POSCAR")
        self.band_dat_path = os.path.join(directory, "BAND.dat")
        self.klabels_path = os.path.join(directory, "KLABELS")
        self.filter_types = filter_types or []
        self.color_by = color_by   # 'element' (default) or 'orbital'
        self.spin_mode = None
        self.fermi_level = self.parse_doscar(directory)
        self.atom_type_map = self.parse_poscar()

        print(f"--- Initialization Complete ---")
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
        """Header-driven, count-based magnitude parser (LORBIT=14).

        Reads '# of k-points / bands / ions' from the PROCAR header, then for
        each (k-point, band) reads exactly N_ions rows of the FIRST (total-charge)
        magnitude block, keeping all 9 lm columns (s, py, pz, px, dxy, dyz, dz2,
        dxz, x2-y2). Navigation is by declared ion count, never by 'tot' summary
        rows (which a single-ion calculation omits) or other landmark lines, so
        it works for any number of ions. After block 0, it advances to the next
        band/k-point marker, which skips the remaining magnetization blocks
        (NCL/SOC) and the complex phase block.

        Stores self.states[(s_idx, k_idx, b_idx)] -> ndarray (N_ions, 9) of the
        total-charge lm-projected magnitudes."""
        if not os.path.exists(self.procar_path): return
        print(f"\n>>> Starting Header-Driven PROCAR Parsing...")

        with open(self.procar_path, "r") as f:
            content = f.read()

        # --- Header counts (structural invariant; works for any ion count) ---
        n_ions_hdr = None
        for ln in content.splitlines()[:5]:
            m = re.search(r"#\s*of\s*k-points:\s*(\d+)\s*#\s*of\s*bands:\s*(\d+)\s*#\s*of\s*ions:\s*(\d+)", ln, re.I)
            if m:
                n_ions_hdr = int(m.group(3))
                break
        # Fall back to POSCAR-derived count if the header is missing.
        N_atoms = n_ions_hdr if n_ions_hdr is not None else (max(self.atom_type_map.keys()) if self.atom_type_map else 0)

        kpt1_matches = list(re.finditer(r"k[- ]?point\s+1\b", content, re.I))
        spin_blocks = [(0, content[:kpt1_matches[1].start()])] if len(kpt1_matches) > 1 else [(0, content)]
        if len(kpt1_matches) > 1:
            self.ispin = 2
            spin_blocks.append((1, content[kpt1_matches[1].start():]))

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

                    # Advance to the first magnitude header ('ion ...'), step in.
                    while i < len(lines) and not lines[i].strip().lower().startswith("ion"): i += 1
                    i += 1

                    # Read exactly N_atoms rows of the total-charge block; keep 9 lm cols.
                    mags = np.zeros((N_atoms, 9), dtype=np.float32)
                    for _ in range(N_atoms):
                        if i >= len(lines):
                            break
                        tokens = lines[i].split()
                        try:
                            ion_idx = int(tokens[0])
                            vals = [float(x) for x in tokens[1:10]]
                            if 1 <= ion_idx <= N_atoms and len(vals) == 9:
                                mags[ion_idx-1, :] = vals
                        except (ValueError, IndexError):
                            pass
                        i += 1

                    # Skip remaining magnetization/phase blocks: advance to next marker.
                    while i < len(lines) and not (re.search(r"band\s+\d+", lines[i], re.I) or
                                                  re.search(r"k[- ]?point", lines[i], re.I)):
                        i += 1

                    self.states[(s_idx, k_idx, b_idx)] = mags

        print(f"<<< Finished Parsing. Locked {len(self.states)} states (N_ions={N_atoms}, 9 lm columns each).")

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
                    current_e.append(float(p[1]))
            if current_k: band_data.append({'k': np.array(current_k), 'e': np.array(current_e)})
        return band_data

    def parse_klabels(self):
        """Parse the VASPKIT KLABELS file into (tick_coords, tick_labels).

        Logic matches the working unfold_bands.py reader: skip empty lines and
        any line containing 'K-Label' or '*'; take the first token as the label
        and the second as the coordinate; convert GAMMA -> $\\Gamma$. If the file
        is absent or unreadable, returns empty lists so the plot keeps normal
        numeric x-axis ticks."""
        tick_coords, tick_labels = [], []
        if not os.path.exists(self.klabels_path):
            return tick_coords, tick_labels
        with open(self.klabels_path, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts or 'K-Label' in line or '*' in line:
                    continue
                try:
                    label = parts[0]
                    coord = float(parts[1])
                    if label.upper() == 'GAMMA':
                        label = r'$\Gamma$'
                    tick_labels.append(label)
                    tick_coords.append(coord)
                except (ValueError, IndexError):
                    continue
        return tick_coords, tick_labels

    def _apply_klabels(self, ax):
        """Apply KLABELS high-symmetry ticks + grey guide lines to an axis.
        Absent/empty KLABELS leaves the default numeric x-axis untouched."""
        tick_coords, tick_labels = self.parse_klabels()
        if tick_coords:
            for coord in tick_coords:
                ax.axvline(x=coord, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_xticks(tick_coords)
            ax.set_xticklabels(tick_labels)

    def plot_colored_bands(self):
        """Build the interactive figure with an 'Orbital' toggle (bottom left).

        Toggle OFF (default): single-panel element mode, element->RGB using the
        input filter_types (None slots skip a channel).
        Toggle ON: one subplot per element NAMED IN filter_types (None slots
        skipped); coloring switches to orbital (hue = s/p/d blend, visibility =
        total population). Clicking rebuilds the figure live."""
        self.band_raw = self.parse_band_dat()
        self.fig = plt.figure(figsize=(12, 8))
        # 'Orbital' toggle, bottom left.
        self._ax_toggle = self.fig.add_axes([0.01, 0.01, 0.12, 0.06])
        self._chk_orbital = CheckButtons(self._ax_toggle, ['Orbital'],
                                         [getattr(self, 'color_by', 'element') == 'orbital'])
        self._chk_orbital.on_clicked(self._on_toggle)
        self._rebuild_figure()
        plt.show()

    def _on_toggle(self, label):
        states = self._chk_orbital.get_status()
        self.color_by = 'orbital' if states[0] else 'element'
        self._rebuild_figure()
        self.fig.canvas.draw_idle()

    def _rebuild_figure(self):
        """Clear and redraw the figure body for the current mode, preserving the
        toggle widget axes."""
        # Remove every axis except the toggle.
        for ax in list(self.fig.axes):
            if ax is not self._ax_toggle:
                self.fig.delaxes(ax)
        if getattr(self, 'color_by', 'element') == 'orbital':
            self._draw_orbital_mode()
        else:
            self._draw_element_mode()

    def _draw_element_mode(self):
        """Single-panel coloring: each band point's RGB is the normalized
        composition of the named elements (0=R, 1=G, 2=B). None/falsy filter
        slots leave their channel empty. Works for any number of ions/elements,
        including one. Per-ion weight is the sum over all 9 lm columns (= the
        total-charge 'tot' for that ion)."""
        band_raw = self.band_raw
        ax = self.fig.add_subplot(1, 1, 1)

        for s_idx in range(self.ispin):
            for b_idx, band in enumerate(band_raw):
                k_coords, e_coords = band['k'], band['e']
                colors = []

                for k_idx in range(len(k_coords)):
                    mags = self.states.get((s_idx, k_idx, b_idx), None)

                    # Named (non-None) filter slots only; None slots leave their channel empty.
                    active_types = [t for t in self.filter_types if t]
                    type_weights = {t: 0.0 for t in active_types}
                    if mags is not None:
                        for ion_i, elem in self.atom_type_map.items():
                            if elem in type_weights:
                                # Sum the 9 lm columns -> per-ion total charge.
                                type_weights[elem] += float(mags[ion_i-1].sum())

                    # Per-position weights; a None/falsy slot contributes 0 to its channel.
                    weights = [type_weights[t] if t else 0.0 for t in self.filter_types]
                    w_total = sum(weights)

                    if w_total > 0:
                        r = weights[0] / w_total
                        g = weights[1] / w_total if len(weights) > 1 else 0.0
                        b = weights[2] / w_total if len(weights) > 2 else 0.0
                        colors.append([r, g, b, 1.0])
                    else:
                        colors.append([0.0, 0.0, 0.0, 1.0])

                ax.scatter(k_coords, e_coords, c=colors, s=15, edgecolors='none', zorder=2)

        base_colors = [[1,0,0], [0,1,0], [0,0,1]]
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=e,
                          markerfacecolor=c, markersize=10)
                          for e, c in zip(self.filter_types, base_colors) if e]
        ax.legend(handles=legend_elements, title="Atomic Projections", loc='upper right')

        self._apply_klabels(ax)
        ax.axhline(0, color='red', ls='-', lw=0.8, alpha=0.5)
        ax.set_ylim(-3, 3); ax.set_ylabel('Energy (eV)')
        ax.set_title(f"Dynamic Normalized Fatbands: {self.filter_types}")

    def _draw_orbital_mode(self):
        """One subplot per element NAMED IN filter_types (None slots skipped).
        Within each element's panel, HUE = the s/p/d orbital blend (s=R, p=G,
        d=B, normalized like element mode) summed over that element's ions, and
        VISIBILITY = the element's total population at that point (high
        population -> vivid hue; low population -> washed toward transparent,
        self-scaled per element). Works for any number of ions/elements."""
        band_raw = self.band_raw

        # Elements to show = the named (non-None) filter slots, in input order,
        # de-duplicated. Orbital mode shows ONLY the filtered elements.
        elements = []
        for t in self.filter_types:
            if t and t not in elements:
                elements.append(t)
        if not elements:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No elements in filter_types to show.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        n_el = len(elements)

        # Per-element max total population, for self-scaled visibility/alpha.
        el_max_pop = {e: 1e-30 for e in elements}
        for s_idx in range(self.ispin):
            for b_idx, band in enumerate(band_raw):
                for k_idx in range(len(band['k'])):
                    mags = self.states.get((s_idx, k_idx, b_idx), None)
                    if mags is None:
                        continue
                    for ion_i, elem in self.atom_type_map.items():
                        if elem in el_max_pop:
                            pop = float(mags[ion_i-1].sum())
                            if pop > el_max_pop[elem]:
                                el_max_pop[elem] = pop

        for ax_idx, elem in enumerate(elements):
            ax = self.fig.add_subplot(1, n_el, ax_idx + 1)
            ion_indices = [ion_i for ion_i, e in self.atom_type_map.items() if e == elem]

            for s_idx in range(self.ispin):
                for b_idx, band in enumerate(band_raw):
                    k_coords, e_coords = band['k'], band['e']
                    colors = []
                    for k_idx in range(len(k_coords)):
                        mags = self.states.get((s_idx, k_idx, b_idx), None)
                        if mags is None:
                            colors.append([1.0, 1.0, 1.0, 0.0])
                            continue

                        # Sum this element's ions over the s/p/d column groups.
                        spd = np.zeros(3, dtype=np.float64)
                        for ci, orb in enumerate(_ORBITAL_ORDER):
                            cols = _L_GROUP_COLS[orb]
                            for ion_i in ion_indices:
                                spd[ci] += float(mags[ion_i-1][cols].sum())

                        pop = float(spd.sum())  # element total population at this point
                        if pop > 0:
                            # Hue = normalized s/p/d blend (same logic as element RGB).
                            r, g, b = spd / pop
                            # Visibility: high population -> vivid (alpha->1);
                            # low population -> faint. Self-scaled per element.
                            alpha = min(1.0, pop / el_max_pop[elem])
                            colors.append([r, g, b, alpha])
                        else:
                            colors.append([1.0, 1.0, 1.0, 0.0])

                    ax.scatter(k_coords, e_coords, c=colors, s=15, edgecolors='none', zorder=2)

            # Per-panel orbital legend (s=R, p=G, d=B).
            orb_colors = [[1,0,0], [0,1,0], [0,0,1]]
            legend_elements = [Line2D([0], [0], marker='o', color='w', label=orb,
                              markerfacecolor=c, markersize=10)
                              for orb, c in zip(_ORBITAL_ORDER, orb_colors)]
            ax.legend(handles=legend_elements, title="Orbitals", loc='upper right')

            self._apply_klabels(ax)
            ax.axhline(0, color='red', ls='-', lw=0.8, alpha=0.5)
            ax.set_ylim(-3, 3)
            ax.set_title(f"{elem}")
            if ax_idx == 0:
                ax.set_ylabel('Energy (eV)')

        self.fig.suptitle(f"Orbital-Resolved Fatbands (per element): {elements}", fontsize=13)

if __name__ == "__main__":
    # Element mode by default; click the bottom-left 'Orbital' toggle to switch
    # to per-element orbital subplots (only the elements named in filter_types).
    plotter = CPUOrbitalBandPlotter(directory=".", filter_types=['Cs', 'Sn', 'I'])
    plotter.parse_procar()
    plotter.plot_colored_bands()
