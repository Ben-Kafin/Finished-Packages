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

    def _build_state_tensor(self):
        """Stack self.states into ONE dense array, ONCE, so both color modes are
        fast array operations and toggling is instant.

        Builds self._state_arr with shape (n_spin, n_bands, max_k, N_ions, 9),
        indexed to match the draw loops: [s_idx, b_idx, k_idx, ion, lm]. Lookup
        misses stay zero. Ragged bands are padded to max_k; per-band valid k-count
        is recorded in self._band_nk so draws only plot real points.

        This is the single heavy step; everything downstream (element sums,
        orbital s/p/d, per-element population, both modes, every toggle) reads
        from this tensor with vectorized numpy, never re-walking the dict."""
        band_raw = self.band_raw
        n_bands = len(band_raw)
        self._band_nk = [len(b['k']) for b in band_raw]
        max_k = max(self._band_nk, default=0)

        # Infer N_ions from any stored state (all share the same first dim).
        n_ions = 0
        for v in self.states.values():
            n_ions = v.shape[0]
            break

        arr = np.zeros((self.ispin, n_bands, max_k, n_ions, 9), dtype=np.float32)
        for (s_idx, k_idx, b_idx), mg in self.states.items():
            if s_idx < self.ispin and b_idx < n_bands and k_idx < max_k:
                arr[s_idx, b_idx, k_idx] = mg
        self._state_arr = arr
        self._n_ions = n_ions

        # Column-group index arrays for s/p/d.
        self._s_cols = np.array(_L_GROUP_COLS['s'])
        self._p_cols = np.array(_L_GROUP_COLS['p'])
        self._d_cols = np.array(_L_GROUP_COLS['d'])

        # Map each element to its 0-based ion indices.
        self._elem_ion0 = {}
        for ion_i, elem in self.atom_type_map.items():
            self._elem_ion0.setdefault(elem, []).append(ion_i - 1)

    def _elem_spd(self, elem):
        """Vectorized per-element s/p/d magnitudes from the state tensor.
        Returns array (n_spin, n_bands, max_k, 3) = (s_sum, p_sum, d_sum) summed
        over that element's ions. Pure array ops; no Python per-point loop."""
        ions0 = self._elem_ion0.get(elem, [])
        if not ions0:
            sh = self._state_arr.shape
            return np.zeros((sh[0], sh[1], sh[2], 3), dtype=np.float64)
        sub = self._state_arr[:, :, :, ions0, :]            # (S,B,K,n_e_ions,9)
        s = sub[..., self._s_cols].sum(axis=(3, 4))         # (S,B,K)
        p = sub[..., self._p_cols].sum(axis=(3, 4))
        d = sub[..., self._d_cols].sum(axis=(3, 4))
        return np.stack([s, p, d], axis=-1)                 # (S,B,K,3)

    def plot_colored_bands(self):
        """Build the interactive figure with an 'Orbital' toggle (bottom left).

        Toggle OFF (default): single-panel element mode, element->RGB using the
        input filter_types (None slots skip a channel).
        Toggle ON: one subplot per element NAMED IN filter_types (None slots
        skipped); coloring switches to orbital (hue = s/p/d blend, visibility =
        total population). Both modes read one precomputed state tensor, so
        clicking the toggle is instant."""
        self.band_raw = self.parse_band_dat()
        self._build_state_tensor()   # single heavy step; both modes read this
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
        including one. Per-element weight is the total charge on that element
        (sum of all 9 lm columns over its ions), computed by vectorized array ops
        from the state tensor. One scatter call per band."""
        ax = self.fig.add_subplot(1, 1, 1)

        # Per-element total population: sum s/p/d -> (S,B,K) for each filter element.
        active = [t for t in self.filter_types if t]
        # weights[channel] is (S,B,K); channel order follows filter_types (None -> 0).
        chan_weight = []
        for t in self.filter_types:
            if t:
                spd = self._elem_spd(t)          # (S,B,K,3)
                chan_weight.append(spd.sum(axis=-1))  # (S,B,K) total charge on element
            else:
                chan_weight.append(None)

        # Stack the up-to-3 channels into an RGB weight array.
        S, B, K = self._state_arr.shape[0], self._state_arr.shape[1], self._state_arr.shape[2]
        rgbw = np.zeros((S, B, K, 3), dtype=np.float64)
        for ci in range(min(3, len(self.filter_types))):
            if chan_weight[ci] is not None:
                rgbw[..., ci] = chan_weight[ci]
        w_total = rgbw.sum(axis=-1)                # (S,B,K)
        with np.errstate(divide='ignore', invalid='ignore'):
            rgb = np.where(w_total[..., None] > 0, rgbw / w_total[..., None], 0.0)
        # Points with zero total -> black (matches original [0,0,0,1]).

        for s_idx in range(self.ispin):
            for b_idx, band in enumerate(self.band_raw):
                nk = self._band_nk[b_idx]
                k_coords, e_coords = band['k'], band['e']
                colors = np.ones((nk, 4), dtype=np.float64)   # alpha 1
                colors[:, :3] = rgb[s_idx, b_idx, :nk, :]
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
        d=B, normalized like element mode), and VISIBILITY (alpha) = the element's
        total population at that point, self-scaled per element. Vectorized array
        ops from the state tensor; one scatter call per band."""
        # Elements to show = named (non-None) filter slots, de-duplicated, input order.
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
        for ax_idx, elem in enumerate(elements):
            ax = self.fig.add_subplot(1, n_el, ax_idx + 1)
            spd = self._elem_spd(elem)                # (S,B,K,3)
            pop = spd.sum(axis=-1)                     # (S,B,K) element total population
            max_pop = float(pop.max()) if pop.size else 0.0
            if max_pop <= 0:
                max_pop = 1e-30
            with np.errstate(divide='ignore', invalid='ignore'):
                hue = np.where(pop[..., None] > 0, spd / pop[..., None], 0.0)  # (S,B,K,3)
            alpha = np.minimum(1.0, pop / max_pop)    # (S,B,K)

            for s_idx in range(self.ispin):
                for b_idx, band in enumerate(self.band_raw):
                    nk = self._band_nk[b_idx]
                    k_coords, e_coords = band['k'], band['e']
                    colors = np.empty((nk, 4), dtype=np.float64)
                    colors[:, :3] = hue[s_idx, b_idx, :nk, :]
                    colors[:, 3] = alpha[s_idx, b_idx, :nk]
                    # Points with zero population -> alpha 0 (invisible), matching original.
                    ax.scatter(k_coords, e_coords, c=colors, s=15, edgecolors='none', zorder=2)

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
