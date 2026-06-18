# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 2026

@author: Benjamin Kafin

Standalone spread plotter.
For a list of component labels and a shared list of band indices, produces a
figure showing how each component MO distributes its squared overlap across the
full-system Bloch states.

Two CheckButtons in the figure (lower-left) toggle the layout live:

  Vertical OFF, Horizontal OFF (base grid):
      one subplot per (band index row x component column).

  Vertical ON only:
      one column per component; that component's band-index spreads are
      overlaid onto a single axis.

  Horizontal ON only:
      one axis per band index; every component's spread of that state is
      overlaid onto the single axis, with each component offset in lightness
      (same hue) so they remain distinguishable.

  both ON:
      a single axis holds every component's spread of every band index.

The toggles are implemented exactly as in stm_from_dft.simulator: a persistent
figure, a _build_ui() that does fig.clf(), recreates the CheckButtons on a
dedicated axes region, rebuilds the per-mode subplot layout, and re-wires the
callback; the callback reads chk.get_status() and rebuilds.

Color/data are obtained through RectAEPAWColorPlotter._prepare so that the
per-component colors (including SOC Kramers pairing, shared-molecule, and metal
handling) are identical to the main vacupot_plotter rendering.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

try:
    import mplcursors
    HAS_MPLCURSORS = True
except Exception:
    HAS_MPLCURSORS = False

if __name__ != "__main__":
    from .vacupot_plotter import (
        RectAEPAWColorPlotter,
        PlotConfig,
        _read_ov_all,
    )


class SpreadPlotter:
    """
    Builds and maintains a spread-plot figure whose layout is controlled by two
    in-figure CheckButtons (Vertical / Horizontal). All data loading and color
    construction happen once in __init__; toggling a checkbox only rebuilds the
    drawing via _build_ui().
    """

    def __init__(
        self,
        output_dir: str,
        components: List[str],
        comp_indices: List[int],
        collapse_vertical: bool = False,
        collapse_horizontal: bool = False,
        darken_factor: float = 0.4,
        cfg: Optional[PlotConfig] = None,
    ):
        self.output_dir = output_dir
        self.components = list(components)
        self.comp_indices = list(comp_indices)
        self.collapse_vertical = bool(collapse_vertical)
        self.collapse_horizontal = bool(collapse_horizontal)
        self.collapse_segmentation = False
        self.darken_factor = float(darken_factor)
        self.cfg = cfg if cfg is not None else PlotConfig()

        self.plotter = RectAEPAWColorPlotter(self.cfg)

        # --------------------------------------------------------------
        # Load data + build colors through _prepare (SOC-correct path,
        # identical to vacupot_plotter). write_summaries=False -> no files.
        # --------------------------------------------------------------
        main_path = os.path.join(self.output_dir, "band_matches_rectangular.txt")
        P = self.plotter._prepare(main_path, bonding=False, write_summaries=False)

        self.component_colors = P.component_colors
        self.comp_pairs = P.comp_pairs
        self.by_full = P.by_full
        self.known_fermis = P.known_fermis
        self.v_aligned = P.v_aligned

        # --------------------------------------------------------------
        # Energy shift: put full-system Fermi at zero
        # --------------------------------------------------------------
        fermi_full = self.known_fermis.get("full", 0.0)
        self.fermi_shift = -fermi_full
        self.v_shifted = (self.v_aligned + self.fermi_shift) if self.v_aligned is not None else None

        # Default x-min: lowest energy present in both systems (pooled over the
        # plotted components, which are identical molecules so share states).
        comp_energies = [E
                         for component in self.components
                         for idx, E in self.comp_pairs.get(component, [])]
        full_energies = [rec["E"] + rec["dE"]
                         for comps in self.by_full.values()
                         for recs in comps.values()
                         for rec in recs]
        if comp_energies and full_energies:
            self.shared_e_min = max(min(comp_energies), min(full_energies)) + self.fermi_shift
        else:
            self.shared_e_min = None

        # --------------------------------------------------------------
        # Per-component component-state energy maps
        # --------------------------------------------------------------
        self.comp_energy_map: Dict[str, Dict[int, float]] = {}
        for component in self.components:
            m: Dict[int, float] = {}
            for idx, E in self.comp_pairs.get(component, []):
                m[idx] = E
            self.comp_energy_map[component] = m

        # --------------------------------------------------------------
        # Z-order ranking by band-index value (lower in back)
        # --------------------------------------------------------------
        self.sorted_idx = sorted(self.comp_indices)
        self.R_count = len(self.comp_indices)

        # Per-axis cursor registry (rebuilt each _build_ui)
        self.cursor_by_axes: Dict[Any, Any] = {}

        # Persistent figure (created once, like simulator.py)
        self.fig = plt.figure(figsize=(self.cfg.figsize[0], max(3.0, 1.8 * max(1, self.R_count))))
        self._build_ui()

    # ==================================================================
    # Color helpers
    # ==================================================================
    def _base_rgb_for(self, component: str, comp_idx: int) -> np.ndarray:
        base_color = self.component_colors.get(component, {}).get(
            comp_idx, (0.4, 0.4, 0.4, 1.0)
        )
        return np.array(base_color[:3])

    def _light_rgb_from(self, base_rgb: np.ndarray, f: float) -> np.ndarray:
        # toward white by blend fraction f
        return base_rgb + (1.0 - base_rgb) * f

    def _dark_rgb_from(self, base_rgb: np.ndarray) -> np.ndarray:
        # toward black by darken_factor/2
        return base_rgb * (1.0 - self.darken_factor / 2.0)

    def _light_f_for(self, j: int, n: int) -> float:
        """
        Per-component toward-white blend fraction. For n components spread
        evenly across [0, darken_factor]: component 0 -> 0 (base, darkest),
        component n-1 -> darken_factor (lightest), midpoint darken_factor/2
        coincides with the non-offset light bar color. For n == 1, no offset
        (darken_factor/2).
        """
        if n <= 1:
            return self.darken_factor / 2.0
        return self.darken_factor * j / (n - 1)

    # ==================================================================
    # Energy / fermi averaging helpers (components are identical molecules)
    # ==================================================================
    def _comp_fermi_shifted(self, component: str) -> float:
        return self.known_fermis.get(component, 0.0) + self.fermi_shift

    def _avg_fermi_shifted(self, comps: List[str]) -> float:
        vals = [self.known_fermis.get(c, 0.0) for c in comps]
        return (sum(vals) / len(vals) + self.fermi_shift) if vals else self.fermi_shift

    def _avg_marker_E(self, comps: List[str], comp_idx: int) -> Optional[float]:
        es = []
        for c in comps:
            e = self.comp_energy_map.get(c, {}).get(comp_idx)
            if e is not None:
                es.append(e)
        if not es:
            return None
        return sum(es) / len(es) + self.fermi_shift

    # ==================================================================
    # Z-order
    # ==================================================================
    def _rank_of(self, comp_idx: int) -> int:
        return self.sorted_idx.index(comp_idx)

    def _marker_z(self, comp_idx: int) -> float:
        return 1.0 + (self._rank_of(comp_idx) / max(self.R_count, 1)) * 0.9

    def _spread_z(self, comp_idx: int) -> float:
        return 2.0 + (self._rank_of(comp_idx) / max(self.R_count, 1)) * 0.9

    # ==================================================================
    # Drawing primitives
    # ==================================================================
    def _draw_spread_bars(self, ax, component, comp_idx, f, zorder):
        base = self._base_rgb_for(component, comp_idx)
        col = tuple(self._light_rgb_from(base, f))
        local_artists = []
        local_hover: Dict[Any, str] = {}
        local_max = 0.0
        for full_idx, comps in self.by_full.items():
            for rec in comps.get(component, []):
                if rec["comp_idx"] != comp_idx:
                    continue
                E_full = rec["E"] + rec["dE"] + self.fermi_shift
                ov = rec["ov"]
                line = ax.vlines(
                    E_full, 0, ov, color=col,
                    linewidth=self.cfg.lw_stick, zorder=zorder,
                )
                local_artists.append(line)
                local_hover[line] = (
                    f"full_idx {full_idx}\n"
                    f"E_full {E_full:+.4f} eV\n"
                    f"ov {ov:.5f}"
                )
                if ov > local_max:
                    local_max = ov
        return local_artists, local_hover, local_max

    def _draw_marker(self, ax, comp_idx, E_shifted, base_rgb, zorder, label_text):
        col = tuple(self._dark_rgb_from(base_rgb))
        line = ax.vlines(
            E_shifted, 0, 1.0, color=col,
            linewidth=self.cfg.lw_stick + 1.0, zorder=zorder,
        )
        return line, label_text

    def _draw_reflines(self, ax, fermi_shifted_val):
        if self.cfg.show_fermi_line:
            ax.axvline(
                0.0, color=self.cfg.fermi_line_color,
                linestyle=self.cfg.fermi_line_style, alpha=0.7, zorder=10,
            )
        ax.axvline(
            fermi_shifted_val, color="red",
            linestyle="--", alpha=0.9, zorder=10,
        )
        if self.v_shifted is not None:
            ax.axvline(
                self.v_shifted, color="black",
                linestyle="-", linewidth=3.5, zorder=10,
            )

    def _finalize_axis(self, ax, max_ov, title, set_xlabel, set_ylabel_flag):
        top = max_ov if max_ov > 0 else 1.0
        ax.set_ylim(0, top)
        if self.cfg.energy_range:
            ax.set_xlim(self.cfg.energy_range)
        else:
            if self.shared_e_min is not None:
                ax.set_xlim(left=self.shared_e_min)
            if self.v_shifted is not None:
                ax.set_xlim(right=self.v_shifted)
        if set_ylabel_flag:
            ax.set_ylabel(self.cfg.ylabel)
        if title is not None:
            ax.set_title(title)
        if set_xlabel:
            ax.set_xlabel(self.cfg.xlabel)

    def _attach_cursor(self, ax, artists, hover_map):
        if self.cfg.annotate_on_hover and HAS_MPLCURSORS and artists:
            cur = mplcursors.cursor(artists, hover=True)
            cur.enabled = False
            self.cursor_by_axes[ax] = cur

            @cur.connect("add")
            def _on_add(sel, hmap=hover_map):
                txt = hmap.get(sel.artist)
                if txt:
                    sel.annotation.set_text(txt)

    # ==================================================================
    # UI build (mirrors simulator._build_ui: clf, checkboxes, layout, wire)
    # ==================================================================
    def _build_ui(self):
        self.fig.clf()
        self.cursor_by_axes = {}

        # CheckButtons in lower-left, dedicated axes region (simulator pattern)
        self.chk = CheckButtons(
            plt.axes([0.01, 0.06, 0.13, 0.10]),
            ['Vertical', 'Horizontal'],
            [self.collapse_vertical, self.collapse_horizontal],
        )
        self.chk.on_clicked(self._on_toggle)

        # Segmentation sub-toggle: indented under Horizontal (sub-bullet
        # style), created only when Horizontal is on so it is visible only
        # then.
        if self.collapse_horizontal:
            self.chk_seg = CheckButtons(
                plt.axes([0.04, 0.01, 0.13, 0.045]),
                ['Segmentation'],
                [self.collapse_segmentation],
            )
            self.chk_seg.on_clicked(self._on_toggle)
        else:
            self.chk_seg = None

        # Reserve room at the bottom for the checkbox strip
        self.fig.subplots_adjust(bottom=0.24)

        self._draw_layout()

        # Figure-level click handler for hover enable/disable (per axis)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self.fig.canvas.draw_idle()

    def _on_toggle(self, label):
        states = self.chk.get_status()
        self.collapse_vertical, self.collapse_horizontal = states[0], states[1]
        if self.chk_seg is not None:
            self.collapse_segmentation = self.chk_seg.get_status()[0]
        self._build_ui()

    def _on_click(self, event):
        if not HAS_MPLCURSORS:
            return
        if event.inaxes is None:
            return
        cur = self.cursor_by_axes.get(event.inaxes)
        if cur is None:
            return
        if event.button == 1:
            if not cur.enabled:
                cur.enabled = True
        elif event.button == 3:
            for sel in list(cur.selections):
                cur.remove_selection(sel)
            cur.enabled = False

    # ==================================================================
    # Per-mode layout drawing
    #
    # Two independent collapses:
    #   vertical   -> collapses the ROW (band-index) dimension: n_rows = 1
    #   horizontal -> collapses the COLUMN (component) dimension: n_cols = 1
    # Neither toggle is aware of the other. The grid is
    # n_rows x n_cols; each cell is drawn by _draw_cell, which receives the
    # band indices and components that fall into that cell. Segmentation only
    # affects the horizontal-on drawing path and is independent of vertical.
    # ==================================================================
    def _draw_layout(self):
        R = len(self.comp_indices)
        C = len(self.components)
        components = self.components
        comp_indices = self.comp_indices

        horizontal_on = self.collapse_horizontal
        vertical_on = self.collapse_vertical
        segmentation_on = self.collapse_segmentation

        n_rows = 1 if vertical_on else R
        n_cols = 1 if horizontal_on else C
        axes = self.fig.subplots(n_rows, n_cols, sharex=True, squeeze=False)

        for row_slot in range(n_rows):
            cell_band_indices = (comp_indices if vertical_on
                                 else [comp_indices[row_slot]])
            for col_slot in range(n_cols):
                cell_components = (components if horizontal_on
                                   else [components[col_slot]])
                ax = axes[row_slot][col_slot]
                self._draw_cell(
                    ax, cell_band_indices, cell_components,
                    horizontal_on, vertical_on, segmentation_on,
                    set_xlabel=(row_slot == n_rows - 1),
                    set_ylabel_flag=(col_slot == 0),
                )

    # ==================================================================
    # Single-cell drawing. cell_band_indices x cell_components are the band
    # indices / components that land on this one axis. All color and z-order
    # decisions go through the frozen helpers.
    # ==================================================================
    def _draw_cell(self, ax, cell_band_indices, cell_components,
                   horizontal_on, vertical_on, segmentation_on,
                   set_xlabel, set_ylabel_flag):
        C = len(self.components)
        artists: List[Any] = []
        hover: Dict[Any, str] = {}
        axis_max = 0.0

        for comp_idx in cell_band_indices:
            zorder = self._spread_z(comp_idx)

            if horizontal_on:
                # ---- sum components mapping to the same full_idx ----
                # contrib[full_idx] = (E_full, [(j, component, ov), ...])
                contrib: Dict[int, Tuple[float, List[Tuple[int, str, float]]]] = {}
                for full_idx, comps in self.by_full.items():
                    for j, component in enumerate(cell_components):
                        for rec in comps.get(component, []):
                            if rec["comp_idx"] != comp_idx:
                                continue
                            E_full = rec["E"] + rec["dE"] + self.fermi_shift
                            ov = rec["ov"]
                            if full_idx not in contrib:
                                contrib[full_idx] = (E_full, [])
                            contrib[full_idx][1].append((j, component, ov))

                for full_idx, (E_full, parts) in contrib.items():
                    total = sum(ov for _, _, ov in parts)
                    if segmentation_on:
                        # darkest (smallest j) on bottom, lightest on top
                        parts_sorted = sorted(parts, key=lambda t: t[0])
                        y_bot = 0.0
                        for j, component, ov in parts_sorted:
                            y_top = y_bot + ov
                            base = self._base_rgb_for(component, comp_idx)
                            col = tuple(self._light_rgb_from(
                                base, self._light_f_for(j, C)))
                            line = ax.vlines(
                                E_full, y_bot, y_top, color=col,
                                linewidth=self.cfg.lw_stick, zorder=zorder,
                            )
                            artists.append(line)
                            hover[line] = (
                                f"{component} band {comp_idx}\n"
                                f"full_idx {full_idx}\n"
                                f"E_full {E_full:+.4f} eV\n"
                                f"ov {ov:.5f}"
                            )
                            y_bot = y_top
                    else:
                        base = self._base_rgb_for(cell_components[0], comp_idx) if cell_components else np.array([0.4, 0.4, 0.4])
                        col = tuple(self._light_rgb_from(
                            base, self.darken_factor / 2.0))
                        line = ax.vlines(
                            E_full, 0, total, color=col,
                            linewidth=self.cfg.lw_stick, zorder=zorder,
                        )
                        artists.append(line)
                        hover[line] = (
                            f"full_idx {full_idx}\n"
                            f"E_full {E_full:+.4f} eV\n"
                            f"ov {total:.5f}"
                        )
                    if total > axis_max:
                        axis_max = total
            else:
                # ---- horizontal off: cell holds exactly one component ----
                component = cell_components[0]
                sa, sh, smax = self._draw_spread_bars(
                    ax, component, comp_idx, self.darken_factor / 2.0,
                    zorder)
                artists += sa
                hover.update(sh)
                if smax > axis_max:
                    axis_max = smax

            # ---- marker for this band index ----
            if horizontal_on:
                E_m = self._avg_marker_E(cell_components, comp_idx)
                base = self._base_rgb_for(cell_components[0], comp_idx) if cell_components else np.array([0.4, 0.4, 0.4])
                marker_label = f"band {comp_idx}\nE {E_m:+.4f} eV (mean of {C})" if E_m is not None else None
            else:
                component = cell_components[0]
                E_m = self._avg_marker_E([component], comp_idx)
                base = self._base_rgb_for(component, comp_idx)
                marker_label = f"{component} band {comp_idx}\nE {E_m:+.4f} eV" if E_m is not None else None
            if E_m is not None:
                mline, mh = self._draw_marker(
                    ax, comp_idx, E_m, base, self._marker_z(comp_idx),
                    marker_label)
                artists.append(mline)
                hover[mline] = mh

        # ---- reflines: averaged when components are combined, else single ----
        if horizontal_on:
            self._draw_reflines(ax, self._avg_fermi_shifted(cell_components))
        else:
            self._draw_reflines(ax, self._comp_fermi_shifted(cell_components[0]))

        # ---- title: reproduce the exact text for this collapse combination ----
        if vertical_on and horizontal_on:
            title = "Spread \u2014 all components, all bands"
        elif horizontal_on:
            comp_idx = cell_band_indices[0]
            E_m = self._avg_marker_E(cell_components, comp_idx)
            title = (f"band {comp_idx}, E = {E_m:+.4f} eV" if E_m is not None
                     else f"band {comp_idx} \u2014 not found")
        elif vertical_on:
            title = f"{cell_components[0]}"
        else:
            component = cell_components[0]
            comp_idx = cell_band_indices[0]
            E_m = self._avg_marker_E([component], comp_idx)
            title = (f"{component} band {comp_idx}, E = {E_m:+.4f} eV"
                     if E_m is not None
                     else f"{component} band {comp_idx} \u2014 not found")

        self._finalize_axis(
            ax, axis_max, title,
            set_xlabel=set_xlabel, set_ylabel_flag=set_ylabel_flag)
        self._attach_cursor(ax, artists, hover)


def plot_spread(
    output_dir: str,
    components: List[str],
    comp_indices: List[int],
    collapse_vertical: bool = False,
    collapse_horizontal: bool = False,
    darken_factor: float = 0.4,
    cfg: Optional[PlotConfig] = None,
) -> Tuple[plt.Figure, "SpreadPlotter"]:
    """
    Build the interactive spread-plot figure. The two CheckButtons in the
    lower-left of the figure toggle the Vertical / Horizontal collapse live;
    collapse_vertical / collapse_horizontal set the initial checkbox state.

    Parameters
    ----------
    output_dir : str
        Directory containing band_matches_rectangular.txt and
        band_matches_rectangular_all.txt (the full_dir from the matcher).
    components : list of str
        Component labels. The same comp_indices are used for every component.
    comp_indices : list of int
        1-based band indices, shared across all components.
    collapse_vertical, collapse_horizontal : bool
        Initial state of the in-figure toggles.
    darken_factor : float
        Controls the light/dark split and the per-component lightness offset
        (see SpreadPlotter).
    cfg : PlotConfig or None
        Plot configuration. Uses defaults if None.

    Returns
    -------
    fig, sp
        The matplotlib Figure and the SpreadPlotter instance (kept so the
        widgets stay responsive; do not let it be garbage-collected).
    """
    sp = SpreadPlotter(
        output_dir=output_dir,
        components=components,
        comp_indices=comp_indices,
        collapse_vertical=collapse_vertical,
        collapse_horizontal=collapse_horizontal,
        darken_factor=darken_factor,
        cfg=cfg,
    )
    return sp.fig, sp


# ======================================================================
if __name__ == "__main__":

    import sys, types
    _pkg_dir = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(_pkg_dir, ".."))
    # Stub package so relative imports resolve without __init__.py
    _pkg = types.ModuleType("matcher")
    _pkg.__path__ = [_pkg_dir]
    sys.modules.setdefault("matcher", _pkg)
    from matcher.vacupot_plotter import (
        RectAEPAWColorPlotter,
        PlotConfig,
        _read_ov_all,
    )

    # ---- Directory containing matcher output files ----
    output_dir = r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551/SOC'

    # ---- Which components and which bands to plot ----
    components = ["NHC_c1m1", "NHC_c2m1", "NHC_c3m1", "NHC_c4m1"]
    comp_indices = [76, 78, 80, 81]      # 1-based band indices, shared

    # ---- Initial toggle state (flip live with the in-figure checkboxes) ----
    collapse_vertical = False
    collapse_horizontal = False

    # ---- Visual tuning ----
    darken_factor = 0.4                  # 0 = base color, 1 = black

    cfg = PlotConfig(
        cmap_name_simple="managua_r",
        cmap_name_metal="vanimo_r",
        energy_range=None,
        shared_molecule_color=True,
        min_total_mol_wspan=0.02,

        power_simple_neg=0.25,
        power_simple_pos=0.75,
        power_metal_neg=0.075,
        power_metal_pos=0.075,
        power_residual=0.5,

        show_fermi_line=True,
        fermi_line_style=":",
        fermi_line_color="k",
        show_local_fermi=True,

        annotate_on_hover=True,
    )

    fig, sp = plot_spread(
        output_dir=output_dir,
        components=components,
        comp_indices=comp_indices,
        collapse_vertical=collapse_vertical,
        collapse_horizontal=collapse_horizontal,
        darken_factor=darken_factor,
        cfg=cfg,
    )
    plt.show()