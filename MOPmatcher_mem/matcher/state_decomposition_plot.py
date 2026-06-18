# -*- coding: utf-8 -*-
"""
state_decomposition_plot.py

@author: Benjamin Kafin

For a given MOPMatcher run, plots how each full-system Bloch state decomposes into
its contributing component states. One panel for the metal component, one
panel for the molecule(s); each full state appears as a vertical stack at
its full-system energy. Segment height = that match's individual overlap
probability (ov); segment color = the canonical MOPMatcher color of the
contributing component state (same color the main plot assigns it on its
component axis). Segments in one state's stack are sorted by comp_idx
ascending (with comp_label as the tiebreaker for the multi-molecule case),
the literal key into component_colors[comp_label] -- invariant under any
energy shift.

The plotted set is selected by a per-full-state cumulative-tail threshold
on the pool across BOTH panels: for each full state, sort all contributions
ascending by ov and drop from the smallest upward while the cumulative
dropped probability mass remains <= sum_threshold; stop the first time
including the next ov would push the dropped mass past sum_threshold.
"""
from __future__ import annotations
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import mplcursors
    HAS_MPLCURSORS = True
except Exception:
    HAS_MPLCURSORS = False

if __name__ != "__main__":
    from .vacupot_plotter import RectAEPAWColorPlotter, PlotConfig


def plot_state_decomposition(
    output_dir: str,
    sum_threshold: float = 0.1,
    cfg: Optional[PlotConfig] = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Parameters
    ----------
    output_dir : str
        Directory containing band_matches_rectangular.txt and
        band_matches_rectangular_all.txt (the full_dir from the matcher).
    sum_threshold : float
        Per-full-state cumulative-tail threshold (see module docstring).
    cfg : PlotConfig or None
        Plot configuration. Uses defaults if None.

    Returns
    -------
    fig, axes
    """
    if cfg is None:
        cfg = PlotConfig()

    plotter = RectAEPAWColorPlotter(cfg)

    # ------------------------------------------------------------------
    # 1. Load via the main plotter's _prepare so colors / Fermis /
    #    vacuum / SOC pairing / shared_molecule_color all come out
    #    identical to the main MOPMatcher plot.
    # ------------------------------------------------------------------
    main_path = os.path.join(output_dir, "band_matches_rectangular.txt")
    P = plotter._prepare(main_path, bonding=False, write_summaries=False)

    by_full          = P.by_full
    known_fermis     = P.known_fermis
    v_aligned        = P.v_aligned
    component_colors = P.component_colors
    mol_labels       = P.mol_labels
    metal_present    = P.metal_present

    # ------------------------------------------------------------------
    # 2. Energy shift: put full-system Fermi at zero (same convention
    #    as spread_plot.py).
    # ------------------------------------------------------------------
    fermi_shift = -float(known_fermis.get("full", 0.0))
    v_shifted = (v_aligned + fermi_shift) if v_aligned is not None else None

    # ------------------------------------------------------------------
    # 3. Panel definitions. Order top -> bottom: metal, molecule(s).
    # ------------------------------------------------------------------
    panels: List[Tuple[str, List[str]]] = []
    if metal_present:
        panels.append(("metal", ["metal"]))
    if mol_labels:
        panels.append(("molecule", list(mol_labels)))

    if not panels:
        raise RuntimeError(
            "No components found in matcher output; cannot plot."
        )

    n_panels = len(panels)

    label_to_grp: Dict[str, int] = {}
    for gi, (_, lbls) in enumerate(panels):
        for lbl in lbls:
            label_to_grp[lbl] = gi
    figure_labels = list(label_to_grp.keys())

    # ------------------------------------------------------------------
    # 4. Per-full-state thresholding on the pool across ALL panels.
    #    kept[grp_idx][full_idx] -> list of
    #    (comp_label, comp_idx, ov, E_comp, dE) for that panel.
    # ------------------------------------------------------------------
    kept: List[Dict[int, List[Tuple[str, int, float, float, float]]]] = [
        defaultdict(list) for _ in panels
    ]

    for full_idx, comps in by_full.items():
        # Pool every contribution across panels for this full state.
        # tuple = (grp_idx, comp_label, comp_idx, ov, E_comp, dE)
        pool: List[Tuple[int, str, int, float, float, float]] = []
        for lbl in figure_labels:
            gi = label_to_grp[lbl]
            for r in comps.get(lbl, []):
                pool.append((
                    gi, lbl, int(r["comp_idx"]),
                    float(r["ov"]), float(r["E"]), float(r["dE"]),
                ))

        # Drop from smallest ov upward while cumulative dropped mass
        # stays at or below sum_threshold.
        pool_sorted = sorted(pool, key=lambda t: t[3])
        dropped_sum = 0.0
        keep_start = 0
        for i, rec in enumerate(pool_sorted):
            ov_i = rec[3]
            if dropped_sum + ov_i <= sum_threshold:
                dropped_sum += ov_i
                keep_start = i + 1
            else:
                break

        for gi, lbl, comp_idx, ov, E_c, dE in pool_sorted[keep_start:]:
            kept[gi][full_idx].append((lbl, comp_idx, ov, E_c, dE))

    # ------------------------------------------------------------------
    # 5. Figure: one panel per (metal / molecule) row, sharex.
    # ------------------------------------------------------------------
    fig_h = max(3.0, 1.8 * n_panels)
    fig, axes_arr = plt.subplots(
        n_panels, 1, sharex=True,
        figsize=(cfg.figsize[0], fig_h),
        squeeze=False,
    )
    axes: List[plt.Axes] = [axes_arr[i, 0] for i in range(n_panels)]

    # ------------------------------------------------------------------
    # 6. Per-axis bookkeeping for the click-armed cursors.
    #    state_artists[ax][full_idx]  -> list of vlines artists
    #    hover_maps[ax][full_idx]     -> dict[artist -> hover text]
    #    state_extent[ax][full_idx]   -> (x_pos, y_total_top)
    # ------------------------------------------------------------------
    state_artists: Dict[Any, Dict[int, List[Any]]] = {}
    hover_maps:    Dict[Any, Dict[int, Dict[Any, str]]] = {}
    state_extent:  Dict[Any, Dict[int, Tuple[float, float]]] = {}

    # ------------------------------------------------------------------
    # 7. Draw each panel.
    # ------------------------------------------------------------------
    for panel_idx, (kind, lbls) in enumerate(panels):
        ax = axes[panel_idx]
        state_artists[ax] = defaultdict(list)
        hover_maps[ax]    = defaultdict(dict)
        state_extent[ax]  = {}

        max_total_h = 0.0

        for full_idx, recs in kept[panel_idx].items():
            if not recs:
                continue
            # Sort by comp_idx ascending; comp_label as deterministic
            # tiebreaker for the multi-molecule case.
            recs_sorted = sorted(recs, key=lambda t: (t[1], t[0]))

            # Recover E_full from any one record: E_full = E_comp + dE.
            # All recs for this full_idx share the same E_full.
            E_full = recs_sorted[0][3] + recs_sorted[0][4]
            x_pos  = E_full + fermi_shift

            y_bot = 0.0
            for lbl, comp_idx, ov, E_c, dE in recs_sorted:
                y_top = y_bot + ov
                color = component_colors.get(lbl, {}).get(
                    comp_idx, (0.4, 0.4, 0.4, 1.0)
                )
                line = ax.vlines(
                    x_pos, y_bot, y_top,
                    color=color, linewidth=cfg.lw_stick,
                )
                state_artists[ax][full_idx].append(line)
                hover_maps[ax][full_idx][line] = (
                    f"{lbl} band {comp_idx}\n"
                    f"E_comp {E_c + fermi_shift:+.4f} eV\n"
                    f"dE     {dE:+.4f} eV\n"
                    f"ov     {ov:.5f}\n"
                    f"full_idx {full_idx}"
                )
                y_bot = y_top

            state_extent[ax][full_idx] = (x_pos, y_bot)
            if y_bot > max_total_h:
                max_total_h = y_bot

        # --- Reference lines (consistent with vacupot_plotter axes) ---
        if cfg.show_fermi_line:
            ax.axvline(
                0.0,
                color=cfg.fermi_line_color,
                linestyle=cfg.fermi_line_style,
                alpha=0.7,
            )

        local_fermi_label = "metal" if kind == "metal" else (
            lbls[0] if lbls else None
        )
        if (cfg.show_local_fermi
                and local_fermi_label is not None
                and local_fermi_label in known_fermis):
            ax.axvline(
                known_fermis[local_fermi_label] + fermi_shift,
                color=cfg.local_fermi_color,
                linestyle=cfg.local_fermi_style,
                alpha=0.9,
                zorder=10,
            )

        if cfg.show_vacuum_line and v_shifted is not None:
            ax.axvline(
                v_shifted,
                color=cfg.vacuum_line_color,
                linewidth=cfg.vacuum_line_width,
                linestyle=cfg.vacuum_line_style,
                zorder=10,
            )

        # --- Axis limits / labels / titles ---
        if max_total_h > 0:
            ax.set_ylim(0, max_total_h * 1.05)
        if cfg.energy_range:
            ax.set_xlim(cfg.energy_range)

        ax.set_ylabel(cfg.ylabel)
        ax.set_title("metal" if kind == "metal" else ", ".join(lbls))

    axes[-1].set_xlabel(cfg.xlabel)

    # ------------------------------------------------------------------
    # 8. Click-armed per-full-state cursors.
    #    Left click on a state's stack -> disable every other state's
    #    cursor on that axis, enable this state's cursor. Right click
    #    on an axis -> clear/disable every state cursor on it.
    # ------------------------------------------------------------------
    state_cursors: Dict[Any, Dict[int, Any]] = {}

    if cfg.annotate_on_hover and HAS_MPLCURSORS:
        for ax in axes:
            state_cursors[ax] = {}
            by_state = state_artists.get(ax, {})
            for full_idx, arts in by_state.items():
                if not arts:
                    continue
                cur = mplcursors.cursor(arts, hover=True)
                cur.enabled = False
                hm = hover_maps[ax][full_idx]
                @cur.connect("add")
                def _on_add(sel, _hm=hm):
                    txt = _hm.get(sel.artist)
                    if txt:
                        sel.annotation.set_text(txt)
                state_cursors[ax][full_idx] = cur

    def _hit_full_idx(ax, ex, ey):
        info = state_extent.get(ax, {})
        if not info or ex is None or ey is None:
            return None
        # ~5-pixel x tolerance, in data units, derived per-axis.
        inv = ax.transData.inverted()
        p0 = inv.transform((0.0, 0.0))
        p1 = inv.transform((5.0, 0.0))
        x_tol = abs(p1[0] - p0[0])

        best = None
        best_dx = float("inf")
        for full_idx, (x_pos, y_top) in info.items():
            if 0.0 <= ey <= y_top and abs(ex - x_pos) <= x_tol:
                dx = abs(ex - x_pos)
                if dx < best_dx:
                    best_dx = dx
                    best = full_idx
        return best

    def _on_click(event):
        if event.inaxes is None:
            return
        ax = event.inaxes
        cursors = state_cursors.get(ax, {})
        if not cursors:
            return
        if event.button == 1:
            hit = _hit_full_idx(ax, event.xdata, event.ydata)
            if hit is None:
                return
            for fi, cur in cursors.items():
                if fi != hit:
                    for sel in list(cur.selections):
                        cur.remove_selection(sel)
                    cur.enabled = False
            cursors[hit].enabled = True
        elif event.button == 3:
            for cur in cursors.values():
                for sel in list(cur.selections):
                    cur.remove_selection(sel)
                cur.enabled = False

    if HAS_MPLCURSORS:
        fig.canvas.mpl_connect("button_press_event", _on_click)

    fig.tight_layout()
    return fig, axes


# ======================================================================
if __name__ == "__main__":

    import sys, types
    _pkg_dir = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(_pkg_dir, ".."))
    # Stub package so relative imports resolve without __init__.py
    _pkg = types.ModuleType("matcher")
    _pkg.__path__ = [_pkg_dir]
    sys.modules.setdefault("matcher", _pkg)
    from matcher.vacupot_plotter import RectAEPAWColorPlotter, PlotConfig

    # ---- Directory containing matcher output files ----
    output_dir = r'C:/Users/Benjamin Kafin/Documents/VASP/lone/fcc/SOC/kp551'

    # ---- Per-full-state cumulative-tail threshold ----
    sum_threshold = 0.01

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

        show_vacuum_line=True,
        vacuum_line_color="black",
        vacuum_line_width=3.5,
        vacuum_line_style="-",

        annotate_on_hover=True,
        soc_pair_thresh_eV=0.005,
    )

    fig, axes = plot_state_decomposition(
        output_dir=output_dir,
        sum_threshold=sum_threshold,
        cfg=cfg,
    )
    plt.show()
