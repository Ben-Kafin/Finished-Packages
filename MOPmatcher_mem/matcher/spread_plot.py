# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 2026

@author: Benjamin Kafin

Standalone spread plotter.
For a given component label and list of band indices, produces a vertically-stacked
figure showing how each component MO distributes its squared overlap across the
full-system Bloch states.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

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


def plot_spread(
    output_dir: str,
    comp_label: str,
    comp_indices: List[int],
    darken_factor: float = 0.4,
    cfg: Optional[PlotConfig] = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Parameters
    ----------
    output_dir : str
        Directory containing band_matches_rectangular.txt and
        band_matches_rectangular_all.txt (the full_dir from the matcher).
    comp_label : str
        Component label (e.g. a molecule folder name or "metal").
    comp_indices : list of int
        1-based band indices within that component. Each gets its own subplot.
    darken_factor : float
        Fraction to blend the MO marker bar toward black (0 = base color, 1 = black).
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
    # 1. Load data + build colors through _prepare (same SOC-correct path
    #    as the main vacupot_plotter and spread_collapse). Reads from
    #    matcher_results.h5 when present, else the .txt outputs.
    #    write_summaries=False -> no behavior files written here.
    # ------------------------------------------------------------------
    main_path = os.path.join(output_dir, "band_matches_rectangular.txt")
    P = plotter._prepare(main_path, bonding=False, write_summaries=False)

    by_full = P.by_full
    comp_pairs = P.comp_pairs
    known_fermis = P.known_fermis
    v_aligned = P.v_aligned
    component_colors = P.component_colors

    # ------------------------------------------------------------------
    # 2. Energy shift: put full-system Fermi at zero
    # ------------------------------------------------------------------
    fermi_full = known_fermis.get("full", 0.0)
    fermi_shift = -fermi_full
    mol_fermi_shifted = known_fermis.get(comp_label, 0.0) + fermi_shift

    v_shifted = (v_aligned + fermi_shift) if v_aligned is not None else None

    # Default x-min: lowest energy present in both systems
    comp_energies = [E for idx, E in comp_pairs.get(comp_label, [])]
    full_energies = [rec["E"] + rec["dE"]
                     for comps in by_full.values()
                     for recs in comps.values()
                     for rec in recs]
    if comp_energies and full_energies:
        shared_e_min = max(min(comp_energies), min(full_energies)) + fermi_shift
    else:
        shared_e_min = None

    # ------------------------------------------------------------------
    # 3. Retrieve component-state energies from comp_pairs
    # ------------------------------------------------------------------
    comp_energy_map: Dict[int, float] = {}
    for idx, E in comp_pairs.get(comp_label, []):
        comp_energy_map[idx] = E

    # ------------------------------------------------------------------
    # 5. Build figure: one subplot per requested band index
    # ------------------------------------------------------------------
    n_panels = len(comp_indices)
    fig_h = max(3.0, 1.8 * n_panels)
    fig, axes = plt.subplots(
        n_panels, 1,
        sharex=True,
        figsize=(cfg.figsize[0], fig_h),
        squeeze=False,
    )
    axes = [axes[i, 0] for i in range(n_panels)]

    cursor_by_axes: Dict[Any, Any] = {}

    for panel_idx, comp_idx in enumerate(comp_indices):
        ax = axes[panel_idx]

        # --- Colors: split darken_factor into light half / dark half ---
        base_color = component_colors.get(comp_label, {}).get(
            comp_idx, (0.4, 0.4, 0.4, 1.0)
        )
        base_rgb = np.array(base_color[:3])
        half = darken_factor / 2.0
        light_rgb = base_rgb + (1.0 - base_rgb) * half   # toward white
        dark_rgb = base_rgb * (1.0 - half)                # toward black

        # --- Component state energy (shifted) ---
        E_comp_raw = comp_energy_map.get(comp_idx)
        if E_comp_raw is None:
            ax.set_title(f"{comp_label} band {comp_idx} — not found")
            continue
        E_comp = E_comp_raw + fermi_shift

        # --- Gather every full-state match ---
        match_bars: List[Dict[str, Any]] = []
        for full_idx, comps in by_full.items():
            for rec in comps.get(comp_label, []):
                if rec["comp_idx"] == comp_idx:
                    E_full = rec["E"] + rec["dE"] + fermi_shift
                    match_bars.append({
                        "E_full": E_full,
                        "ov": rec["ov"],
                        "full_idx": full_idx,
                    })

        # --- Determine heights ---
        max_ov = max((mb["ov"] for mb in match_bars), default=0.0)
        mo_height = 1.1 * max_ov if max_ov > 0 else 1.0

        # --- Draw match bars (lightened) ---
        artists = []
        hover_map = {}

        for mb in match_bars:
            ov = mb["ov"]
            line = ax.vlines(
                mb["E_full"], 0, ov,
                color=tuple(light_rgb), linewidth=cfg.lw_stick,
            )
            artists.append(line)
            hover_map[line] = (
                f"full_idx {mb['full_idx']}\n"
                f"E_full {mb['E_full']:+.4f} eV\n"
                f"ov {ov:.5f}"
            )

        # --- Draw the MO marker bar: 1.1x tallest match, on top ---
        mo_line = ax.vlines(
            E_comp, 0, mo_height,
            color=tuple(dark_rgb), linewidth=cfg.lw_stick + 1.0,
        )
        artists.append(mo_line)
        hover_map[mo_line] = (
            f"{comp_label} band {comp_idx}\n"
            f"E {E_comp:+.4f} eV"
        )

        # --- Reference lines ---
        if cfg.show_fermi_line:
            ax.axvline(
                0.0, color=cfg.fermi_line_color,
                linestyle=cfg.fermi_line_style, alpha=0.7,
            )
        ax.axvline(
            mol_fermi_shifted, color="red",
            linestyle="--", alpha=0.9,
        )
        if v_shifted is not None:
            ax.axvline(
                v_shifted, color="black",
                linestyle="-", linewidth=3.5,
            )

        # --- Axis limits & labels ---
        ax.set_ylim(0, mo_height)
        if cfg.energy_range:
            ax.set_xlim(cfg.energy_range)
        else:
            if shared_e_min is not None:
                ax.set_xlim(left=shared_e_min)
            if v_shifted is not None:
                ax.set_xlim(right=v_shifted)
        ax.set_ylabel(cfg.ylabel)
        ax.set_title(
            f"{comp_label} band {comp_idx},  E = {E_comp:+.4f} eV"
        )

        # --- Hover annotations ---
        if cfg.annotate_on_hover and HAS_MPLCURSORS and artists:
            cur = mplcursors.cursor(artists, hover=True); cur.enabled = False
            cursor_by_axes[ax] = cur

            @cur.connect("add")
            def _on_add(sel, hmap=hover_map):
                txt = hmap.get(sel.artist)
                if txt:
                    sel.annotation.set_text(txt)

    # --- Shared x-label on bottom axis ---
    axes[-1].set_xlabel(cfg.xlabel)

    if HAS_MPLCURSORS:
        def _on_click(event, _map=cursor_by_axes):
            if event.inaxes is None: return
            cur = _map.get(event.inaxes)
            if cur is None: return
            if event.button == 1:
                if not cur.enabled:
                    cur.enabled = True
            elif event.button == 3:
                for sel in list(cur.selections):
                    cur.remove_selection(sel)
                cur.enabled = False
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
    from matcher.vacupot_plotter import (
        RectAEPAWColorPlotter,
        PlotConfig,
        _read_ov_all,
    )

    # ---- Directory containing matcher output files ----
    output_dir = r'C:/Users/Benjamin Kafin/Documents/VASP/lone/fcc/SOC/kp551'

    # ---- Which component and which bands to plot ----
    comp_label = "clean_kp551"
    comp_indices = [76,78,80,81]          # 1-based band indices

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

    fig, axes = plot_spread(
        output_dir=output_dir,
        comp_label=comp_label,
        comp_indices=comp_indices,
        darken_factor=darken_factor,
        cfg=cfg,
    )
    plt.show()