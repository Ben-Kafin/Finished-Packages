# -*- coding: utf-8 -*-
"""
combined_vacupot_plotter.py

Combine two MOPMatcher runs onto a single global energy axis and plot them together.

Scenario
--------
Two MOPMatcher runs share the same (physically identical) molecule component(s):

  * "complex" run : full system = the NHC + gold-adatom complex;
                    components   = the molecule(s) + a lone gold atom ("metal").
  * "full" run    : full system = that complex adsorbed on a surface;
                    components   = the same molecule(s) + the bulk surface
                                   with the adatom ("metal").

Alignment
---------
Each run's matcher already aligned all of its own subsystems to that run's own
metal vacuum and slid them so that run's chosen zero-reference Fermi sits at 0.
Each run's ``band_matches_rectangular.txt`` records the single aligned vacuum
position in its ``# VACUUMS: aligned=`` header, on that run's own (already
processed) energy scale.

To put both runs on one axis we vacuum-align them: shift every energy of the
*complex* run by

    delta = aligned_full - aligned_complex

so the complex run's vacuum line coincides with the full run's vacuum line. The
*full* run is left untouched; because it was processed with
``zero_reference="full"`` its full-system Fermi already sits at x = 0, which
therefore becomes the global zero for every row.

Only the two runs' ``band_matches_rectangular.txt`` files (plus the sibling
``band_matches_rectangular_all.txt`` and each directory's ``INCAR`` for SOC
auto-detection) are read -- exactly the files the single-run plotter reads. No
LOCPOT / vacuum_potential.h5 is needed: the relative shift is fully determined
by the two ``aligned=`` header values.

Subplot order (top -> bottom)
-----------------------------
  1. molecule row(s)            <- FULL run molecule component(s), no shift
  2. Lone gold atom             <- COMPLEX run "metal" component, shifted +delta
  3. Complex                    <- COMPLEX run full system,       shifted +delta
  4. Bulk surface and adatom    <- FULL run "metal" component,    no shift
  5. Full                       <- FULL run full system,          no shift

The molecule is shown once (from the FULL run, per the molecule being identical
across runs). Its hover is a two-column ("twice as wide") annotation: the left
column is the COMPLEX run's classification of that molecule band evaluated on
the shifted (global) energy axis, the right column is the FULL run's
classification. All color-selection, classification, stick-drawing and cursor
logic is the single-run plotter's, reused unchanged via RectAEPAWColorPlotter.

Re-classifying the complex run on shifted energies
--------------------------------------------------
The classifier's numeric outputs (mean_shift, variance, E_plus/zero/minus) are
functions of dE = E_full - E_comp, which is intra-run and invariant under a
rigid shift. The *bonding* classification, however, clamps shifts that cross the
Fermi level (0 eV) using the absolute sign of E_comp / E_full. After the complex
run is shifted by +delta its physical zero becomes the global Fermi, so the
bonding clamp must be evaluated against the global zero. We therefore re-run the
classifier (StateBehaviorClassifier.classify_maps, no file writes) on a copy of
the complex run's records whose component energies have been shifted by +delta;
energy differences dE are left unchanged. This is the only classifier "re-run";
it produces values identical to a single-run plot for the full run, and the
correct global-axis values for the complex run.
"""
from __future__ import annotations

import copy
import os
from collections import defaultdict
from typing import Optional, Tuple

import matplotlib.pyplot as plt

from .vacupot_plotter import RectAEPAWColorPlotter, PlotConfig


def _resolve_main_path(path: str) -> str:
    """Accept either a run directory or a direct path to that run's
    ``band_matches_rectangular.txt``.

    The matcher writes ``band_matches_rectangular.txt`` (and the sibling
    ``band_matches_rectangular_all.txt``) into each run's ``full_dir``. If a
    directory is given, the main-file name is appended; a file path is returned
    unchanged. The ``_all`` file and ``INCAR`` are subsequently located in the
    same directory by ``_prepare``."""
    if os.path.isdir(path):
        return os.path.join(path, "band_matches_rectangular.txt")
    return path


def _shift_by_full(by_full, delta):
    """Return a copy of a run's ``by_full`` mapping with every record's absolute
    component energy ``E`` shifted by ``delta``. The energy difference ``dE`` is
    intentionally left unchanged (it is shift-invariant), so that the bonding
    classifier's zero-crossing clamp is evaluated against the global Fermi while
    the shift/variance statistics are unaffected."""
    shifted = defaultdict(lambda: defaultdict(list))
    for full_idx, comps in by_full.items():
        for comp_label, recs in comps.items():
            for r in recs:
                r2 = copy.copy(r)
                r2["E"] = r.get("E", 0.0) + delta
                shifted[full_idx][comp_label].append(r2)
    return shifted


def plot_combined_runs(
    complex_main_path: str,
    full_main_path: str,
    cfg: Optional[PlotConfig] = None,
    bonding: bool = True,
    title_molecule: Optional[str] = None,
    title_lone_metal: str = "Lone gold atom",
    title_complex: str = "Complex",
    title_bulk: str = "Bulk surface and adatom",
    title_full: str = "Full",
) -> Tuple[plt.Figure, list]:
    """
    Plot two vacuum-aligned MOPMatcher runs on one global energy axis.

    Parameters
    ----------
    complex_main_path, full_main_path
        Either each run's directory (its ``full_dir``) or a direct path to that
        run's ``band_matches_rectangular.txt``. When a directory is given, the
        main file is located inside it; the sibling
        ``band_matches_rectangular_all.txt`` and ``INCAR`` are read from the same
        directory, as in the single-run plotter.
    cfg
        PlotConfig controlling all color / line / cursor behavior. Reused
        unchanged; defaults to ``PlotConfig()``.
    bonding
        Selects bonding vs. normal classification for the hovers, exactly as in
        ``RectAEPAWColorPlotter.plot``.
    title_molecule
        Title for the molecule row(s). ``None`` -> use each molecule's own
        component label (matching single-run behavior).
    title_lone_metal, title_complex, title_bulk, title_full
        Titles for the remaining four rows.

    Returns
    -------
    (fig, axes)
    """
    cfg = cfg or PlotConfig()
    plotter = RectAEPAWColorPlotter(cfg)
    plotter._cursor_by_axes.clear()

    # Accept either a run directory or a direct band_matches_rectangular.txt path.
    complex_main_path = _resolve_main_path(complex_main_path)
    full_main_path = _resolve_main_path(full_main_path)

    # Build each run's color/classification bundle WITHOUT writing summary files.
    C = plotter._prepare(complex_main_path, bonding, write_summaries=False)
    F = plotter._prepare(full_main_path, bonding, write_summaries=False)

    if C.v_aligned is None or F.v_aligned is None:
        raise ValueError(
            "Both runs must record a '# VACUUMS: aligned=' header to be aligned. "
            f"complex aligned={C.v_aligned!r}, full aligned={F.v_aligned!r}."
        )

    # Vacuum-align: shift every COMPLEX-run energy so its vacuum line lands on
    # the FULL run's vacuum line. The FULL run is the global-zero reference.
    delta = F.v_aligned - C.v_aligned

    # Re-classify the complex run on the shifted (global) energy axis so the
    # bonding clamp is evaluated against the global Fermi. classify_maps writes
    # no files. Select bonding vs normal consistently with `bonding`.
    c_norm_shift, c_bond_shift = C.classifier.classify_maps(_shift_by_full(C.by_full, delta))
    C_shift_maps = c_bond_shift if bonding else c_norm_shift

    if len(C.mol_labels) != len(F.mol_labels):
        raise ValueError(
            "The two runs expose different numbers of molecule components "
            f"(complex: {C.mol_labels}, full: {F.mol_labels}). The molecules are "
            "expected to be identical and in the same order so they can be paired "
            "for the two-column molecule hover."
        )

    mol_labels_F = F.mol_labels
    mol_labels_C = C.mol_labels
    n_rows = len(mol_labels_F) + 4

    figsize = (cfg.figsize[0], max(3, 1.5 * n_rows))
    fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=figsize)
    axes = list(axes) if n_rows > 1 else [axes]

    r = 0

    # 1) molecule row(s): FULL run sticks/colors, no shift. Two-column hover
    #    (complex shifted on the left, full on the right). Paired by index.
    for i, lbl_F in enumerate(mol_labels_F):
        lbl_C = mol_labels_C[i]
        plotter._draw_component_axis(
            axes[r], lbl_F,
            F.comp_pairs.get(lbl_F, []),
            F.component_colors.get(lbl_F, {}),
            F.component_class_maps.get(lbl_F, {}),
            F.known_fermis.get(lbl_F), F.v_aligned,
            shift=0.0,
            title=(title_molecule if title_molecule is not None else lbl_F),
            key=f"F:{lbl_F}",
            dual={
                "class_map": C_shift_maps.get(lbl_C, {}),
                "left_title": "complex (shifted)",
                "right_title": "full",
            },
        )
        r += 1

    # 2) Lone gold atom: COMPLEX run "metal" component, shifted +delta.
    plotter._draw_component_axis(
        axes[r], "metal",
        C.comp_pairs.get("metal", []),
        C.component_colors.get("metal", {}),
        C_shift_maps.get("metal", {}),
        C.known_fermis.get("metal"), C.v_aligned,
        shift=delta, title=title_lone_metal, key="C:metal",
    )
    r += 1

    # 3) Complex: COMPLEX run full system, shifted +delta.
    plotter._draw_full_axis(
        axes[r], C.rows, C.by_full, C.comp_iter_order, C.component_colors,
        C.classifier, C.mol_labels, C.known_fermis, C.v_aligned,
        shift=delta, title=title_complex, xlabel=False,
    )
    r += 1

    # 4) Bulk surface and adatom: FULL run "metal" component, no shift.
    plotter._draw_component_axis(
        axes[r], "metal",
        F.comp_pairs.get("metal", []),
        F.component_colors.get("metal", {}),
        F.component_class_maps.get("metal", {}),
        F.known_fermis.get("metal"), F.v_aligned,
        shift=0.0, title=title_bulk, key="F:metal",
    )
    r += 1

    # 5) Full: FULL run full system, no shift. Bottom row carries the x-label.
    plotter._draw_full_axis(
        axes[r], F.rows, F.by_full, F.comp_iter_order, F.component_colors,
        F.classifier, F.mol_labels, F.known_fermis, F.v_aligned,
        shift=0.0, title=title_full, xlabel=True,
    )
    r += 1

    if cfg.energy_range:
        axes[-1].set_xlim(cfg.energy_range)

    plotter._wire_global_click(fig)
    fig.tight_layout()
    return fig, axes