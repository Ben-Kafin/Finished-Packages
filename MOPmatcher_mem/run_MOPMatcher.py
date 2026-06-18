# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:20:10 2026

@author: Benjamin Kafin
"""

# run_matcher.py
from matcher import run_match, PlotConfig, RectAEPAWColorPlotter
import matplotlib.pyplot as plt

run_kwargs = {
    # --- Builder settings ---
    "k_index": 1,                  # which k-point (1-based)
    "tol_map": 1e-3,               # atom-matching tolerance (Å)
    "check_species": True,         # enforce species matching
    "reuse_cached": True,          # skip builder if .h5 files exist

    # --- Band window: slice object or None per molecule ---
    "band_window_molecules": [slice(0, 82)],

    # --- Vacuum alignment ---
    "reuse_vac_cache": True,
    "curvature_tol": 5e-8,
    "dipole_threshold": 0.15,

    # --- Global zero reference ---
    "zero_reference": "full",      # "metal", "full", or a molecule folder name

    # --- Local Fermi-line mode per directory ---
    # False (default): draw that system's red dashed line at its reported/aligned
    #   Fermi energy. True: draw it exactly midway between that system's HOMO and
    #   LUMO instead. Only changes the drawn line (the '# FERMIS:' header value);
    #   energy alignment, color assignment, and matching/classification are
    #   unaffected. Re-run the matcher for changes to take effect; both the
    #   individual and combined plots then inherit the line positions.
    # fermi_mid_molecules may be a single bool (applied to all molecules) or a
    # list parallel to molecule_dirs (short lists are padded with False).
    "fermi_mid_molecules": True,  # e.g. [True, True] for a two-molecule run
    "fermi_mid_metal": False,      # e.g. True for a lone-atom "metal"; False for a bulk surface
    "fermi_mid_full": False,       # e.g. True for a discrete complex; False for a metallic adsorbed system

    # --- NCL override (optional) ---
    # "lsorbit": True,             # force NCL if INCAR is missing; auto-detected otherwise
}

if __name__ == '__main__':
    match_files = run_match(
        # Each molecule gets its own isolated VASP directory

            molecule_dirs=[
                r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551/SOC/NHC_c1m1',
                r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551/SOC/NHC_c2m1',
                r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551/SOC/NHC_c3m1',
                r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551/SOC/NHC_c4m1'
            ],
            # Metal subsystem directory
            metal_dir=r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551/SOC/adatom_surface',
            # Full combined system directory
            full_dir=r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551/SOC',
            **run_kwargs,
            )    

    # --- Plot ---
    cfg = PlotConfig(
        cmap_name_simple="managua_r",
        cmap_name_metal="vanimo_r",
        energy_range=(-15, 6),
        shared_molecule_color=True,
        min_total_mol_wspan=0.01,
        power_simple_neg=0.25,
        power_simple_pos=0.75,
        power_metal_neg=0.075,
        power_metal_pos=0.075,
        power_residual=0.5,
        pick_primary=False,
        show_local_fermi=True,
        show_vacuum_line=True,
        vacuum_line_color="black",
        vacuum_line_width=2,
        vacuum_line_style="-",
        soc_pair_thresh_eV=0.005,
    )

    plotter = RectAEPAWColorPlotter(cfg)
    fig, axes = plotter.plot(match_files[0], bonding=True)
    plt.show()
