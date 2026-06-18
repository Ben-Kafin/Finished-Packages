# -*- coding: utf-8 -*-
"""
run_combined_MOPMatcher.py

Plot two already-completed MOPMatcher runs on a single vacuum-aligned global energy
axis (analogous to run_MOPMatcher.py, but for combining a "complex" run and a "full"
run rather than producing one run).

Prerequisite: each run has already been produced by run_MOPMatcher.py / run_match, so
that each run directory contains its band_matches_rectangular.txt (with the
'# VACUUMS: aligned=' header), band_matches_rectangular_all.txt and INCAR. This
script only reads those files; it performs no VASP/builder work and writes no
files. Each input may be the run directory itself or the direct path to its
band_matches_rectangular.txt.

The complex run's energies are shifted by (aligned_full - aligned_complex) so
the two vacuum levels coincide; the full run's full-system Fermi (already at 0
because the full run used zero_reference="full") becomes the global zero.

Subplot order (top -> bottom):
    molecule row(s) | Lone gold atom | Complex | Bulk surface and adatom | Full
"""
from matcher import PlotConfig
from matcher.combined_vacupot_plotter import plot_combined_runs
import matplotlib.pyplot as plt

# --- Point these at each run's directory (or its band_matches_rectangular.txt) ---
# "complex" run: full = NHC + gold-adatom complex; components = molecule(s) + lone gold atom
complex_main_path = r'C:/Users/Benjamin Kafin/Documents/VASP/lone/fcc/SOC/NHC_complex/kp551'
# "full" run: full = complex adsorbed on surface; components = molecule(s) + surface-with-adatom
full_main_path    = r'C:/Users/Benjamin Kafin/Documents/VASP/lone/fcc/SOC/kp551'

# All color / line / cursor behavior is the single-run plotter's, controlled here.
cfg = PlotConfig(
    cmap_name_simple="managua_r",
    cmap_name_metal="vanimo_r",
    energy_range=(-7, 4),
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

fig, axes = plot_combined_runs(
    complex_main_path,
    full_main_path,
    cfg=cfg,
    bonding=True,
    # Row titles (defaults shown); set title_molecule to override the molecule
    # row label(s), otherwise each molecule's own component label is used.
    title_molecule="NHC(iPr)",
    title_lone_metal="Lone gold atom",
    title_complex="Complex",
    title_bulk="Bulk surface and adatom",
    title_full="Full",
)
plt.show()