# -*- coding: utf-8 -*-
"""
Standalone runner for the vacupot plotter (RectAEPAWColorPlotter).

This re-plots an EXISTING matcher output (band_matches_rectangular.txt) without
re-running the matcher/builder. Point MATCH_FILE at the band_matches_rectangular.txt
written into your full-system directory (or wherever you set output_path).

The plotter reads the matcher's two text outputs from the same directory:
    band_matches_rectangular.txt        (main; this is the path you pass)
    band_matches_rectangular_all.txt    (per-overlap; read automatically alongside)
so both must be present next to each other.
"""

import os
import matplotlib
# For an interactive window (hover/click), use a GUI backend. Comment this out to
# keep your environment default (Spyder usually provides one). If you only want a
# saved PNG with no window, set: matplotlib.use("Agg")
# matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from matcher import PlotConfig, RectAEPAWColorPlotter, matcher_vacupot


# ---------------------------------------------------------------------------
# Path to the EXISTING matcher main output to plot.
# This is exactly what run_match() returns as match_files[0]; by default the
# matcher writes it into full_dir.
# ---------------------------------------------------------------------------
MATCH_FILE = r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551/SOC/band_matches_rectangular.txt'

# Optional: also save the figure to disk (set to None to skip saving).
SAVE_PATH = None   # e.g. r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551/SOC/vacupot_plot.png'
SAVE_DPI = 300

# bonding=True matches how run_MOPMatcher.py currently calls plot().
BONDING = True


# ---------------------------------------------------------------------------
# Plot configuration. Values below are the ones from your run_MOPMatcher.py; every
# PlotConfig field is shown so you can adjust any of them here.
# ---------------------------------------------------------------------------
cfg = PlotConfig(
    # --- Colormaps and color warping ---
    cmap_name_simple="managua_r",
    cmap_name_metal="vanimo_r",
    power_simple_neg=0.25,
    power_simple_pos=0.75,
    power_metal_neg=0.075,
    power_metal_pos=0.075,
    power_residual=2.0,

    # --- Figure / sticks ---
    figsize=(8.0, 3.0),
    lw_stick=2.0,
    xlabel="Energy (eV)",
    ylabel="Normalized",

    # --- Fermi lines ---
    show_fermi_line=True,
    fermi_line_style=":",
    fermi_line_color="k",
    show_local_fermi=True,
    local_fermi_style="--",
    local_fermi_color="red",

    # --- Vacuum line ---
    show_vacuum_line=True,
    vacuum_line_color="black",
    vacuum_line_width=3.5,
    vacuum_line_style="-",

    # --- Interaction / selection behavior ---
    annotate_on_hover=True,
    interactive=True,
    shared_molecule_color=False,
    energy_range=None,            # e.g. (-8.0, 4.0) to clip the x-axis
    title_full="Full system",
    pick_primary=False,       # "blended" | True | (other: per-molecule split)
    min_total_mol_wspan=0.01,
    soc_pair_thresh_eV=0.005,
)


def main():
    if not os.path.isfile(MATCH_FILE):
        raise FileNotFoundError(
            f"Matcher output not found:\n  {MATCH_FILE}\n"
            "Run run_MOPMatcher.py (run_match) first, or fix MATCH_FILE to point at an "
            "existing band_matches_rectangular.txt."
        )

    all_file = os.path.join(os.path.dirname(MATCH_FILE), "band_matches_rectangular_all.txt")
    if not os.path.isfile(all_file):
        print(f"[WARN] {all_file} not found next to the main file; the plotter "
              "expects both in the same directory.")

    plotter = RectAEPAWColorPlotter(cfg)
    fig, axes = plotter.plot(MATCH_FILE, bonding=BONDING)

    if SAVE_PATH:
        fig.savefig(SAVE_PATH, dpi=SAVE_DPI, bbox_inches="tight")
        print(f"[OK] Saved figure to {SAVE_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
