# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:18:23 2026

@author: Benjamin Kafin
"""

# run_stm.py
from stm_from_dft import Interactive_STM_Simulator
from matplotlib.colors import LinearSegmentedColormap

# Point at the VASP directory containing DOSCAR, LOCPOT, POSCAR/CONTCAR
v_dir = r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551/dpl_corr/kp551'

sim = Interactive_STM_Simulator(
    filepath=v_dir,
    erange=[-2.525, -1.3],          # initial energy slider range
    ldos_height=1.3,               # Å above highest atom for LDOS sampling
    cmap_topo=LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"]),
)

sim.run_interactive(
    grid_res=64,                   # NxN topography grid
    topo_bias=0.2,                # V for constant-current topography
    topo_height=2.5,                 # initial tip height above highest atom (Å)
    ldos_bias_sign='neg',          # 'pos' or 'neg' — which slider sets the bias
    use_decay_topo=True,           # Chen barrier model for topography
    use_decay_ldos=True,           # Chen barrier model for LDOS

    # Optional: pre-set line scan endpoints
    line_endpoints=([11.849932, -7.506780], [5.490312, 22.283022]),

    # Optional: pre-set marker positions
    marker_positions=[[8.697210, 7.261233], [7.080217, 14.835572]],

    # Optional: extra Gaussian broadening per atom type (σ in eV)
    extra_broadening={'N': 0.3, 'C': 0.3, 'H': 0.3},
)
