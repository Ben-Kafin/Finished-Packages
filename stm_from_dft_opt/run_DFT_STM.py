# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:18:23 2026

@author: Benjamin Kafin
"""

# run_stm.py
from stm_from_dft import Interactive_STM_Simulator
from matplotlib.colors import LinearSegmentedColormap

# Point at the VASP directory containing DOSCAR, LOCPOT, POSCAR/CONTCAR
v_dir = r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551/SOC'

sim = Interactive_STM_Simulator(
    filepath=v_dir,
    erange=[-2.47, -1.25],          # initial energy slider range
    ldos_height=1.25,               # Å above highest atom for LDOS sampling
    cmap_topo=LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"]),
    unit_cell_num=1,
)

sim.run_interactive(
    grid_res=64,                   # NxN topography grid
    topo_bias=0.2,                # V for constant-current topography
    topo_height=2.5,                 # initial tip height above highest atom (Å)
    ldos_bias_sign='neg',          # 'pos' or 'neg' — which slider sets the bias
    use_decay_topo=True,           # Chen barrier model for topography
    use_decay_ldos=True,           # Chen barrier model for LDOS
    show_decay_toggle=True,        # show the 'Decay' checkbox on the figure (False hides it; decay stays on)

    # Optional: pre-set line scan endpoints
    line_endpoints=([11.849932, -7.506780], [5.490312, 22.283022]),

    # Optional: pre-set marker positions
    marker_positions=[[8.697210, 7.261233], [7.080217, 14.835572]],

    # Optional: extra Gaussian broadening per atom type (σ in eV)
    extra_broadening={'N': 0.0, 'C': 0.0, 'H': 0.0},

    # Optional: Path mode seed — list of 1-based atom indices (exact order).
    # In Path mode each of these becomes a draggable node; the "Path Pts" slider
    # adds/removes nodes. After dragging, nodes are free XY points (no longer
    # tied to atoms). Leave as None to start Path mode with a default 3-node path.
    path_atoms=[[154], [223], [193], [210], [167]],

    # Optional: Path end-extension toggle (also flippable live via the 'Extend'
    # checkbox). When True, the path is extended from the first/last node along
    # the first/last segment direction to the edge of the local unit cell that
    # node sits in. When False, the path starts/ends exactly on the first/last
    # nodes.
    path_extend=False,
)
