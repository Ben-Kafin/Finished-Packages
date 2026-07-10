# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:18:23 2026

@author: Benjamin Kafin
"""

# run_stm.py
from stm_from_dft import Interactive_STM_Simulator
from matplotlib.colors import LinearSegmentedColormap

# Point at the VASP directory containing DOSCAR, LOCPOT, POSCAR/CONTCAR
#v_dir = r'C:/Users/Benjamin Kafin/Documents/VASP/CsSnI3/I_vacancy/kp661/kp10_10_1'
v_dir='C:/dir/'

sim = Interactive_STM_Simulator(
    filepath=v_dir,
    erange=[-2.475, -1.275],          # initial energy slider range
    ldos_height=1.2,              # Å above highest atom for LDOS sampling
    cmap_topo=LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"]),
    unit_cell_num=2,
)

sim.run_interactive(
    grid_res=64,                   # NxN topography grid
    topo_bias=0.2,                # V for constant-current topography
    topo_height=2.5,                # initial tip height above highest atom (Å)
    ldos_bias_sign='neg',          # 'pos' or 'neg' — which slider sets the bias
    use_decay=True,                # Chen barrier model for topography convergence and LDOS together
    show_decay_toggle=True,        # show the 'Decay' checkbox on the figure (False hides it; decay stays on)
    reuse_cache=False,              # load existing global_topo_*/ldos_topo_* .npy caches; False recomputes and rewrites them

    # Optional: pre-set line scan endpoints
    line_endpoints=([11.849932, -7.506780], [5.490312, 22.283022]),

    # Optional: pre-set marker positions
    marker_positions=[[8.697210, 7.261233], [7.080217, 14.835572]],
    #marker_positions=[[12.270892, 12.538389], [12.235102, 5.988768], [6.393806, 12.763640], [6.122548, 6.219541]],

    # Optional: extra Gaussian broadening per atom type (σ in eV)
    extra_broadening={'N': 0.0, 'C': 0.0, 'H': 0.0},

    # Optional: Path mode seed — list of 1-based atom indices (exact order).
    # In Path mode each of these becomes a draggable node; the "Path Pts" slider
    # adds/removes nodes. After dragging, nodes are free XY points (no longer
    # tied to atoms). Leave as None to start Path mode with a default 3-node path.
    path=[[12.228758, 15.837908], [12.220715, 3.355913], [9.396806, 3.287211], [9.676762, 15.757682], [6.178678, 15.516096], [5.988031, 2.974321], [3.189683, 2.910722], [3.189683, 15.694083]],

    # Optional: Path end-extension toggle (also flippable live via the 'Extend'
    # checkbox). When True, the path is extended from the first/last node along
    # the first/last segment direction to the edge of the local unit cell that
    # node sits in. When False, the path starts/ends exactly on the first/last
    # nodes.
    path_extend=False,
)
