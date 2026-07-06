# -*- coding: utf-8 -*-
"""
Interactive STM Simulator: Multi-Tiered GPU Optimization
Auto-detects ISPIN=1, ISPIN=2, and NCL/SOC calculations.

Developed by Benjamin Kafin
"""

import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import cupyx.scipy.ndimage as cp_ndimage
from os.path import exists, getsize, join
from os import chdir
from numpy.linalg import norm, inv
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, CheckButtons, Button, RadioButtons, TextBox
import matplotlib.gridspec as gridspec
import matplotlib.colors as mc, colorsys
from matplotlib.lines import Line2D
import mplcursors
from time import time
import hashlib

from .doscar_parser import SpinAwareDosParser, SpinMode
from .locpot_manager import LocpotManager


# --- CORE UTILITIES ---
def gpu_simpson(y, x):
    """Vectorized Simpson's Rule for GPU parity."""
    n = y.shape[1]
    if n % 2 == 0:
        return cp.trapz(y, x=x, axis=1)
    dx = (x[-1] - x[0]) / (n - 1)
    weights = cp.ones(n)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2
    return (dx / 3.0) * cp.sum(weights * y, axis=1)


def gpu_chen_tunneling_factor(V, E, phi):
    """Vectorized Julian Chen barrier model character-for-character."""
    me, h, q = 9.1093837e-31, 6.62607015e-34, 1.60217663e-19
    V_eff = cp.where(cp.abs(V) < 1e-6, 1e-6, V)
    V_j, E_j, phi_j = cp.abs(V_eff) * q, E * q, phi * q
    prefactor = (8.0 / (3.0 * V_j)) * cp.pi * cp.sqrt(2.0 * me) / h
    term1 = cp.power(cp.maximum(0.0, phi_j - E_j + V_j), 1.5)
    term2 = cp.power(cp.maximum(0.0, phi_j - E_j), 1.5)
    return prefactor * (term1 - term2)


class Unified_STM_Simulator:
    def __init__(self, filepath, unit_cell_num=1):
        self.filepath = filepath
        # 3×3 = 9 periodic images per atom. Far images have sf ≈ exp(-2κd) ≈ 0 well before
        # the second neighbor ring, so the topo and LDOS sums are converged with one ring.
        # Override to higher values only for unusually small unit cells or very low κ.
        self.unit_cell_num = unit_cell_num
        chdir(filepath)
        self.dev = cp.cuda.Device(0)
        print("--- INITIALIZING GPU TENSOR SIMULATOR ---")

    def _converge_tip_height(self, z_map_gpu, grid_xy_gpu, emin, emax, target_ldos,
                             target_threshold=0.01, topo_gain=0.5, max_iter=1000, use_decay=True):
        """Exhaustive Point-Wise Convergence Engine."""
        t_start = time()
        print("--- INITIALIZING TIP CONVERGENCE ENGINE ---")
        for i in range(max_iter):
            t0 = time()
            ld_1, _, a_grid = self._calculate_ldos_at_points_gpu(
                cp.hstack([grid_xy_gpu, z_map_gpu[:, None]]), emin, emax,
                use_energy_decay=use_decay, preserve_orbitals=False, topo_only=True)
            # Topography always uses total LDOS — ld_1 is already combined for POL when topo_only=True.
            cur_ldos = gpu_simpson(ld_1, a_grid)
            rat = cp.maximum(cur_ldos, 1e-20) / target_ldos

            active_mask = cp.abs(rat - 1.0) > target_threshold
            active_count = int(cp.sum(active_mask))
            max_error = float(cp.max(cp.abs(rat - 1.0)))
            z_min, z_max = float(cp.min(z_map_gpu)), float(cp.max(z_map_gpu))
            print(f"   Iter {i+1:02d}: Active Pts ={active_count:4d} | Max Error ={max_error*100:6.2f}% | Z_min ={z_min:6.3f} Å | Z_max ={z_max:6.3f} Å | Range ={z_max-z_min:8.4f} Å | Time ={time()-t0:6.3f}s")
            if active_count == 0:
                break

            pts_active = cp.hstack([grid_xy_gpu, z_map_gpu[:, None]])[active_mask]
            idx_g = (cp.dot(pts_active, self.inv_lv_gpu) % 1.0).T * self.locpot_dims_gpu[:, None]
            pot_g = self._get_barrier_potential()
            phi_l = cp_ndimage.map_coordinates(pot_g, idx_g, order=1, mode='wrap') - self.ef
            kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_l))
            z_map_gpu[active_mask] += (topo_gain / (2.0 * kappa)) * cp.log(rat[active_mask])
        print(f"--- CONVERGENCE COMPLETE: Total Time ={time() - t_start:6.3f}s ---")
        return z_map_gpu

    def _get_barrier_potential(self):
        """Return the single 3D potential grid used for the tunneling barrier."""
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            # V_total = V_up + V_dn (stored as [V_up, V_dn])
            return self.locpot_gpu[0] + self.locpot_gpu[1]
        # ISPIN=1 and NCL: locpot_gpu is already the total 3D potential
        return self.locpot_gpu

    def _parse_poscar(self, ifile):
        with open(ifile, 'r') as f:
            lines = f.readlines()
            sf = float(lines[1])
            lv = np.array([float(c) for c in ' '.join(lines[2:5]).split()]).reshape(3, 3) * sf
            atomtypes = lines[5].split()
            atomnums = [int(i) for i in lines[6].split()]
            start_line = 7 if lines[7].strip().lower()[0] in ['d', 'c'] else 8
            coord = np.array([[float(c) for c in line.split()[:3]]
                              for line in lines[start_line + 1:sum(atomnums) + start_line + 1]])
            if 'direct' in lines[start_line].lower():
                coord = np.dot(coord, lv)
        return lv, coord, atomtypes, atomnums

    def _unwrap_z_vacuum(self, coord):
        """Place every atom on the slab side of the vacuum gap in z.

        z is the surface-normal direction and is NOT periodic for a slab: an atom
        whose fractional z has wrapped through the cell boundary appears at the top
        of the cell (in the vacuum above the slab) when it physically belongs below
        the slab bottom. Such phantom atoms corrupt the vacuum LDOS as the tip is
        raised. This detects the vacuum as the largest gap in the circular
        fractional-z distribution (definitionally the vacuum in a slab cell) and
        shifts any atom above the gap's midpoint down by one c-vector so the slab is
        contiguous with no atom in the top vacuum. System-independent: no hardcoded
        thresholds, and a no-op when no atom has wrapped.
        """
        inv_lv = inv(self.lv)
        frac = np.dot(coord, inv_lv)
        fz = frac[:, 2] % 1.0
        n = len(fz)
        order = np.argsort(fz)
        fz_sorted = fz[order]
        gaps = np.empty(n)
        gaps[:-1] = np.diff(fz_sorted)
        gaps[-1] = (fz_sorted[0] + 1.0) - fz_sorted[-1]   # circular wrap-around gap
        vac_i = int(np.argmax(gaps))
        low_edge = fz_sorted[vac_i]
        high_edge = fz_sorted[(vac_i + 1) % n] + (1.0 if vac_i == n - 1 else 0.0)
        cut = ((low_edge + high_edge) / 2.0) % 1.0
        fz_unwrapped = fz.copy()
        fz_unwrapped[fz_unwrapped > cut] -= 1.0
        frac[:, 2] = fz_unwrapped
        return np.dot(frac, self.lv)

    def parse_vasp_outputs(self, locpot_path):
        poscar_path = './CONTCAR' if exists('./CONTCAR') and getsize('./CONTCAR') > 0 else './POSCAR'
        self.lv, self.coord, self.atomtypes, self.atomnums = self._parse_poscar(poscar_path)
        self.coord = self._unwrap_z_vacuum(self.coord)

        # --- DOSCAR: auto-detects spin mode ---
        dos_parser = SpinAwareDosParser(join(self.filepath, 'DOSCAR'))
        self.energies, self.ef = dos_parser.energies, dos_parser.ef
        self.energies_gpu = cp.asarray(self.energies, dtype=cp.float32)
        self.spin_mode = dos_parser.spin_mode
        self.is_polarized = dos_parser.is_polarized  # backward compat (True only for COLLINEAR_POL)

        print(f"[*] Detected spin mode: {self.spin_mode.name}")

        # --- Orbital names ---
        num_cols = dos_parser.spin_up_dos.shape[2]
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            # num_cols is already per-spin (halved by parser)
            mapping_pol = {
                3: ['s_up', 's_down', 'p_up', 'p_down', 'd_up', 'd_down'],
                9: ['s_up', 's_down', 'py_up', 'py_down', 'pz_up', 'pz_down',
                    'px_up', 'px_down', 'dxy_up', 'dxy_down', 'dyz_up', 'dyz_down',
                    'dz2_up', 'dz2_down', 'dxz_up', 'dxz_down', 'dx2-y2_up', 'dx2-y2_down'],
                16: ['s_up', 's_down', 'py_up', 'py_down', 'pz_up', 'pz_down',
                     'px_up', 'px_down', 'dxy_up', 'dxy_down', 'dyz_up', 'dyz_down',
                     'dz2_up', 'dz2_down', 'dxz_up', 'dxz_down', 'dx2-y2_up', 'dx2-y2_down',
                     'fy3x2_up', 'fy3x2_down', 'fxyz_up', 'fxyz_down',
                     'fyz2_up', 'fyz2_down', 'fz3_up', 'fz3_down',
                     'fxz2_up', 'fxz2_down', 'fzx2_up', 'fzx2_down',
                     'fx3_up', 'fx3_down'],
            }
            self.orbitals = mapping_pol.get(num_cols, [])
        else:
            # ISPIN=1 and NCL: no spin suffix
            mapping_unpol = {
                3: ['s', 'p', 'd'],
                9: ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2'],
                16: ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2',
                     'fy3x2', 'fxyz', 'fyz2', 'fz3', 'fxz2', 'fzx2', 'fx3'],
            }
            self.orbitals = mapping_unpol.get(num_cols, [])

        # --- Atom labels ---
        current_global = 1
        self.vesta_label_map = {}
        for idx, t in enumerate(self.atomtypes):
            for n_rel in range(1, self.atomnums[idx] + 1):
                self.vesta_label_map[current_global] = f"{t}{n_rel}"
                current_global += 1

        # --- LOCPOT: auto-detects from section count ---
        lpt_mgr = LocpotManager(self.filepath, spin_mode=self.spin_mode)
        locpot_raw = cp.array(lpt_mgr.get_data(), dtype=cp.float32)

        if self.spin_mode == SpinMode.COLLINEAR_POL:
            # LocpotManager stores [V_up, V_dn].
            # Tunneling barrier uses TOTAL electrostatic potential = V_up + V_dn.
            # Spin physics enters through the DOS, not the barrier.
            # We keep both channels for the per-spin barrier in _compute_channel.
            v_total = locpot_raw[0] + locpot_raw[1]
            self.locpot_gpu = cp.stack([v_total, v_total])
        else:
            # ISPIN=1 and NCL: single 3D grid (total potential)
            self.locpot_gpu = locpot_raw

        self.inv_lv_gpu = cp.array(inv(self.lv), dtype=cp.float32)
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            self.locpot_dims_gpu = cp.array(self.locpot_gpu.shape[-3:], dtype=cp.float32)
        else:
            self.locpot_dims_gpu = cp.array(self.locpot_gpu.shape, dtype=cp.float32)

        self.num_total_atoms = sum(self.atomnums)

        # --- DOS arrays on GPU ---
        self.dos_up_gpu = cp.array(dos_parser.get_dos_for_simulator(spin='up'), dtype=cp.float32)
        self.dos_up_collapsed = cp.ascontiguousarray(cp.sum(self.dos_up_gpu, axis=2))

        if self.spin_mode == SpinMode.COLLINEAR_POL:
            self.dos_dn_gpu = cp.array(dos_parser.get_dos_for_simulator(spin='down'), dtype=cp.float32)
            self.dos_dn_collapsed = cp.ascontiguousarray(cp.sum(self.dos_dn_gpu, axis=2))
        elif self.spin_mode == SpinMode.NCL_SOC:
            # NCL: second channel is magnetization magnitude |m|
            self.dos_dn_gpu = cp.array(dos_parser.get_dos_for_simulator(spin='mag'), dtype=cp.float32)
            self.dos_dn_collapsed = cp.ascontiguousarray(cp.sum(self.dos_dn_gpu, axis=2))
        else:
            self.dos_dn_gpu = None
            self.dos_dn_collapsed = None

        # Store pristine copies for extra broadening re-derivation
        self._dos_up_gpu_orig = self.dos_up_gpu.copy()
        self._dos_dn_gpu_orig = self.dos_dn_gpu.copy() if self.dos_dn_gpu is not None else None

        # Precompute total-ρ DOS for COLLINEAR_POL topography (avoids computing two channels and adding).
        # NCL_SOC: dos_up_gpu IS already ρ_total per orbital. COLLINEAR_UNPOL: single channel reuses dos_up_gpu.
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            self.dos_total_gpu = self.dos_up_gpu + self.dos_dn_gpu
            self.dos_total_collapsed = cp.ascontiguousarray(cp.sum(self.dos_total_gpu, axis=2))
        else:
            self.dos_total_gpu = None
            self.dos_total_collapsed = None

        inv_lv = inv(self.lv)
        frac_coords = np.dot(self.coord, inv_lv)
        z_filter_mask = frac_coords[:, 2] < 0.9
        self.z_highest_atom = np.max(self.coord[z_filter_mask, 2])

        coords, idx_list = [], []
        base_idx = np.arange(len(self.coord))
        for i in range(-self.unit_cell_num, self.unit_cell_num + 1):
            for j in range(-self.unit_cell_num, self.unit_cell_num + 1):
                coords.append(self.coord + self.lv[0] * i + self.lv[1] * j)
                idx_list.append(base_idx)
        self.periodic_coord_gpu = cp.array(np.concatenate(coords), dtype=cp.float32)
        self.atom_indices_periodic_gpu = cp.array(np.concatenate(idx_list))
        self.map_mat_gpu = cp.zeros((self.num_total_atoms, len(self.atom_indices_periodic_gpu)), dtype=cp.float32)
        self.map_mat_gpu[self.atom_indices_periodic_gpu, cp.arange(len(self.atom_indices_periodic_gpu))] = 1.0

    def _apply_extra_broadening(self):
        """Rebuild working DOS arrays from pristine originals with extra Gaussian broadening."""
        self.dos_up_gpu = self._dos_up_gpu_orig.copy()
        if self._dos_dn_gpu_orig is not None:
            self.dos_dn_gpu = self._dos_dn_gpu_orig.copy()

        dE = self.energies[1] - self.energies[0]
        atom_types_exp = np.repeat(self.atomtypes, self.atomnums)

        for atype, sigma_eV in getattr(self, 'extra_broadening', {}).items():
            if sigma_eV <= 0:
                continue
            sigma_pts = sigma_eV / dE
            indices = [i for i, t in enumerate(atom_types_exp) if t == atype]
            if not indices:
                continue
            idx = cp.array(indices)
            self.dos_up_gpu[idx] = cp_ndimage.gaussian_filter1d(self.dos_up_gpu[idx], sigma=sigma_pts, axis=1)
            if self.dos_dn_gpu is not None:
                self.dos_dn_gpu[idx] = cp_ndimage.gaussian_filter1d(self.dos_dn_gpu[idx], sigma=sigma_pts, axis=1)

        self.dos_up_collapsed = cp.ascontiguousarray(cp.sum(self.dos_up_gpu, axis=2))
        if self.dos_dn_gpu is not None:
            self.dos_dn_collapsed = cp.ascontiguousarray(cp.sum(self.dos_dn_gpu, axis=2))

        # Refresh COLLINEAR_POL total-ρ DOS after broadening changes (NCL_SOC and UNPOL reuse dos_up_gpu).
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            self.dos_total_gpu = self.dos_up_gpu + self.dos_dn_gpu
            self.dos_total_collapsed = cp.ascontiguousarray(cp.sum(self.dos_total_gpu, axis=2))

    def _broadening_cache_suffix(self):
        """Generate cache filename suffix encoding active broadening state."""
        active = sorted((t, s) for t, s in getattr(self, 'extra_broadening', {}).items() if s > 0)
        if not active:
            return ""
        return "_broad_" + "_".join(f"{t}{s:.3f}" for t, s in active)

    def _angular_cache_suffix(self):
        # Angular-mode results carry their own cache tag so they never mix with
        # isotropic results or with each other across kernel revisions.
        return "_ang3" if getattr(self, 'use_angular', False) else ""

    def _unit_cell_cache_suffix(self):
        # Periodic-image count affects every sf sum. Embed in cache filename so old caches
        # auto-invalidate when unit_cell_num changes.
        return f"_uc{getattr(self, 'unit_cell_num', 1)}"

    def _decay_cache_suffix(self):
        # Converged topographies depend on the decay model used to produce them.
        return "" if getattr(self, 'use_decay_topo', True) else "_nodecay"

    def _get_angular_azimuthal(self, wrapped_tip_pos_gpu):
        """Azimuthal factors of the orbital angular densities, one per tip-image pair.

        These depend only on the in-plane tip layout and the periodic atom
        positions, not on z. During constant-current convergence only z changes,
        so the four arrays are computed once per tip layout and served from a
        cache on every subsequent call with the same layout.
        """
        xy_key = hashlib.blake2b(cp.asnumpy(wrapped_tip_pos_gpu[:, :2]).tobytes(),
                                 digest_size=16).hexdigest()
        cache = getattr(self, '_azim_cache', None)
        if cache is not None and cache['key'] == xy_key:
            return cache['arrays']
        delta_xy = wrapped_tip_pos_gpu[None, :, :2] - self.periodic_coord_gpu[:, None, :2]
        phi = cp.arctan2(delta_xy[..., 1], delta_xy[..., 0])
        cos_phi = cp.cos(phi)
        sin_phi = cp.sin(phi)
        sin2_phi = (sin_phi * sin_phi).astype(cp.float32)
        cos2_phi = (cos_phi * cos_phi).astype(cp.float32)
        sin2_2phi = (4.0 * sin2_phi * cos2_phi).astype(cp.float32)
        cos2_2phi = ((2.0 * cos2_phi - 1.0) ** 2).astype(cp.float32)
        arrays = (sin2_phi, cos2_phi, sin2_2phi, cos2_2phi)
        self._azim_cache = {'key': xy_key, 'arrays': arrays}
        return arrays

    def _calculate_ldos_at_points_gpu(self, tip_positions, emin, emax,
                                      use_energy_decay=False, preserve_orbitals=False,
                                      global_bias=None, topo_only=False):
        estart = np.searchsorted(self.energies, emin)
        eend = np.searchsorted(self.energies, emax, side='right')
        energy_indices = cp.arange(estart, eend)
        calc_energies_gpu = self.energies_gpu[estart:eend]
        num_pts, num_e = tip_positions.shape[0], len(calc_energies_gpu)
        tip_pos_gpu = cp.asarray(tip_positions, dtype=cp.float32)
        frac_coords = cp.dot(tip_pos_gpu, self.inv_lv_gpu)

        wrapped_frac_coords = cp.empty_like(frac_coords)
        wrapped_frac_coords[:, :2] = frac_coords[:, :2] % 1.0
        wrapped_frac_coords[:, 2] = frac_coords[:, 2]

        lv_gpu = cp.array(self.lv, dtype=cp.float32)
        wrapped_tip_pos_gpu = cp.dot(wrapped_frac_coords, lv_gpu)

        grid_indices = (frac_coords % 1.0).T * self.locpot_dims_gpu[:, None]
        dists = cp.sqrt(cp.sum((self.periodic_coord_gpu[:, None, :] - wrapped_tip_pos_gpu[None, :, :]) ** 2, axis=2))

        def _compute_channel(pot, dos_gpu, dos_collapsed):
            phi_local = cp_ndimage.map_coordinates(pot, grid_indices, order=1, mode='wrap') - self.ef
            if self.use_angular:
                # Tersoff-Hamann point-LDOS reconstruction: each atom's
                # lm-projected DOSCAR population is placed into the vacuum with
                # the angular density 4*pi*|Y_lm|^2 of the orthonormal real
                # spherical harmonic it was projected onto (LORBIT=14 basis;
                # Schueler et al., JPCM 30, 475901 (2018), Eqs. (4)-(12)),
                # weighted by the tunneling decay of the same tip-image pair.
                # VASP lm column order: s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2.
                # Every channel angle-averages to 1, so the isotropic average of
                # this mode equals the plain PDOS sum of the toggle-off path.
                # Each factor splits into a polar part (z-dependent, built from
                # the same distances the decay uses) and an azimuthal part
                # (z-independent, served from the per-layout cache).
                sin2_phi, cos2_phi, sin2_2phi, cos2_2phi = self._get_angular_azimuthal(wrapped_tip_pos_gpu)
                dz = wrapped_tip_pos_gpu[None, :, 2] - self.periodic_coord_gpu[:, None, 2]
                inv_d = 1.0 / cp.maximum(dists, 1e-30)
                cos2_theta = (dz * inv_d) ** 2
                sin2_theta = cp.maximum(0.0, 1.0 - cos2_theta)
                sin4_theta = sin2_theta * sin2_theta
                sin2_2theta = 4.0 * sin2_theta * cos2_theta
                dz2_polar = (3.0 * cos2_theta - 1.0) ** 2

                # chan[b, p, t] = 4*pi*|Y_b|^2 at the angle from periodic image p to tip t.
                chan = cp.empty((9,) + dists.shape, dtype=cp.float32)
                chan[0] = 1.0                                   # s
                chan[1] = 3.0 * sin2_theta * sin2_phi           # py
                chan[2] = 3.0 * cos2_theta                      # pz
                chan[3] = 3.0 * sin2_theta * cos2_phi           # px
                chan[4] = 3.75 * sin4_theta * sin2_2phi         # dxy    (15/4)
                chan[5] = 3.75 * sin2_2theta * sin2_phi         # dyz    (15/4)
                chan[6] = 1.25 * dz2_polar                      # dz2    (5/4)(3cos2t-1)^2
                chan[7] = 3.75 * sin2_2theta * cos2_phi         # dxz    (15/4)
                chan[8] = 3.75 * sin4_theta * cos2_2phi         # dx2-y2 (15/4)

                if not preserve_orbitals:
                    # Orbital channel folded into the same contraction axis as the
                    # periodic images: q = (b, p), so the kernel is the identical
                    # GEMM/GEMV the isotropic path runs, with a 9x longer axis.
                    dos_q = dos_gpu[self.atom_indices_periodic_gpu][:, energy_indices, :]
                    dos_q = cp.ascontiguousarray(dos_q.transpose(2, 0, 1).reshape(-1, num_e))
                    if use_energy_decay:
                        bias_v = cp.asarray(global_bias if global_bias is not None else (emax - emin), dtype=cp.float32)
                        K_all = gpu_chen_tunneling_factor(bias_v, calc_energies_gpu[:, None], phi_local)
                        output_ldos = cp.zeros((num_pts, num_e), dtype=cp.float32)
                        for e_idx in range(num_e):
                            sf = cp.exp(-1.0 * dists * K_all[e_idx][None, :] * 1e-10)
                            W = (chan * sf[None, :, :]).reshape(-1, num_pts)
                            output_ldos[:, e_idx] = cp.dot(W.T, dos_q[:, e_idx])
                    else:
                        kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_local))
                        sf = cp.exp(-2.0 * kappa[None, :] * dists)
                        W = (chan * sf[None, :, :]).reshape(-1, num_pts)
                        output_ldos = cp.dot(W.T, dos_q)
                else:
                    # Atom- and orbital-resolved output: collapse images onto base
                    # atoms with one batched matmul over the 9 channels, then
                    # broadcast against the per-atom orbital DOS.
                    dos_active = dos_gpu[:, energy_indices, :]
                    if use_energy_decay:
                        bias_v = cp.asarray(global_bias if global_bias is not None else (emax - emin), dtype=cp.float32)
                        K_all = gpu_chen_tunneling_factor(bias_v, calc_energies_gpu[:, None], phi_local)
                        output_ldos = cp.zeros((num_pts, num_e, self.num_total_atoms, dos_gpu.shape[2]), dtype=cp.float32)
                        for e_idx in range(num_e):
                            sf = cp.exp(-1.0 * dists * K_all[e_idx][None, :] * 1e-10)
                            w_orb = cp.matmul(self.map_mat_gpu[None, :, :], chan * sf[None, :, :])
                            output_ldos[:, e_idx, :, :] = (w_orb.transpose(2, 1, 0)
                                                           * dos_active[:, e_idx, :][None, :, :])
                    else:
                        kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_local))
                        sf = cp.exp(-2.0 * kappa[None, :] * dists)
                        w_orb = cp.matmul(self.map_mat_gpu[None, :, :], chan * sf[None, :, :])
                        output_ldos = (w_orb.transpose(2, 1, 0)[:, None, :, :]
                                       * dos_active.transpose(1, 0, 2)[None, :, :, :])
            else:
                if not preserve_orbitals:
                    output_ldos = cp.zeros((num_pts, num_e), dtype=cp.float32)
                    dos_periodic = dos_collapsed[self.atom_indices_periodic_gpu, :]
                    dos_active = dos_periodic[:, energy_indices]
                    if use_energy_decay:
                        bias_v = cp.asarray(global_bias if global_bias is not None else (emax - emin), dtype=cp.float32)
                        K_all = gpu_chen_tunneling_factor(bias_v, calc_energies_gpu[:, None], phi_local)
                        for e_idx in range(num_e):
                            sf = cp.exp(-1.0 * dists * K_all[e_idx][None, :] * 1e-10)
                            output_ldos[:, e_idx] = cp.dot(sf.T, dos_active[:, e_idx])
                    else:
                        kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_local))
                        sf = cp.exp(-2.0 * kappa[None, :] * dists)
                        output_ldos = cp.dot(sf.T, dos_active)
                else:
                    output_ldos = cp.zeros((num_pts, num_e, self.num_total_atoms, dos_gpu.shape[2]), dtype=cp.float32)
                    if use_energy_decay:
                        bias_v = cp.asarray(global_bias if global_bias is not None else (emax - emin), dtype=cp.float32)
                        K_all = gpu_chen_tunneling_factor(bias_v, calc_energies_gpu[:, None], phi_local)
                        for e_idx in range(num_e):
                            sf = cp.exp(-1.0 * dists * K_all[e_idx][None, :] * 1e-10)
                            w_atom = cp.dot(self.map_mat_gpu, sf)
                            output_ldos[:, e_idx, :, :] = w_atom.T[:, :, None] * dos_gpu[:, energy_indices[e_idx], :][None, :, :]
                    else:
                        kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_local))
                        sf = cp.exp(-2.0 * kappa[None, :] * dists)
                        w_atom = cp.dot(self.map_mat_gpu, sf)
                        output_ldos = w_atom.T[:, None, :, None] * dos_gpu[:, energy_indices, :].transpose(1, 0, 2)[None, :, :, :]
            return output_ldos

        # All modes use total potential for the barrier
        barrier_pot = self._get_barrier_potential()

        if topo_only:
            # Topography only needs ρ_total. Skip the second (magnetization) channel that
            # _converge_tip_height / topo callers ignore — 2× speedup for NCL_SOC and POL.
            if self.spin_mode == SpinMode.COLLINEAR_POL:
                # Use precomputed (dos_up + dos_dn) instead of computing two channels and adding.
                return (_compute_channel(barrier_pot, self.dos_total_gpu, self.dos_total_collapsed),
                        None, calc_energies_gpu)
            # NCL_SOC: dos_up_gpu IS already ρ_total per orbital.
            # COLLINEAR_UNPOL: dos_up_gpu IS the single channel.
            return (_compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed),
                    None, calc_energies_gpu)

        if self.spin_mode == SpinMode.COLLINEAR_POL:
            # Collinear polarized: two channels (up, down), both against total barrier
            return (_compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed),
                    _compute_channel(barrier_pot, self.dos_dn_gpu, self.dos_dn_collapsed),
                    calc_energies_gpu)
        elif self.spin_mode == SpinMode.NCL_SOC:
            # NCL: channel 1 = total DOS, channel 2 = |m| DOS
            return (_compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed),
                    _compute_channel(barrier_pot, self.dos_dn_gpu, self.dos_dn_collapsed),
                    calc_energies_gpu)
        else:
            # Unpolarized: single channel
            return (_compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed),
                    None, calc_energies_gpu)

    # ------------------------------------------------------------------
    # _combine_ldos: central dispatch for how channels are combined at display time
    # ------------------------------------------------------------------
    def _combine_ldos_total(self, ld_1, ld_2):
        """Combine channels into total LDOS (for topography / non-mag display)."""
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            return ld_1 + ld_2
        # NCL and unpol: ld_1 is already the total
        return ld_1

    def _combine_ldos_display(self, ld_1, ld_2, show_mag):
        """Combine channels for display, respecting show_mag toggle."""
        if not show_mag or ld_2 is None:
            return self._combine_ldos_total(ld_1, ld_2)
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            return ld_1 - ld_2   # spin density (signed)
        if self.spin_mode == SpinMode.NCL_SOC:
            return ld_2           # |m| (unsigned, always ≥ 0)
        return ld_1

    def _mag_is_signed(self):
        """Whether the magnetization display data is signed (needs bwr) or not."""
        return self.spin_mode == SpinMode.COLLINEAR_POL

    def _has_mag_channel(self):
        """Whether this spin mode supports the Mag checkbox."""
        return self.spin_mode in (SpinMode.COLLINEAR_POL, SpinMode.NCL_SOC)


class Interactive_STM_Simulator(Unified_STM_Simulator):
    def __init__(self, filepath, erange, ldos_height, cmap_topo, unit_cell_num=1):
        super().__init__(filepath, unit_cell_num=unit_cell_num)
        self.parse_vasp_outputs("LOCPOT")
        self.p1, self.p2 = None, None
        self.erange, self.ldos_height, self.cmap_topo = list(erange), ldos_height, cmap_topo
        self.npts = 72
        self.is_running, self.normalize, self.show_mag = False, False, False
        self.show_atoms, self.show_unit_cell = True, False
        self.use_decay_topo, self.use_decay_ldos = True, True
        self.show_decay_toggle = True
        self.display_cells = 1
        self.mode = 'Single Point'
        # 'Dcmp Norm' checkbox removed from the UI; the per-panel normalization
        # logic it controlled is retained throughout the display code and
        # hard-set off here. Restore a control for this flag to re-enable it.
        self.show_dcmp_norm = False
        self.use_angular = False
        # PowerNorm gamma for all LDOS heatmaps (1.0 = linear; <1 squashes the
        # colormap toward the minimum). Live-controlled by the bottom-row slider.
        self.ldos_gamma = 1.0

        self.m_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#00ced1', '#e377c2',
                         '#17becf', '#bcbd22', '#7f7f7f', '#8c564b', '#d62728']
        self.marker_ratios = [0.25, 0.75]
        self.marker_coords = [[self.lv[0, 0] * 0.2, self.lv[1, 1] * 0.2],
                              [self.lv[0, 0] * 0.8, self.lv[1, 1] * 0.8]]
        self.active_obj = None
        self.active_marker_idx = 0
        self.plot_level = 0
        self.active_element, self.active_atom = None, None
        self._type_color_map = {'Au': 'orange', 'N': 'blue', 'C': 'brown', 'H': 'grey'}
        self.extra_broadening = {t: 0.0 for t in self.atomtypes}
        self._broadening_topo_dirty = False

        self.cached_p1, self.cached_p2 = None, None
        self.cached_emin, self.cached_emax = None, None
        self.cached_d_topo_line, self.cached_d_topo_map, self.cached_d_ldos = None, None, None
        self.cached_bias_energy_line, self.cached_bias_energy_map, self.cached_nepts = None, None, None
        self.cached_ld_up, self.cached_ld_dn, self.cached_eg = None, None, None
        self.cached_marker_coords, self.cached_spec_ldos = None, None

        # --- Path mode state ---
        self.path = None                  # initial seed: all atom-index lists [[154],...] OR all [x,y] coords; no mix
        self.path_extend = False          # toggle: extend ends to first local cell wall
        self.path_nodes = None            # live list of N draggable [x, y] node positions
        self.cached_path_nodes = None     # cache-invalidation key for path topo
        self.cached_path_extend = None
        # Registry of (slider, textbox, fmt) for two-way slider<->textbox sync
        self._slider_textboxes = []

    def run_interactive(self, grid_res=64, topo_bias=0.2, topo_height=2.5,
                        ldos_bias_sign='neg', use_decay_topo=True, use_decay_ldos=True,
                        show_decay_toggle=True,
                        line_endpoints=None, marker_positions=None, extra_broadening=None,
                        use_angular=False, path=None, path_extend=False):
        self.ldos_bias_sign = ldos_bias_sign
        self.use_decay_topo, self.use_decay_ldos = use_decay_topo, use_decay_ldos
        self.show_decay_toggle = bool(show_decay_toggle)
        self.use_angular = use_angular
        if line_endpoints is not None:
            self.p1, self.p2 = np.array(line_endpoints[0], dtype=float), np.array(line_endpoints[1], dtype=float)
        elif self.p1 is None:
            self.p1, self.p2 = np.array([0.0, 0.0]), self.lv[0, :2] + self.lv[1, :2]

        if marker_positions is not None:
            self.marker_coords = [list(pt) for pt in marker_positions]
            v = self.p2 - self.p1
            v_sq = np.dot(v, v)
            if v_sq > 1e-9:
                self.marker_ratios = [np.clip(np.dot(np.array(pt) - self.p1, v) / v_sq, 0, 1) for pt in marker_positions]
            else:
                self.marker_ratios = list(np.linspace(0.1, 0.9, len(marker_positions)))
        self.global_topo_bias = topo_bias
        self.topo_height = topo_height

        # --- Path mode seeding ---
        self.path_extend = path_extend
        if path is not None:
            self.path = path
            lengths = {len(p) for p in path}
            if lengths == {1}:
                # Atom numbers (1-based): look up each atom's xy from self.coord
                self.path_nodes = [self.coord[p[0] - 1, :2].astype(float).copy() for p in path]
            elif lengths == {2}:
                # Cartesian [x, y] coordinates: use directly
                self.path_nodes = [np.asarray(p, dtype=float).copy() for p in path]
            else:
                raise ValueError(
                    "path must be either all atom-number elements like [[154],[223],...] "
                    "or all coordinate elements like [[x,y],[x,y],...] — not mixed."
                )
        elif self.path_nodes is None:
            # Default 3-node path so Path mode is usable even if no seed supplied
            self.path_nodes = [np.array([0.0, 0.0]),
                               0.5 * (self.lv[0, :2] + self.lv[1, :2]),
                               self.lv[0, :2] + self.lv[1, :2]]

        # --- Tip-height slider range: 0 .. (cell-top Z minus highest atom Z) ---
        self.tip_height_max = float(self.lv[2, 2] - self.z_highest_atom)

        # Initialize extra broadening
        self.extra_broadening = {t: 0.0 for t in self.atomtypes}
        if extra_broadening is not None:
            for t, sigma in extra_broadening.items():
                if t in self.extra_broadening:
                    self.extra_broadening[t] = sigma
        self._apply_extra_broadening()

        print("\n--- Phase 1: Global Topography Pre-Calculation ---")
        grid_xy = (np.meshgrid(np.linspace(0, 1, grid_res), np.linspace(0, 1, grid_res))[0].ravel()[:, None] * self.lv[0, :2]) + \
                  (np.meshgrid(np.linspace(0, 1, grid_res), np.linspace(0, 1, grid_res))[1].ravel()[:, None] * self.lv[1, :2])
        grid_xy_gpu = cp.array(grid_xy, dtype=cp.float32)
        z_fixed = cp.full(grid_xy_gpu.shape[0], self.z_highest_atom + topo_height, dtype=cp.float32)
        t_emin, t_emax = sorted([0.0, topo_bias])
        cache_name = f"global_topo_{topo_bias}V_{topo_height}A_{grid_res}px{self._broadening_cache_suffix()}{self._angular_cache_suffix()}{self._unit_cell_cache_suffix()}{self._decay_cache_suffix()}.npy"
        if exists(cache_name):
            print(f"[*] Loading Cached Global Topography: {cache_name}")
            self.current_z_map = np.load(cache_name)
            z_map_gpu = cp.array(self.current_z_map, dtype=cp.float32)
        else:
            print(f"[*] Cache not found. Calculating Global Topography: {cache_name}")
            ld_1, _, init_engs = self._calculate_ldos_at_points_gpu(
                cp.hstack([grid_xy_gpu, z_fixed[:, None]]), t_emin, t_emax,
                use_energy_decay=self.use_decay_topo, preserve_orbitals=False, topo_only=True)
            target_setp = cp.max(gpu_simpson(ld_1, init_engs))
            print(f"[*] Global Setpoint LDOS: {float(target_setp):.6e}")
            z_map_gpu = self._converge_tip_height(z_fixed, grid_xy_gpu, t_emin, t_emax, target_setp, use_decay=self.use_decay_topo)
            self.current_z_map = cp.asnumpy(z_map_gpu)
            np.save(cache_name, self.current_z_map)
        self.grid_xy = grid_xy
        self.global_z_map = self.current_z_map.copy()
        self.grid_xy_gpu = grid_xy_gpu
        self.fig = plt.figure(figsize=(20, 14))
        self._build_ui()
        plt.show()

    def _get_partitions(self, f_ldos_raw):
        partitions = []
        if f_ldos_raw is not None:
            if self.plot_level == 0:
                partitions.append(("Total", np.sum(f_ldos_raw, axis=(2, 3))))
            elif self.plot_level == 1:
                atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
                au_idx = [i for i, t in enumerate(atom_types_exp) if t == 'Au']
                mol_idx = [i for i, t in enumerate(atom_types_exp) if t != 'Au']
                au_data = np.sum(f_ldos_raw[:, :, au_idx, :], axis=(2, 3)) if au_idx else np.zeros_like(f_ldos_raw[:, :, 0, 0])
                mol_data = np.sum(f_ldos_raw[:, :, mol_idx, :], axis=(2, 3)) if mol_idx else np.zeros_like(f_ldos_raw[:, :, 0, 0])
                partitions.extend([("Au", au_data), ("Molecule", mol_data)])
            elif self.plot_level == 2:
                atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
                for t in self.atomtypes:
                    t_idx = [i for i, x in enumerate(atom_types_exp) if x == t]
                    t_data = np.sum(f_ldos_raw[:, :, t_idx, :], axis=(2, 3)) if t_idx else np.zeros_like(f_ldos_raw[:, :, 0, 0])
                    partitions.append((t, t_data))
            elif self.plot_level == 3:
                atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
                if getattr(self, 'active_element', None) is not None:
                    e_idx = [i for i, x in enumerate(atom_types_exp) if x == self.active_element]
                    for col_idx, orb in enumerate(self.orbitals):
                        if e_idx:
                            D = cp.sum(self.dos_up_gpu[e_idx, :, col_idx], axis=0)
                            dD = cp.gradient(D)
                            d2D = cp.gradient(dD)
                            if not (float(cp.max(cp.abs(D))) < 1e-5 and float(cp.max(cp.abs(dD))) < 1e-5 and float(cp.max(cp.abs(d2D))) < 1e-5):
                                orb_data = np.sum(f_ldos_raw[:, :, e_idx, col_idx], axis=2)
                                partitions.append((f"{self.active_element} {orb}", orb_data))
            elif self.plot_level == 4:
                atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
                if getattr(self, 'active_element', None) is not None:
                    e_idx = [i for i, x in enumerate(atom_types_exp) if x == self.active_element]
                    for col_idx, orb in enumerate(self.orbitals):
                        if e_idx:
                            D = cp.sum(self.dos_up_gpu[e_idx, :, col_idx], axis=0)
                            dD = cp.gradient(D)
                            d2D = cp.gradient(dD)
                            if not (float(cp.max(cp.abs(D))) < 1e-5 and float(cp.max(cp.abs(dD))) < 1e-5 and float(cp.max(cp.abs(d2D))) < 1e-5):
                                orb_data = np.sum(f_ldos_raw[:, :, e_idx, col_idx], axis=2)
                                partitions.append((f"{self.active_element} {orb}", orb_data))
        return partitions

    def _build_path_xy(self):
        """Build resampled path sample points from the current draggable nodes.

        Reuses only the path logic: indices->XY (done at seed time), optional
        directional end-extension to the first local unit-cell wall, and
        arc-length resampling to self.npts uniform points.

        Returns (p_xy, p_dist, p_len, verts).
        """
        verts = [np.asarray(n, dtype=float) for n in self.path_nodes]

        if self.path_extend and len(verts) >= 2:
            inv2 = inv(self.lv)[:2, :2]   # 2x2 XY fractional transform

            def _wall_crossing(p0, d):
                # smallest t>0 where ray p0 + t*d crosses a local cell wall
                f0 = p0 @ inv2
                fd = d @ inv2
                ts = []
                for k in range(2):
                    if abs(fd[k]) > 1e-12:
                        target = np.floor(f0[k]) + (1.0 if fd[k] > 0 else 0.0)
                        if np.isclose(f0[k], target) and fd[k] < 0:
                            target -= 1.0
                        t = (target - f0[k]) / fd[k]
                        if t > 1e-9:
                            ts.append(t)
                if not ts:
                    return None
                return p0 + min(ts) * d

            start_ext = _wall_crossing(verts[0], verts[0] - verts[1])
            end_ext = _wall_crossing(verts[-1], verts[-1] - verts[-2])
            if start_ext is not None:
                verts = [start_ext] + verts
            if end_ext is not None:
                verts = verts + [end_ext]

        verts = np.array(verts)
        seg = np.linalg.norm(np.diff(verts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        p_len = float(cum[-1]) if cum[-1] > 1e-12 else 1e-12
        samp = np.linspace(0, p_len, self.npts)
        px = np.interp(samp, cum, verts[:, 0])
        py = np.interp(samp, cum, verts[:, 1])
        p_xy = np.column_stack([px, py])
        p_dist = np.linspace(0, p_len, self.npts)
        return p_xy, p_dist, p_len, verts

    def _set_textbox_text(self, box, s):
        """Set a TextBox's displayed text WITHOUT calling set_val().

        matplotlib's TextBox.set_val() triggers cursor re-rendering
        (_rendercursor -> get_window_extent -> fig.dpi), which raises
        'NoneType has no attribute dpi' when called re-entrantly from within an
        on_submit handler, and can leave the mouse grab leaked. Updating the
        underlying Text artist directly avoids that path entirely.
        """
        try:
            box.text = s
            box.text_disp.set_text(s)
            if self.fig is not None and self.fig.canvas is not None:
                self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _add_slider_textbox(self, slider, rect, fmt="{:.3f}"):
        """Attach an editable text box mirroring a slider's value.

        Two-way: typing a complete number (Enter) clamps to the slider range and
        calls slider.set_val (which fires the slider's existing on_changed).
        Incomplete/partial input (empty, '-', '+', '.', '-.') is treated as
        in-progress and the display silently reverts. The slider's built-in value
        text is hidden to avoid a duplicate number.
        """
        try:
            slider.valtext.set_visible(False)
        except Exception:
            pass
        ax_tb = plt.axes(rect)
        tb = TextBox(ax_tb, "", initial=fmt.format(slider.val))

        def _submit(text, s=slider, box=tb, f=fmt):
            # Guard the whole body so a raised exception can never leave the
            # TextBox holding the canvas mouse grab (which would lock out every
            # other widget with 'Another Axes already grabs mouse input').
            try:
                stripped = (text or "").strip()
                if stripped in ("", "-", "+", ".", "-.", "+.", "e", "E"):
                    # partial / in-progress entry: revert display, do nothing
                    self._set_textbox_text(box, f.format(s.val))
                    return
                try:
                    v = float(stripped)
                except (ValueError, TypeError):
                    self._set_textbox_text(box, f.format(s.val))
                    return
                v = min(max(v, s.valmin), s.valmax)
                s.set_val(v)
                self._set_textbox_text(box, f.format(s.val))
            except Exception:
                pass
            finally:
                # Release a leaked mouse grab if this textbox's axes holds it.
                try:
                    cv = box.ax.figure.canvas
                    if getattr(cv, 'mouse_grabber', None) is box.ax:
                        cv.release_mouse(box.ax)
                except Exception:
                    pass

        tb.on_submit(_submit)
        self._slider_textboxes.append((slider, tb, fmt))
        return tb

    def _sync_textboxes(self):
        """Refresh every slider's text box to reflect the current slider value."""
        for s, box, fmt in self._slider_textboxes:
            try:
                cur = fmt.format(s.val)
                if box.text != cur:
                    self._set_textbox_text(box, cur)
            except Exception:
                pass

    def _layout_bottom_row(self, entries):
        """Evenly distribute the active-mode bottom-row sliders across the full
        figure width. `entries` is a list of (slider, has_textbox, fmt). Each
        slot reserves room on its LEFT for the slider's label (matplotlib draws
        the Slider label to the left of its axes), then the slider, then (if
        requested) its textbox on the right, with a gap before the next slot so
        nothing overlaps. Only the widgets active for the current mode are passed
        in, so the row always spans the whole window.
        """
        n = len(entries)
        if n == 0:
            return
        x_start, x_end = 0.215, 0.99      # full usable width right of the left column
        y0 = 0.045                          # vertical band for the row
        h = 0.022
        total = x_end - x_start
        slot = total / n
        label_pad = slot * 0.22             # left room for the slider's text label
        inner_gap = slot * 0.12             # gap between adjacent slots
        usable = slot - label_pad - inner_gap
        for i, (slider, has_tb, fmt) in enumerate(entries):
            sx = x_start + i * slot + label_pad
            if has_tb:
                sld_w = usable * 0.62
                tb_w = usable * 0.30
                tb_gap = usable * 0.08
                slider.ax.set_position([sx, y0, sld_w, h])
                self._add_slider_textbox(slider, [sx + sld_w + tb_gap, y0, tb_w, h], fmt=fmt)
            else:
                slider.ax.set_position([sx, y0, usable, h])

    def _build_slider_column(self, cell, label, valinit, on_rerun):
        """Build a vertical height-slider control as ONE gridspec column.

        `cell` is a SubplotSpec (a single column of the top-row gridspec). The
        column is subdivided vertically into slider (top) / textbox (middle) /
        Rerun button (bottom), so the whole control is one entity in the grid and
        reflows with the plots at any window size. Returns (slider, textbox, btn).
        """
        sub = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=cell, height_ratios=[0.62, 0.10, 0.13], hspace=0.45)
        ax_sld = self.fig.add_subplot(sub[0])
        ax_tb = self.fig.add_subplot(sub[1])
        ax_btn = self.fig.add_subplot(sub[2])
        sld = Slider(ax_sld, label, 0.0, self.tip_height_max,
                     valinit=valinit, orientation='vertical')
        # Build the editable textbox on the dedicated axes (reuse the safe
        # textbox logic but with a pre-made axes).
        try:
            sld.valtext.set_visible(False)
        except Exception:
            pass
        ax_tb.set_navigate(False)
        tb = TextBox(ax_tb, "", initial="{:.3f}".format(sld.val))

        def _submit(text, s=sld, box=tb, f="{:.3f}"):
            try:
                stripped = (text or "").strip()
                if stripped in ("", "-", "+", ".", "-.", "+.", "e", "E"):
                    self._set_textbox_text(box, f.format(s.val)); return
                try:
                    v = float(stripped)
                except (ValueError, TypeError):
                    self._set_textbox_text(box, f.format(s.val)); return
                v = min(max(v, s.valmin), s.valmax)
                s.set_val(v)
                self._set_textbox_text(box, f.format(s.val))
            except Exception:
                pass
            finally:
                try:
                    cv = box.ax.figure.canvas
                    if getattr(cv, 'mouse_grabber', None) is box.ax:
                        cv.release_mouse(box.ax)
                except Exception:
                    pass

        tb.on_submit(_submit)
        self._slider_textboxes.append((sld, tb, "{:.3f}"))
        btn = Button(ax_btn, 'Rerun', color='lightgray', hovercolor='lime')
        btn.on_clicked(on_rerun)
        return sld, tb, btn

    def _disconnect_stale_widgets(self):
        """Disconnect canvas event callbacks of all widgets from the PREVIOUS
        build before the figure is cleared.

        fig.clf() detaches widget artists (their .figure becomes None) but does
        NOT remove the canvas-level event handlers that matplotlib widgets
        register (TextBox._click, Slider/Button press handlers, etc.). Those
        stale handlers survive the rebuild and, when later triggered by a click,
        call get_window_extent() on a detached Text artist -> fig.dpi is None ->
        'NoneType has no attribute dpi'. Explicitly disconnecting each prior
        widget's events prevents that.
        """
        widgets = []
        for v in list(self.__dict__.values()):
            if hasattr(v, 'disconnect_events'):
                widgets.append(v)
            elif isinstance(v, dict):
                for vv in v.values():
                    if hasattr(vv, 'disconnect_events'):
                        widgets.append(vv)
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    # _slider_textboxes holds (slider, textbox, fmt) tuples
                    if isinstance(vv, (list, tuple)):
                        for w in vv:
                            if hasattr(w, 'disconnect_events'):
                                widgets.append(w)
                    elif hasattr(vv, 'disconnect_events'):
                        widgets.append(vv)
        for w in widgets:
            try:
                w.disconnect_events()
            except Exception:
                pass

    def _build_ui(self):
        # Tear down the previous build's widget event handlers BEFORE clearing the
        # figure, so no stale click handler can fire on a detached (figure=None)
        # textbox after the rebuild.
        self._disconnect_stale_widgets()
        self.fig.clf()
        # fig.clf() destroys the widget axes, but the Python attributes persist.
        # Drop mode-specific sliders so later hasattr() checks reflect ONLY the
        # widgets the current mode actually builds (otherwise e.g. a stale
        # s_nepts from a previous Map view leaks into the Path/Line bottom row).
        for _attr in ('s_path_pts', 's_nepts'):
            if hasattr(self, _attr):
                delattr(self, _attr)
        self.fig.suptitle(self.spin_mode.name, fontsize=12, y=0.995, va='top')
        # Reset textbox registry each rebuild (axes are recreated on fig.clf)
        self._slider_textboxes = []
        ax_radio = plt.axes([0.02, 0.90, 0.1, 0.08], facecolor='lightgray')
        _radio_labels = ('Single Point', 'Line', 'Path', 'Map')
        self.ui_radio = RadioButtons(ax_radio, _radio_labels, active=list(_radio_labels).index(self.mode))
        self.ui_radio.on_clicked(self._on_mode_change)

        self.btn_run = Button(plt.axes([0.02, 0.02, 0.05, 0.06]), 'RUN', color='lightgray', hovercolor='lime')
        # One list defines every checkbox: label and the attribute it drives.
        # CheckButtons construction and the click handler both iterate this list.
        toggle_defs = [('Atoms', 'show_atoms'), ('Decay', 'use_decay_ldos'),
                       ('Norm', 'normalize'), ('Mag', 'show_mag'),
                       ('Cell', 'show_unit_cell'), ('Angular', 'use_angular'),
                       ('Extend', 'path_extend')]
        if not getattr(self, 'show_decay_toggle', True):
            toggle_defs = [d for d in toggle_defs if d[0] != 'Decay']
        self._toggle_defs = toggle_defs
        self.chk = CheckButtons(plt.axes([0.08, 0.02, 0.12, 0.092]),
                                [label for label, _ in toggle_defs],
                                [getattr(self, attr) for _, attr in toggle_defs])
        self.s_cell = Slider(plt.axes([0.22, 0.05, 0.1, 0.03]), 'Cells', 0, 4, valinit=self.display_cells, valstep=1)
        self.s_emin = Slider(plt.axes([0.50, 0.05, 0.17, 0.02]), 'E Min', -5.0, 5.0, valinit=self.erange[0])
        self.s_emax = Slider(plt.axes([0.50, 0.02, 0.17, 0.02]), 'E Max', -5.0, 5.0, valinit=self.erange[1])
        # LDOS heatmap colormap gamma (PowerNorm). 1.0 = linear; <1 squashes the
        # colormap toward the minimum. Present in all modes; placed in the bottom
        # row by _layout_bottom_row. Only re-colors (no recompute) on change.
        self.s_ldos_gamma = Slider(plt.axes([0.50, 0.08, 0.17, 0.02]), 'LDOS γ', 0.1, 1.0,
                                   valinit=self.ldos_gamma, valstep=0.05)

        # Extra Broadening UI. Each σ slider is only VISIBLE when its checkbox is
        # checked (toggled live in _on_broadening_change), and is raised above the
        # plots in zorder so it is never hidden behind one when shown.
        n_types = len(self.atomtypes)
        chk_h = max(0.04, 0.022 * n_types)
        broad_chk_init = [self.extra_broadening.get(t, 0.0) > 0 for t in self.atomtypes]
        broad_y_base = 0.90 - chk_h - 0.01
        self.chk_broad = CheckButtons(plt.axes([0.02, broad_y_base, 0.05, chk_h]), self.atomtypes, broad_chk_init)
        self.s_broad = {}
        self._sbroad_tb = {}
        for i, t in enumerate(self.atomtypes):
            y_pos = broad_y_base + chk_h - (i + 1) * (chk_h / n_types) + (chk_h / n_types - 0.016) / 2
            ax_s = plt.axes([0.08, y_pos, 0.06, 0.014])
            ax_s.set_zorder(100)
            init_val = self.extra_broadening.get(t, 0.0)
            self.s_broad[t] = Slider(ax_s, f'σ {t}', 0.0, 1.0, valinit=init_val, valstep=0.01)
            self.s_broad[t].on_changed(self._on_broadening_change)
            tb = self._add_slider_textbox(self.s_broad[t], [0.145, y_pos, 0.035, 0.014], fmt="{:.2f}")
            tb.ax.set_zorder(100)
            self._sbroad_tb[t] = tb
            # Hidden unless its checkbox is on
            ax_s.set_visible(broad_chk_init[i])
            tb.ax.set_visible(broad_chk_init[i])
        self.chk_broad.on_clicked(self._on_broadening_change)

        if self.mode == 'Single Point':
            # Columns: Global Topo | Topo h | Partitioned LDOS (no LDOS-topo)
            self.gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.22, 1], wspace=0.30)
            self.ax_map = self.fig.add_subplot(self.gs[0, 0])
            self.ax_spec = self.fig.add_subplot(self.gs[0, 2])
            self.s_num_marks = Slider(plt.axes([0.35, 0.02, 0.1, 0.03]), 'Markers', 1, 10, valinit=len(self.marker_coords), valstep=1)
            self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5)
            self.s_num_marks.on_changed(self._on_ui_change)
            # Global-topo height slider as its own gridspec column.
            self.s_topo_h, _, self.btn_topo_rerun = self._build_slider_column(
                self.gs[0, 1], 'Topo h', self.topo_height, self._on_topo_height_rerun)

        elif self.mode in ('Line', 'Path'):
            self.gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.15], hspace=0.35)
            # Top row columns: Global Topo | Topo h | LDOS h | (LDOS-topo cluster).
            # The cluster (top_gs[3]) is what _update_all rebuilds each refresh;
            # the two slider columns are built once here and left untouched.
            self.top_gs = gridspec.GridSpecFromSubplotSpec(
                1, 4, subplot_spec=self.gs[0, 0], width_ratios=[1, 0.22, 0.22, 2.5], wspace=0.30)
            self.ax_map = self.fig.add_subplot(self.top_gs[0])
            if self.mode == 'Line':
                lgs = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=self.top_gs[3], width_ratios=[1, 0.06, 0.05, 1, 0.03, 0.05], wspace=0.0)
                self.ax_line_topo = self.fig.add_subplot(lgs[0])
                self.ax_stripe, self.ax_ldos, self.cax = self.fig.add_subplot(lgs[2]), self.fig.add_subplot(lgs[3]), self.fig.add_subplot(lgs[5])
            else:  # Path: no 'Topo' box; keep the strip + LDOS heatmap, filling the cluster
                self.ax_line_topo = None
                lgs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=self.top_gs[3], width_ratios=[0.05, 1, 0.05], wspace=0.0)
                self.ax_stripe, self.ax_ldos, self.cax = self.fig.add_subplot(lgs[0]), self.fig.add_subplot(lgs[1]), self.fig.add_subplot(lgs[2])
            bot_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=self.gs[1, 0], width_ratios=[2.5, 1], wspace=0.25)
            self.ax_prof = self.fig.add_subplot(bot_gs[0])
            self.ax_spec = self.fig.add_subplot(bot_gs[1])
            self.line_art, = self.ax_map.plot([], [], 'r--', lw=2.5, zorder=5)
            if self.mode == 'Line':
                self.ends = self.ax_map.scatter([], [], c='white', edgecolors='red', s=100, zorder=10, picker=5)
            else:  # Path: draggable nodes instead of two endpoints
                self.path_pts = self.ax_map.scatter([], [], c='white', edgecolors='red', s=100, zorder=10, picker=5)
            self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5)
            self.s_num_marks = Slider(plt.axes([0.35, 0.02, 0.1, 0.03]), 'Markers', 1, 10,
                                      valinit=len(self.marker_ratios), valstep=1)
            self.s_num_marks.on_changed(self._on_ui_change)
            if self.mode == 'Path':
                self.s_path_pts = Slider(plt.axes([0.35, 0.055, 0.1, 0.03]), 'Path Pts', 2, 12,
                                         valinit=len(self.path_nodes), valstep=1)
                self.s_path_pts.on_changed(self._on_ui_change)
            # Order: Global Topo | Topo h | LDOS h | LDOS-topo | ...
            self.s_topo_h, _, self.btn_topo_rerun = self._build_slider_column(
                self.top_gs[1], 'Topo h', self.topo_height, self._on_topo_height_rerun)
            self.s_ldos_h, _, self.btn_ldos_rerun = self._build_slider_column(
                self.top_gs[2], 'LDOS h', self.ldos_height, self._on_ldos_height_rerun)

        elif self.mode == 'Map':
            # Top row columns: Global Topo | Topo h | LDOS h | Map Topo | Part. LDOS
            # Bottom row spans all columns. Topo plots keep set_aspect('equal') so
            # the slider columns do not distort them.
            self.gs = gridspec.GridSpec(2, 5, height_ratios=[2.5, 1],
                                        width_ratios=[1, 0.22, 0.22, 1, 4.0],
                                        hspace=0.35, wspace=0.05)
            self.ax_map_global = self.fig.add_subplot(self.gs[0, 0])
            self.ax_map = self.fig.add_subplot(self.gs[0, 3])
            self.ax_spec = self.fig.add_subplot(self.gs[0, 4])
            self.map_axes = []
            self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5)
            self.s_num_marks = Slider(plt.axes([0.35, 0.02, 0.1, 0.03]), 'Markers', 1, 10, valinit=len(self.marker_coords), valstep=1)
            self.s_nepts = Slider(plt.axes([0.75, 0.035, 0.15, 0.02]), 'E Pts', 1, 20, valinit=5 if self.cached_nepts is None else self.cached_nepts, valstep=1)
            self.s_num_marks.on_changed(self._on_ui_change)
            self.s_nepts.on_changed(self._on_ui_change)
            # Order: Global Topo | Topo h | LDOS h | Map Topo (LDOS-topo) | LDOS
            self.s_topo_h, _, self.btn_topo_rerun = self._build_slider_column(
                self.gs[0, 1], 'Topo h', self.topo_height, self._on_topo_height_rerun)
            self.s_ldos_h, _, self.btn_ldos_rerun = self._build_slider_column(
                self.gs[0, 2], 'LDOS h', self.ldos_height, self._on_ldos_height_rerun)

        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_rel)
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.btn_run.on_clicked(self._toggle_run)
        self.chk.on_clicked(self._on_ui_change)
        for s in [self.s_cell, self.s_emin, self.s_emax]:
            s.on_changed(self._on_ui_change)
        # Gamma only re-colors the heatmaps; the cache logic skips recompute since
        # gamma touches none of the needs_* conditions, so _on_ui_change is cheap.
        self.s_ldos_gamma.on_changed(self._on_ui_change)

        # Evenly distribute the bottom-row sliders active in THIS mode across the
        # full figure width, each with an adjacent (non-overlapping) text box.
        # Order is left-to-right. Only widgets that exist for the current mode
        # are included, so the row always spans the whole window.
        bottom_entries = [(self.s_cell, True, "{:.0f}")]
        if hasattr(self, 's_path_pts'):
            bottom_entries.append((self.s_path_pts, True, "{:.0f}"))
        bottom_entries.append((self.s_num_marks, True, "{:.0f}"))
        if hasattr(self, 's_nepts'):
            bottom_entries.append((self.s_nepts, True, "{:.0f}"))
        bottom_entries.append((self.s_emin, True, "{:.3f}"))
        bottom_entries.append((self.s_emax, True, "{:.3f}"))
        bottom_entries.append((self.s_ldos_gamma, True, "{:.2f}"))
        self._layout_bottom_row(bottom_entries)

        self.cached_emin, self.cached_emax = None, None
        self._update_all(full_refresh=True)

    # ---- The remaining UI callbacks and _update_all are character-identical ----
    # ---- to the original, except show_mag display logic uses the new helpers ----

    def _on_mode_change(self, label):
        self.mode = label
        self.active_marker_idx = 0
        self.plot_level = 0
        self._build_ui()

    def _toggle_run(self, event):
        self.is_running = not self.is_running
        self.btn_run.label.set_text('STOP' if self.is_running else 'RUN')
        if self.is_running:
            self._update_all()
        else:
            if self.mode == 'Line' and self.p1 is not None and self.p2 is not None:
                v = self.p2 - self.p1
                m_pos = [self.p1 + r * v for r in self.marker_ratios]
                print("\n--- Line Endpoints & Markers (copy-paste into run_interactive) ---")
                print(f"line_endpoints=([{self.p1[0]:.6f}, {self.p1[1]:.6f}], [{self.p2[0]:.6f}, {self.p2[1]:.6f}]),")
                pos_str = ", ".join(f"[{p[0]:.6f}, {p[1]:.6f}]" for p in m_pos)
                print(f"marker_positions=[{pos_str}]")
            elif self.mode == 'Path' and self.path_nodes is not None:
                p_xy, _, _, _ = self._build_path_xy()
                mark_idx = (np.array(self.marker_ratios) * (self.npts - 1)).astype(int)
                m_pos = p_xy[mark_idx]
                print("\n--- Path Nodes & Markers (current dragged positions) ---")
                node_str = ", ".join(f"[{n[0]:.6f}, {n[1]:.6f}]" for n in self.path_nodes)
                print(f"# path_nodes (free XY, no longer atom indices): [{node_str}]")
                print(f"# path_extend={self.path_extend}")
                pos_str = ", ".join(f"[{p[0]:.6f}, {p[1]:.6f}]" for p in m_pos)
                print(f"marker_positions=[{pos_str}]")
            elif self.mode in ['Single Point', 'Map']:
                print("\n--- Marker Positions (copy-paste into run_interactive) ---")
                pos_str = ", ".join(f"[{p[0]:.6f}, {p[1]:.6f}]" for p in self.marker_coords)
                print(f"marker_positions=[{pos_str}]")

    def _recompute_global_topo(self):
        """Re-run Phase 1 global topography with current broadened DOS (cache-aware)."""
        grid_res = int(np.sqrt(len(self.grid_xy)))
        cache_name = f"global_topo_{self.global_topo_bias}V_{self.topo_height}A_{grid_res}px{self._broadening_cache_suffix()}{self._angular_cache_suffix()}{self._unit_cell_cache_suffix()}{self._decay_cache_suffix()}.npy"
        if exists(cache_name):
            print(f"[*] Loading Cached Global Topography: {cache_name}")
            self.current_z_map = np.load(cache_name)
        else:
            print(f"[*] Calculating Global Topography with broadening: {cache_name}")
            t_emin, t_emax = sorted([0.0, self.global_topo_bias])
            z_fixed = cp.full(self.grid_xy_gpu.shape[0], self.z_highest_atom + self.topo_height, dtype=cp.float32)
            ld_1, _, init_engs = self._calculate_ldos_at_points_gpu(
                cp.hstack([self.grid_xy_gpu, z_fixed[:, None]]), t_emin, t_emax,
                use_energy_decay=self.use_decay_topo, preserve_orbitals=False, topo_only=True)
            target_setp = cp.max(gpu_simpson(ld_1, init_engs))
            print(f"[*] Global Setpoint LDOS: {float(target_setp):.6e}")
            z_map_gpu = self._converge_tip_height(z_fixed, self.grid_xy_gpu, t_emin, t_emax, target_setp, use_decay=self.use_decay_topo)
            self.current_z_map = cp.asnumpy(z_map_gpu)
            np.save(cache_name, self.current_z_map)
        self.global_z_map = self.current_z_map.copy()

    def _on_topo_height_rerun(self, event):
        """Apply the global-topo tip-height slider and recompute the global topo.
        Static-label, single-action button. Does nothing unless the main RUN
        button is active. Never deletes/overwrites caches (height-encoded
        filenames are load-or-create)."""
        if not self.is_running:
            return
        self.topo_height = float(self.s_topo_h.val)
        self._recompute_global_topo()
        self._broadening_topo_dirty = False
        self._update_all(full_refresh=True)

    def _on_ldos_height_rerun(self, event):
        """Apply the LDOS-topo tip-height slider and recompute LDOS topo.
        Static-label, single-action button. Does nothing unless the main RUN
        button is active. Resets only in-RAM frame caches (forces recompute /
        reload at the new height); never deletes/overwrites disk caches."""
        if not self.is_running:
            return
        self.ldos_height = float(self.s_ldos_h.val)
        # Invalidate in-RAM caches that depend on LDOS tip height
        self.cached_p1 = self.cached_p2 = None
        self.cached_path_nodes = None
        self.cached_bias_energy_line = self.cached_bias_energy_map = None
        self.cached_d_topo_line = self.cached_d_topo_map = None
        self.cached_emin = self.cached_emax = None
        self.cached_ld_up = self.cached_ld_dn = None
        self.cached_marker_coords = self.cached_spec_ldos = None
        self._update_all(full_refresh=True)

    def _on_broadening_change(self, val):
        states = self.chk_broad.get_status()
        for i, t in enumerate(self.atomtypes):
            # Show the σ slider (and its textbox) only while its checkbox is on.
            if t in self.s_broad:
                self.s_broad[t].ax.set_visible(states[i])
            if t in getattr(self, '_sbroad_tb', {}):
                self._sbroad_tb[t].ax.set_visible(states[i])
            self.extra_broadening[t] = self.s_broad[t].val if states[i] else 0.0
        self._apply_extra_broadening()
        self._sync_textboxes()
        self.cached_p1 = self.cached_p2 = None
        self.cached_emin = self.cached_emax = None
        self.cached_d_topo_line = self.cached_d_topo_map = None
        self.cached_d_ldos = None
        self.cached_bias_energy_line = self.cached_bias_energy_map = None
        self.cached_marker_coords = self.cached_spec_ldos = None
        self.cached_ld_up = self.cached_ld_dn = None
        self._broadening_topo_dirty = True
        if self.is_running:
            self._recompute_global_topo()
            self._broadening_topo_dirty = False
            self._update_all(full_refresh=True)

    def _on_ui_change(self, val):
        prev_use_angular = self.use_angular
        prev_path_extend = self.path_extend
        states = self.chk.get_status()
        for (label, attr), state in zip(self._toggle_defs, states):
            setattr(self, attr, state)
        self.display_cells = int(self.s_cell.val)
        if hasattr(self, 's_ldos_gamma'):
            self.ldos_gamma = float(self.s_ldos_gamma.val)
        self._sync_textboxes()

        # Angular toggle OR path-extend toggle: invalidate caches and recompute.
        if (self.use_angular != prev_use_angular) or (self.path_extend != prev_path_extend):
            self.cached_p1 = self.cached_p2 = None
            self.cached_path_nodes = None
            self.cached_emin = self.cached_emax = None
            self.cached_d_topo_line = self.cached_d_topo_map = None
            self.cached_d_ldos = None
            self.cached_bias_energy_line = self.cached_bias_energy_map = None
            self.cached_marker_coords = self.cached_spec_ldos = None
            self.cached_ld_up = self.cached_ld_dn = None
            # Only the angular toggle changes the global topo; path-extend does not.
            if self.use_angular != prev_use_angular:
                self._broadening_topo_dirty = True
                if self.is_running:
                    self._recompute_global_topo()
                    self._broadening_topo_dirty = False
            if self.is_running:
                self._update_all(full_refresh=True)
            return

        # Path Pts slider: adjust number of draggable path nodes.
        if self.mode == 'Path' and hasattr(self, 's_path_pts'):
            new_nodes = int(self.s_path_pts.val)
            if new_nodes != len(self.path_nodes):
                if new_nodes > len(self.path_nodes):
                    # Append nodes spread along the current path arc length.
                    p_xy, _, _, _ = self._build_path_xy()
                    n_add = new_nodes - len(self.path_nodes)
                    ratios = np.linspace(0.15, 0.85, n_add)
                    for r in ratios:
                        idx = int(r * (self.npts - 1))
                        self.path_nodes.append(np.array(p_xy[idx], dtype=float))
                else:
                    self.path_nodes = self.path_nodes[:new_nodes]
                self.cached_path_nodes = None

        new_count = int(self.s_num_marks.val)
        if self.mode in ('Line', 'Path'):
            if new_count != len(self.marker_ratios):
                if new_count > len(self.marker_ratios):
                    self.marker_ratios = list(np.linspace(0.1, 0.9, new_count))
                else:
                    self.marker_ratios = self.marker_ratios[:new_count]
        else:
            if new_count != len(self.marker_coords):
                if new_count > len(self.marker_coords):
                    for _ in range(new_count - len(self.marker_coords)):
                        self.marker_coords.append([self.lv[0, 0] * 0.5 + (np.random.rand() - 0.5) * 0.1,
                                                   self.lv[1, 1] * 0.5 + (np.random.rand() - 0.5) * 0.1])
                else:
                    self.marker_coords = self.marker_coords[:new_count]
                self.cached_marker_coords = None

        self._update_all(full_refresh=True)

    def _update_all(self, full_refresh=False):
        if full_refresh:
            self.ax_map.clear()
            n = int(self.s_cell.val)
            if self.mode == 'Map':
                self.ax_map_global.clear()
            t_ax = self.ax_map_global if self.mode == 'Map' else self.ax_map
            z_data = self.global_z_map

            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    off = i * self.lv[0, :2] + j * self.lv[1, :2]
                    t_ax.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], z_data, levels=60, cmap=self.cmap_topo, zorder=1)
                    if self.mode == 'Map':
                        self.ax_map.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], self.current_z_map, levels=60, cmap=self.cmap_topo, zorder=1)
                    if self.show_atoms:
                        tr = np.repeat(self.atomtypes, self.atomnums)
                        for t_idx, t_name in enumerate(self.atomtypes):
                            m = (tr == t_name)
                            t_ax.scatter(self.coord[m, 0] + off[0], self.coord[m, 1] + off[1], s=10, color=plt.cm.tab10(t_idx / 10), alpha=0.3, zorder=2)
                            if self.mode == 'Map':
                                self.ax_map.scatter(self.coord[m, 0] + off[0], self.coord[m, 1] + off[1], s=10, color=plt.cm.tab10(t_idx / 10), alpha=0.3, zorder=2)
            if self.show_unit_cell:
                v0, v1, v2, v3 = np.array([0, 0]), self.lv[0, :2], self.lv[0, :2] + self.lv[1, :2], self.lv[1, :2]
                cell_pts = np.array([v0, v1, v2, v3, v0])
                t_ax.plot(cell_pts[:, 0], cell_pts[:, 1], color='cyan', lw=2.0, ls='-', zorder=4, label='Unit Cell')
                if self.mode == 'Map':
                    self.ax_map.plot(cell_pts[:, 0], cell_pts[:, 1], color='cyan', lw=2.0, ls='-', zorder=4, label='Unit Cell')

            t_ax.set_aspect('equal')
            t_ax.set_title(f"Global Topo\nBias: {self.global_topo_bias} V\nHeight: {self.topo_height} Å")
            if self.mode == 'Map':
                self.ax_map.set_aspect('equal')
                self.ax_map.set_title(f"Map Topo\nBias: {self.s_emin.val if str(self.ldos_bias_sign).lower() in ['neg', '-', 'negative'] else self.s_emax.val} V\nHeight: {self.ldos_height} Å")
            if self.mode in ('Line', 'Path'):
                self.ax_map.add_line(self.line_art)
                if self.mode == 'Line':
                    self.ax_map.add_collection(self.ends)
                else:
                    self.ax_map.add_collection(self.path_pts)
            self.ax_map.add_collection(self.marks)

        if self.mode == 'Line':
            v = self.p2 - self.p1
            p_len = norm(v)
            p_dist = np.linspace(0, p_len, self.npts)
            p_xy = np.array([self.p1 + r * v for r in np.linspace(0, 1, self.npts)])
            self.line_art.set_data([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]])
            self.ends.set_offsets([self.p1, self.p2])
            self.marks.set_offsets(np.array([self.p1 + r * v for r in self.marker_ratios]))
            self.marks.set_facecolors(self.m_colors[:len(self.marker_ratios)])
        elif self.mode == 'Path':
            p_xy, p_dist, p_len, verts = self._build_path_xy()
            self.line_art.set_data(verts[:, 0], verts[:, 1])
            self.path_pts.set_offsets(np.array(self.path_nodes))
            mark_idx = (np.array(self.marker_ratios) * (self.npts - 1)).astype(int)
            self.marks.set_offsets(p_xy[mark_idx])
            self.marks.set_facecolors(self.m_colors[:len(self.marker_ratios)])
        else:
            self.marks.set_offsets(self.marker_coords)
            self.marks.set_facecolors(self.m_colors[:len(self.marker_coords)])

        if not self.is_running:
            self.fig.canvas.draw_idle()
            return

        if getattr(self, '_broadening_topo_dirty', False):
            self._recompute_global_topo()
            self._broadening_topo_dirty = False
            self._update_all(full_refresh=True)
            return

        bias_e = self.s_emin.val if str(self.ldos_bias_sign).lower() in ['neg', '-', 'negative'] else self.s_emax.val
        nepts = int(self.s_nepts.val) if hasattr(self, 's_nepts') else None
        needs_topo = ((self.mode == 'Line' and (self.cached_p1 is None or not np.array_equal(self.p1, self.cached_p1) or not np.array_equal(self.p2, self.cached_p2) or self.cached_bias_energy_line != bias_e or self.cached_d_topo_line != self.use_decay_topo)) or
                      (self.mode == 'Path' and (self.cached_path_nodes is None or not np.array_equal(np.array(self.path_nodes), self.cached_path_nodes) or self.cached_path_extend != self.path_extend or self.cached_bias_energy_line != bias_e or self.cached_d_topo_line != self.use_decay_topo)) or
                      (self.mode == 'Map' and (self.cached_bias_energy_map != bias_e or self.cached_d_topo_map != self.use_decay_topo)))
        needs_ldos = (needs_topo or self.cached_emin != self.s_emin.val or self.cached_emax != self.s_emax.val or
                      self.cached_d_ldos != self.use_decay_ldos or getattr(self, 'cached_mode', None) != self.mode or
                      (self.mode == 'Map' and self.cached_nepts != nepts))
        needs_spec = (needs_ldos or (self.mode in ['Single Point', 'Map'] and (self.cached_marker_coords is None or not np.array_equal(self.marker_coords, self.cached_marker_coords))))

        if needs_topo:
            if self.mode in ('Line', 'Path'):
                l_emin, l_emax = sorted([0.0, bias_e])
                p_xy_gpu = cp.array(p_xy, dtype=cp.float32)
                ld_1, _, l_engs = self._calculate_ldos_at_points_gpu(
                    cp.hstack([p_xy_gpu, cp.full((self.npts, 1), self.z_highest_atom + self.ldos_height, dtype=cp.float32)]),
                    l_emin, l_emax, use_energy_decay=self.use_decay_topo, preserve_orbitals=False, topo_only=True)
                target = cp.max(gpu_simpson(ld_1, l_engs))
                print(f"[*] Path Setpoint LDOS: {float(target):.6e}")
                z_line = self._converge_tip_height(cp.full(self.npts, self.z_highest_atom + self.ldos_height, dtype=cp.float32), p_xy_gpu, l_emin, l_emax, target, use_decay=self.use_decay_topo)
                self.current_z_line = cp.asnumpy(z_line)
                if self.mode == 'Line':
                    self.cached_p1, self.cached_p2 = self.p1.copy(), self.p2.copy()
                else:
                    self.cached_path_nodes = np.array(self.path_nodes).copy()
                    self.cached_path_extend = self.path_extend
                self.cached_bias_energy_line, self.cached_d_topo_line = bias_e, self.use_decay_topo
            elif self.mode == 'Map':
                t_emin, t_emax = sorted([0.0, bias_e])
                grid_res = int(np.sqrt(len(self.grid_xy)))
                ldos_cache_name = f"ldos_topo_{bias_e}V_{self.ldos_height}A_{grid_res}px{self._broadening_cache_suffix()}{self._angular_cache_suffix()}{self._unit_cell_cache_suffix()}{self._decay_cache_suffix()}.npy"
                if exists(ldos_cache_name):
                    print(f"[*] Loading Cached LDOS Topography: {ldos_cache_name}")
                    self.current_z_map = np.load(ldos_cache_name)
                else:
                    print(f"[*] Cache not found. Calculating LDOS Topography: {ldos_cache_name}")
                    z_fixed = cp.full(self.grid_xy_gpu.shape[0], self.z_highest_atom + self.ldos_height, dtype=cp.float32)
                    ld_1, _, init_engs = self._calculate_ldos_at_points_gpu(
                        cp.hstack([self.grid_xy_gpu, z_fixed[:, None]]), t_emin, t_emax,
                        use_energy_decay=self.use_decay_topo, preserve_orbitals=False, topo_only=True)
                    target_setp = cp.max(gpu_simpson(ld_1, init_engs))
                    print(f"[*] Map Local Setpoint LDOS: {float(target_setp):.6e}")
                    z_map_gpu = self._converge_tip_height(z_fixed, self.grid_xy_gpu, t_emin, t_emax, target_setp, use_decay=self.use_decay_topo)
                    self.current_z_map = cp.asnumpy(z_map_gpu)
                    np.save(ldos_cache_name, self.current_z_map)
                self.cached_bias_energy_map, self.cached_d_topo_map = bias_e, self.use_decay_topo
                self.ax_map.clear()
                n = int(self.s_cell.val)
                for i in range(-n, n + 1):
                    for j in range(-n, n + 1):
                        off = i * self.lv[0, :2] + j * self.lv[1, :2]
                        self.ax_map.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], self.current_z_map, levels=60, cmap=self.cmap_topo, zorder=1)
                self.ax_map.set_aspect('equal')
                self.ax_map.add_collection(self.marks)

        if needs_ldos:
            if self.mode in ('Line', 'Path'):
                ld_up, ld_dn, eg = self._calculate_ldos_at_points_gpu(
                    np.hstack([p_xy, self.current_z_line[:, None]]),
                    self.s_emin.val, self.s_emax.val,
                    use_energy_decay=self.use_decay_ldos, preserve_orbitals=True)
            elif self.mode == 'Map':
                self.map_e_targets = np.linspace(self.s_emin.val, self.s_emax.val, nepts)
                eg = cp.array(self.energies[np.searchsorted(self.energies, self.s_emin.val):np.searchsorted(self.energies, self.s_emax.val, side='right')])
                ld_up_list, ld_dn_list = [], []
                grid_z = cp.hstack([self.grid_xy_gpu, cp.array(self.current_z_map)[:, None]])
                for t_e in self.map_e_targets:
                    e_idx = np.searchsorted(self.energies, t_e)
                    t_up, t_dn, _ = self._calculate_ldos_at_points_gpu(
                        grid_z, self.energies[e_idx],
                        self.energies[min(e_idx + 1, len(self.energies) - 1)] + 1e-6,
                        use_energy_decay=self.use_decay_ldos, preserve_orbitals=True,
                        global_bias=abs(self.s_emax.val - self.s_emin.val))
                    ld_up_list.append(t_up[:, 0:1])
                    if t_dn is not None:
                        ld_dn_list.append(t_dn[:, 0:1])
                ld_up = cp.concatenate(ld_up_list, axis=1)
                ld_dn = cp.concatenate(ld_dn_list, axis=1) if ld_dn_list else None
            else:
                eg = cp.array(self.energies[np.searchsorted(self.energies, self.s_emin.val):np.searchsorted(self.energies, self.s_emax.val, side='right')])
                ld_up, ld_dn = None, None

            if ld_up is not None:
                self.cached_ld_up = cp.asnumpy(ld_up)
                self.cached_ld_dn = cp.asnumpy(ld_dn) if ld_dn is not None else None
            self.cached_eg = cp.asnumpy(eg)
            self.cached_emin, self.cached_emax = self.s_emin.val, self.s_emax.val
            self.cached_d_ldos, self.cached_nepts = self.use_decay_ldos, nepts
            self.cached_mode = self.mode

        if needs_spec and self.mode in ['Single Point', 'Map']:
            m_coords = np.array(self.marker_coords)
            z_marks = []
            inv_lv_np = inv(self.lv)
            for pt in m_coords:
                pt_3d = np.array([pt[0], pt[1], 0.0])
                f_pt = np.dot(pt_3d, inv_lv_np)
                f_pt[:2] = f_pt[:2] % 1.0
                wrapped_pt = np.dot(f_pt, self.lv)
                dist_sq = (self.grid_xy[:, 0] - wrapped_pt[0]) ** 2 + (self.grid_xy[:, 1] - wrapped_pt[1]) ** 2
                z_marks.append(self.current_z_map[np.argmin(dist_sq)])
            z_marks = np.array(z_marks)

            pt_gpu = cp.array(np.hstack([m_coords, z_marks[:, None]]), dtype=cp.float32)
            s_up, s_dn, _ = self._calculate_ldos_at_points_gpu(
                pt_gpu, self.s_emin.val, self.s_emax.val,
                use_energy_decay=self.use_decay_ldos, preserve_orbitals=True)
            s_up_np = cp.asnumpy(s_up)
            s_dn_np = cp.asnumpy(s_dn) if s_dn is not None else None

            spec_ldos = self._combine_ldos_display(s_up_np, s_dn_np, self.show_mag)
            self.cached_spec_ldos = spec_ldos
            self.cached_marker_coords = np.array(self.marker_coords).copy()

        if self.mode in ['Line', 'Path', 'Map'] and self.cached_ld_up is not None:
            f_up = self.cached_ld_up.copy()
            f_dn = self.cached_ld_dn.copy() if self.cached_ld_dn is not None else None
            f_ldos_raw = self._combine_ldos_display(f_up, f_dn, self.show_mag)
            partitions = self._get_partitions(f_ldos_raw)
        else:
            f_ldos_raw = None
            partitions = []

        # --- Determine whether mag data is signed (for colormap selection) ---
        mag_signed = self._mag_is_signed()
        has_mag = self._has_mag_channel()
        use_diverging = self.show_mag and has_mag and mag_signed

        if self.mode in ('Line', 'Path'):
            active_idx = int(self.marker_ratios[min(self.active_marker_idx, len(self.marker_ratios) - 1)] * (self.npts - 1))
            active_ldos = f_ldos_raw[active_idx] if f_ldos_raw is not None else np.zeros_like(self.cached_spec_ldos[0])
        else:
            active_idx = min(self.active_marker_idx, len(self.marker_coords) - 1)
            active_ldos = self.cached_spec_ldos[active_idx]

        self.ax_spec.clear()

        def _orbit_base(orb):
            if orb.endswith('_up'):
                return orb[:-3]
            elif orb.endswith('_down'):
                return orb[:-5]
            return orb

        def _lighten_color(color, amount=0.3):
            c = mc.to_rgb(color)
            h, l, s = colorsys.rgb_to_hls(*c)
            return colorsys.hls_to_rgb(h, min(1, l + amount * (1 - l)), s)

        unique_bases = sorted(set(_orbit_base(o) for o in self.orbitals))
        styles = ['-', '--', ':', '-.'] + [(0, (3 + i, 2)) for i in range(max(0, len(unique_bases) - 4))]
        linestyle_map = dict(zip(unique_bases, styles))
        S = 0.25

        active_ldos_norm = active_ldos.copy()
        if self.normalize:
            total_for_norm = np.sum(active_ldos_norm, axis=(1, 2))
            norm_factor = (np.trapezoid(total_for_norm, x=self.cached_eg) + 1e-15)
            active_ldos_norm /= norm_factor

        if self.plot_level == 0:
            if self.mode in ('Line', 'Path'):
                for i, r in enumerate(self.marker_ratios):
                    idx = int(r * (self.npts - 1))
                    c_ldos = f_ldos_raw[idx].copy()
                    t_y = np.sum(c_ldos, axis=(1, 2))
                    if self.normalize:
                        t_y /= (np.trapezoid(t_y, x=self.cached_eg) + 1e-15)
                    color = self.m_colors[i % len(self.m_colors)]
                    self.ax_spec.plot(self.cached_eg, t_y, color=color, lw=2.5, picker=True, pickradius=5, label=f'marker_{i}')
            else:
                for i, pt in enumerate(self.marker_coords):
                    c_ldos = self.cached_spec_ldos[i].copy()
                    t_y = np.sum(c_ldos, axis=(1, 2))
                    if self.normalize:
                        t_y /= (np.trapezoid(t_y, x=self.cached_eg) + 1e-15)
                    color = self.m_colors[i % len(self.m_colors)]
                    self.ax_spec.plot(self.cached_eg, t_y, color=color, lw=2.5, picker=True, pickradius=5, label=f'marker_{i}')
            self.ax_spec.legend(loc='upper right', frameon=False)
        elif self.plot_level == 1:
            self.ax_spec.plot(self.cached_eg, np.sum(active_ldos_norm, axis=(1, 2)), color='black', lw=1.5, alpha=0.1, zorder=1)
            atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
            au_idx = [i for i, t in enumerate(atom_types_exp) if t == 'Au']
            mol_idx = [i for i, t in enumerate(atom_types_exp) if t != 'Au']
            au_y = np.sum(active_ldos_norm[:, au_idx, :], axis=(1, 2)) if au_idx else np.zeros_like(self.cached_eg)
            mol_y = np.sum(active_ldos_norm[:, mol_idx, :], axis=(1, 2)) if mol_idx else np.zeros_like(self.cached_eg)
            self.ax_spec.plot(self.cached_eg, au_y, color='orange', lw=2, label='Au', picker=True, pickradius=5, zorder=2)
            self.ax_spec.plot(self.cached_eg, mol_y, color='black', lw=2, label='Molecule', picker=True, pickradius=5, zorder=2)
            self.ax_spec.legend([Line2D([0], [0], color='orange', lw=2), Line2D([0], [0], color='black', lw=2)], ['Au', 'Molecule'], title="Partition", loc='upper right', frameon=False)
        elif self.plot_level == 2:
            atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
            for t in self.atomtypes:
                t_idx = [i for i, x in enumerate(atom_types_exp) if x == t]
                t_y = np.sum(active_ldos_norm[:, t_idx, :], axis=(1, 2)) if t_idx else np.zeros_like(self.cached_eg)
                self.ax_spec.plot(self.cached_eg, t_y, color=self._type_color_map.get(t, 'grey'), lw=2, label=t, picker=True, pickradius=3)
            self.ax_spec.legend([Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2) for t in self.atomtypes], self.atomtypes, title="Atom Types", loc='upper right', frameon=False)
        elif self.plot_level == 3:
            atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
            e_idx = [i for i, x in enumerate(atom_types_exp) if x == self.active_element]
            e_y = np.sum(active_ldos_norm[:, e_idx, :], axis=(1, 2)) if e_idx else np.zeros_like(self.cached_eg)
            self.ax_spec.plot(self.cached_eg, e_y, color=self._type_color_map.get(self.active_element, 'grey'), lw=2, alpha=0.15, zorder=1)
            for a_idx, label in self.vesta_label_map.items():
                if label.startswith(self.active_element):
                    y_sum = np.sum(active_ldos_norm[:, a_idx - 1, :], axis=1)
                    self.ax_spec.plot(self.cached_eg, y_sum, color=self._type_color_map.get(self.active_element, 'grey'), lw=2, label=label, picker=True, pickradius=3, zorder=2)
            self.ax_spec.legend([Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2) for t in self.atomtypes], self.atomtypes, title="Atom Types", loc='upper right', frameon=False)
        elif self.plot_level == 4:
            for a_idx, label in self.vesta_label_map.items():
                if label.startswith(self.active_element):
                    y_sum = np.sum(active_ldos_norm[:, a_idx - 1, :], axis=1)
                    element_color = self._type_color_map.get(self.active_element, 'grey')
                    if a_idx == self.active_atom:
                        self.ax_spec.plot(self.cached_eg, y_sum, color=element_color, lw=2.5, alpha=1.0, zorder=5)
                        orb_artists = []
                        for orb in self.orbitals:
                            col_idx = self.orbitals.index(orb)
                            y_orb = active_ldos_norm[:, a_idx - 1, col_idx]
                            ls = linestyle_map[_orbit_base(orb)]
                            p_color = _lighten_color(element_color, 0.3) if orb.endswith('_up') else element_color
                            o_line, = self.ax_spec.plot(self.cached_eg, y_orb, color=p_color, linestyle=ls, lw=1.2, label=f"{label} – {orb}", zorder=10)
                            orb_artists.append(o_line)
                        if hasattr(self, 'cursor'):
                            self.cursor.remove()
                        self.cursor = mplcursors.cursor(orb_artists, hover=True)
                        self.cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
                    else:
                        orig = mc.to_rgb(element_color)
                        lumi = 0.299 * orig[0] + 0.587 * orig[1] + 0.114 * orig[2]
                        faded_color = (S * np.array(orig)) + ((1 - S) * lumi)
                        self.ax_spec.plot(self.cached_eg, y_sum, color=faded_color, lw=1.5, alpha=0.05, zorder=2)
            leg1 = self.ax_spec.legend([Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2) for t in self.atomtypes], self.atomtypes, title="Atom Types", loc='upper right', frameon=False)
            self.ax_spec.add_artist(leg1)
            self.ax_spec.legend([Line2D([0], [0], color='black', linestyle=linestyle_map[b], lw=1.5) for b in unique_bases], unique_bases, title="Orbitals", loc='upper left', frameon=False)

        lines = self.ax_spec.get_lines()
        active_maxes = [np.max(l.get_ydata()) for l in lines if l.get_visible() and l.get_alpha() == 1.0]
        if active_maxes:
            self.ax_spec.set_ylim(0, max(active_maxes) * 1.1)

        p_dist_val = p_dist[active_idx] if self.mode in ('Line', 'Path') else f"[{self.marker_coords[active_idx][0]:.1f}, {self.marker_coords[active_idx][1]:.1f}]"
        self.ax_spec.set(title=f"Partitioned LDOS (Marker {self.active_marker_idx}: {p_dist_val} Å)", xlabel="Energy (eV)")
        if self.mode in ('Line', 'Path'):
            self.ax_spec.set_box_aspect(1)

        if self.mode == 'Map':
            if not hasattr(self, 'map_e_targets') or len(self.map_e_targets) != nepts or full_refresh:
                self.map_e_targets = np.linspace(self.s_emin.val, self.s_emax.val, nepts)
            for i, e_val in enumerate(self.map_e_targets):
                self.ax_spec.axvline(x=e_val, color='black', linestyle='-', lw=2, picker=5, label=f'emarker_{i}')

        if self.mode in ('Line', 'Path'):
            if hasattr(self, 'line_decomp_axes'):
                for ax in self.line_decomp_axes:
                    if ax in self.fig.axes:
                        ax.remove()
            self.ax_prof.clear()
            if getattr(self, 'ax_ldos', None) and self.ax_ldos in self.fig.axes:
                self.ax_ldos.remove()
                self.ax_ldos = None
            if getattr(self, 'ax_stripe', None) and self.ax_stripe in self.fig.axes:
                self.ax_stripe.remove()
                self.ax_stripe = None
            if getattr(self, 'ax_line_topo', None) and self.ax_line_topo in self.fig.axes:
                self.ax_line_topo.remove()
                self.ax_line_topo = None
            if getattr(self, 'cax', None) and self.cax in self.fig.axes:
                self.cax.remove()
                self.cax = None
            import matplotlib.ticker as ticker
            self.line_decomp_axes = []

            processed_partitions = []
            global_vmax = 0.0

            if self.normalize and f_ldos_raw is not None:
                total_ldos_line = np.sum(f_ldos_raw, axis=(2, 3))
                norm_denom_line = np.trapezoid(total_ldos_line, x=self.cached_eg, axis=1)[:, None] + 1e-15
            else:
                norm_denom_line = 1.0

            for p_label, p_data in partitions:
                t_data = p_data.copy()
                if self.normalize:
                    t_data /= norm_denom_line
                processed_partitions.append((p_label, t_data))
                v_max = np.max(np.abs(t_data))
                if v_max > global_vmax:
                    global_vmax = v_max
            if global_vmax == 0:
                global_vmax = 1e-15

            num_p = max(1, len(partitions))
            # In Line mode the cluster leads with the 'Topo' box + spacer columns
            # (offset 3) then stripe at index 2. In Path mode the 'Topo' box is
            # removed: the cluster leads with just the stripe column.
            if self.mode == 'Line':
                if self.show_dcmp_norm:
                    w_ratios = [1, 0.06, 0.05] + [1, 0.05, 0.1] * num_p
                else:
                    w_ratios = [1, 0.06, 0.05] + [1, 0.0, 0.0] * (num_p - 1) + [1, 0.05, 0.0]
                lgs = gridspec.GridSpecFromSubplotSpec(1, 3 + num_p * 3, subplot_spec=self.top_gs[3], width_ratios=w_ratios, wspace=0.0)
                stripe_idx = 2
                decomp_base = 3
            else:  # Path: no Topo box / spacer; stripe leads
                if self.show_dcmp_norm:
                    w_ratios = [0.05, 0.05] + [1, 0.05, 0.1] * num_p
                else:
                    w_ratios = [0.05, 0.05] + [1, 0.0, 0.0] * (num_p - 1) + [1, 0.05, 0.0]
                lgs = gridspec.GridSpecFromSubplotSpec(1, 2 + num_p * 3, subplot_spec=self.top_gs[3], width_ratios=w_ratios, wspace=0.0)
                stripe_idx = 0
                decomp_base = 2

            if self.mode == 'Line':
                # Line Topo panel (the unwrapped 'Topo' corridor) — Line mode only.
                self.ax_line_topo = self.fig.add_subplot(lgs[0])
                self.line_decomp_axes.append(self.ax_line_topo)

                ax_spacer = self.fig.add_subplot(lgs[1])
                ax_spacer.axis('off')
                self.line_decomp_axes.append(ax_spacer)

                v_line = self.p2 - self.p1
                v_hat = v_line / p_len
                v_perp = np.array([-v_hat[1], v_hat[0]])
                half_w = p_len / 2.0

                n_tile = max(int(self.s_cell.val), 1) + 1
                margin = p_len * 0.15
                all_rx, all_ry, all_z = [], [], []
                for i in range(-n_tile, n_tile + 1):
                    for j in range(-n_tile, n_tile + 1):
                        off = i * self.lv[0, :2] + j * self.lv[1, :2]
                        pts = self.grid_xy + off
                        dx = pts[:, 0] - self.p1[0]
                        dy = pts[:, 1] - self.p1[1]
                        rot_x = dx * v_perp[0] + dy * v_perp[1]
                        rot_y = dx * v_hat[0] + dy * v_hat[1]
                        mask = (rot_x > -half_w - margin) & (rot_x < half_w + margin) & (rot_y > -margin) & (rot_y < p_len + margin)
                        all_rx.append(rot_x[mask])
                        all_ry.append(rot_y[mask])
                        all_z.append(self.global_z_map[mask])
                all_rx, all_ry, all_z = np.concatenate(all_rx), np.concatenate(all_ry), np.concatenate(all_z)
                self.ax_line_topo.tricontourf(all_rx, all_ry, all_z, levels=60, cmap=self.cmap_topo, zorder=1)

                if self.show_atoms:
                    tr = np.repeat(self.atomtypes, self.atomnums)
                    for t_idx, t_name in enumerate(self.atomtypes):
                        m = (tr == t_name)
                        a_rx_all, a_ry_all = [], []
                        for i_tile in range(-n_tile, n_tile + 1):
                            for j_tile in range(-n_tile, n_tile + 1):
                                off = i_tile * self.lv[0, :2] + j_tile * self.lv[1, :2]
                                a_dx = self.coord[m, 0] + off[0] - self.p1[0]
                                a_dy = self.coord[m, 1] + off[1] - self.p1[1]
                                a_rx = a_dx * v_perp[0] + a_dy * v_perp[1]
                                a_ry = a_dx * v_hat[0] + a_dy * v_hat[1]
                                vis = (a_rx > -half_w) & (a_rx < half_w) & (a_ry > 0) & (a_ry < p_len)
                                a_rx_all.append(a_rx[vis])
                                a_ry_all.append(a_ry[vis])
                        if a_rx_all:
                            self.ax_line_topo.scatter(np.concatenate(a_rx_all), np.concatenate(a_ry_all), s=10, color=plt.cm.tab10(t_idx / 10), alpha=0.3, zorder=2)

                self.ax_line_topo.plot([0, 0], [0, p_len], color='red', ls='--', lw=2.0, zorder=6)
                self.ax_line_topo.scatter([0], [0], c='white', edgecolors='red', s=60, zorder=8)
                self.ax_line_topo.scatter([0], [p_len], c='white', edgecolors='red', s=60, zorder=8)

                for i, r in enumerate(self.marker_ratios):
                    color = self.m_colors[i % len(self.m_colors)]
                    self.ax_line_topo.scatter([0], [r * p_len], c=color, edgecolors='black', s=80, zorder=10)

                self.ax_line_topo.set_xlim(-half_w, half_w)
                self.ax_line_topo.set_ylim(0, p_len)
                self.ax_line_topo.set_box_aspect(1)
                self.ax_line_topo.set_xlabel("Perp. (Å)", fontsize=10)
                self.ax_line_topo.set_ylabel("Position (Å)", fontsize=10)
                self.ax_line_topo.set_title("Topo", fontsize=10)
            else:
                self.ax_line_topo = None

            # Topo stripe
            self.ax_stripe = self.fig.add_subplot(lgs[stripe_idx])
            self.line_decomp_axes.append(self.ax_stripe)
            lc = LineCollection(
                np.array([np.array([np.zeros_like(p_dist), p_dist]).T[:-1],
                          np.array([np.zeros_like(p_dist), p_dist]).T[1:]]).transpose(1, 0, 2),
                cmap=self.cmap_topo,
                norm=plt.Normalize(self.current_z_line.min(), self.current_z_line.max()), linewidth=40)
            lc.set_array(self.current_z_line[:-1])
            self.ax_stripe.add_collection(lc)
            self.ax_stripe.set(xlim=(-0.1, 0.1), ylim=(0, p_len))
            self.ax_stripe.set_xticks([])
            self.ax_prof.plot(p_dist, self.current_z_line, 'k-', lw=1.5)
            self.ax_prof.set(ylabel="Height (Å)", title="Tip Height", xlabel="Dist (Å)")

            for p_idx, (p_label, t_data) in enumerate(processed_partitions):
                ax_l = self.fig.add_subplot(lgs[decomp_base + p_idx * 3])
                cax_l = self.fig.add_subplot(lgs[decomp_base + 1 + p_idx * 3])
                self.line_decomp_axes.extend([ax_l, cax_l])

                v_max = np.max(np.abs(t_data)) if self.show_dcmp_norm else global_vmax
                if v_max == 0:
                    v_max = 1e-15

                if use_diverging:
                    mesh = ax_l.pcolormesh(self.cached_eg, p_dist, t_data, cmap='bwr', shading='auto', vmin=-v_max, vmax=v_max)
                elif self.show_mag and has_mag:
                    # NCL: |m| is non-negative, use sequential colormap
                    mesh = ax_l.pcolormesh(self.cached_eg, p_dist, t_data, cmap='hot', shading='auto', norm=mc.PowerNorm(self.ldos_gamma, vmin=0, vmax=v_max))
                else:
                    mesh = ax_l.pcolormesh(self.cached_eg, p_dist, t_data, cmap='jet', shading='auto', norm=mc.PowerNorm(self.ldos_gamma, vmin=0, vmax=v_max))

                ax_l.set_box_aspect(1)
                ax_l.set_anchor('W')

                if self.plot_level >= 3:
                    ax_l.set_title(p_label.split()[-1], fontsize=10)
                else:
                    ax_l.set_title(f"LDOS: {p_label}", fontsize=10)
                ax_l.set_yticks([])

                if self.show_dcmp_norm or p_idx == num_p - 1:
                    cb = self.fig.colorbar(mesh, cax=cax_l)
                    exp = int(np.floor(np.log10(v_max)))
                    cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos, e=exp: f"{x / (10 ** e):.1f}"))
                    cax_l.set_title(f"1e{exp}", fontsize=10)
                else:
                    cax_l.axis('off')
                cax_l.set_anchor('W')

                for i, r in enumerate(self.marker_ratios):
                    idx = int(r * (self.npts - 1))
                    color = self.m_colors[i % len(self.m_colors)]
                    if p_idx == 0:
                        self.ax_prof.axvline(x=p_dist[idx], color=color, ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}')
                    ax_l.axhline(y=p_dist[idx], color=color, ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}')
                    if p_idx == 0:
                        self.ax_stripe.axhline(y=p_dist[idx], color=color, ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}')

            if self.plot_level >= 3:
                ax_super = self.fig.add_subplot(self.top_gs[3])
                ax_super.axis('off')
                ax_super.set_title(f"LDOS: {self.active_element}", fontsize=12, pad=20)
                self.line_decomp_axes.append(ax_super)

        elif self.mode == 'Map':
            num_p = max(1, len(partitions))
            h_topo = max(1.0, 3.5 - float(num_p))
            self.gs.set_height_ratios([h_topo, float(num_p)])
            self.ax_map_global.set_subplotspec(self.gs[0, 0])
            self.ax_map.set_subplotspec(self.gs[0, 3])
            self.ax_spec.set_subplotspec(self.gs[0, 4])

            if len(self.map_axes) != nepts * num_p or full_refresh:
                if getattr(self, 'cax_list', None):
                    for cax in self.cax_list:
                        if cax in self.fig.axes:
                            cax.remove()
                self.cax_list = []
                for ax in self.map_axes:
                    if ax in self.fig.axes:
                        ax.remove()
                self.map_axes.clear()
                w_ratios = [1.0] * nepts + [0.08]
                sub_gs = gridspec.GridSpecFromSubplotSpec(num_p, nepts + 1, subplot_spec=self.gs[1, :], width_ratios=w_ratios, wspace=0.1, hspace=0.2)
                for r in range(num_p):
                    for c in range(nepts):
                        self.map_axes.append(self.fig.add_subplot(sub_gs[r, c]))
                    if self.show_dcmp_norm:
                        self.cax_list.append(self.fig.add_subplot(sub_gs[r, nepts]))
                if not self.show_dcmp_norm:
                    self.cax_list.append(self.fig.add_subplot(sub_gs[:, nepts]))

            if not hasattr(self, 'map_e_targets') or len(self.map_e_targets) != nepts or full_refresh:
                self.map_e_targets = np.linspace(self.s_emin.val, self.s_emax.val, nepts)
            m_coords_np = np.array(self.marker_coords)

            processed_partitions = []
            global_vmax = 0.0
            if self.normalize and f_ldos_raw is not None:
                total_ldos_map = np.sum(f_ldos_raw, axis=(2, 3))
                s_idx = np.argsort(self.map_e_targets)
                norm_denom_map = np.trapezoid(total_ldos_map[:, s_idx], x=self.map_e_targets[s_idx], axis=1)[:, None] + 1e-15
            else:
                norm_denom_map = 1.0

            for p_label, p_data in partitions:
                t_data = p_data.copy()
                if self.normalize:
                    t_data /= norm_denom_map
                t_data = np.nan_to_num(t_data, nan=0.0, posinf=0.0, neginf=0.0)
                processed_partitions.append((p_label, t_data))
                v_max = np.max(np.abs(t_data))
                if v_max > global_vmax:
                    global_vmax = v_max
            if global_vmax == 0:
                global_vmax = 1e-15

            import matplotlib.ticker as ticker
            for p_idx, (p_label, t_data) in enumerate(processed_partitions):
                v_max = np.max(np.abs(t_data)) if self.show_dcmp_norm else global_vmax
                if v_max == 0:
                    v_max = 1e-15
                for i, target_e in enumerate(self.map_e_targets):
                    ax = self.map_axes[p_idx * nepts + i]
                    ax.clear()
                    slice_data = t_data[:, i]
                    for nx in range(2):
                        for ny in range(2):
                            off = nx * self.lv[0, :2] + ny * self.lv[1, :2]
                            if use_diverging:
                                mesh = ax.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], slice_data, levels=40, cmap='bwr', vmin=-v_max, vmax=v_max)
                            elif self.show_mag and has_mag:
                                mesh = ax.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], slice_data, levels=40, cmap='hot', norm=mc.PowerNorm(self.ldos_gamma, vmin=0, vmax=v_max))
                            else:
                                mesh = ax.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], slice_data, levels=40, cmap='jet', norm=mc.PowerNorm(self.ldos_gamma, vmin=0, vmax=v_max))
                    ax.scatter(m_coords_np[:, 0], m_coords_np[:, 1], color=self.m_colors[:len(m_coords_np)], s=30, edgecolors='white', zorder=5)
                    title_str = f"E = {target_e:.3f} eV" if p_idx == 0 else ""
                    if self.plot_level >= 3:
                        ylabel_str = p_label.split()[-1] if i == 0 else ""
                    else:
                        ylabel_str = p_label if i == 0 else ""
                    ax.set_title(title_str, fontsize=10)
                    if ylabel_str:
                        ax.set_ylabel(ylabel_str, fontsize=10)
                    ax.set_aspect('equal')
                    ax.set_xticks([])
                    ax.set_yticks([])

                if self.show_dcmp_norm:
                    cax = self.cax_list[p_idx]
                    cax.clear()
                    cmap_choice = 'bwr' if use_diverging else ('hot' if (self.show_mag and has_mag) else 'jet')
                    norm_choice = mc.Normalize(vmin=-v_max, vmax=v_max) if use_diverging else mc.Normalize(vmin=0, vmax=v_max)
                    sm = plt.cm.ScalarMappable(cmap=cmap_choice, norm=norm_choice)
                    sm.set_array([])
                    cb = self.fig.colorbar(sm, cax=cax)
                    exp = int(np.floor(np.log10(v_max)))
                    cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos, e=exp: f"{x / (10 ** e):.1f}"))
                    cax.set_title(f"1e{exp}", fontsize=10)

            if not self.show_dcmp_norm and len(processed_partitions) > 0:
                cax = self.cax_list[0]
                cax.clear()
                cmap_choice = 'bwr' if use_diverging else ('hot' if (self.show_mag and has_mag) else 'jet')
                norm_choice = mc.Normalize(vmin=-global_vmax, vmax=global_vmax) if use_diverging else mc.Normalize(vmin=0, vmax=global_vmax)
                sm = plt.cm.ScalarMappable(cmap=cmap_choice, norm=norm_choice)
                sm.set_array([])
                cb = self.fig.colorbar(sm, cax=cax)
                exp = int(np.floor(np.log10(global_vmax)))
                cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos, e=exp: f"{x / (10 ** e):.1f}"))
                cax.set_title(f"1e{exp}", fontsize=10)

            if self.plot_level >= 3:
                ax_super = self.fig.add_subplot(self.gs[1, :])
                ax_super.axis('off')
                ax_super.set_title(f"LDOS: {self.active_element}", fontsize=12, pad=25)
                self.map_axes.append(ax_super)

        self.fig.canvas.draw_idle()

    def _redraw_map_slice(self, i):
        target_e = self.map_e_targets[i]
        f_up = self.cached_ld_up.copy()
        f_dn = self.cached_ld_dn.copy() if self.cached_ld_dn is not None else None
        f_ldos_raw = self._combine_ldos_display(f_up, f_dn, self.show_mag)
        partitions = self._get_partitions(f_ldos_raw)

        mag_signed = self._mag_is_signed()
        has_mag = self._has_mag_channel()
        use_diverging = self.show_mag and has_mag and mag_signed

        m_coords_np = np.array(self.marker_coords)
        nepts = len(self.map_e_targets)

        processed_partitions = []
        global_vmax = 0.0
        if self.normalize and f_ldos_raw is not None:
            total_ldos_map = np.sum(f_ldos_raw, axis=(2, 3))
            s_idx = np.argsort(self.map_e_targets)
            norm_denom_map = np.trapezoid(total_ldos_map[:, s_idx], x=self.map_e_targets[s_idx], axis=1)[:, None] + 1e-15
        else:
            norm_denom_map = 1.0

        for p_label, p_data in partitions:
            t_data = p_data.copy()
            if self.normalize:
                t_data /= norm_denom_map
            t_data = np.nan_to_num(t_data, nan=0.0, posinf=0.0, neginf=0.0)
            processed_partitions.append((p_label, t_data))
            v_max = np.max(np.abs(t_data))
            if v_max > global_vmax:
                global_vmax = v_max
        if global_vmax == 0:
            global_vmax = 1e-15

        for p_idx, (p_label, t_data) in enumerate(processed_partitions):
            ax = self.map_axes[p_idx * nepts + i]
            ax.clear()
            v_max = np.max(np.abs(t_data)) if self.show_dcmp_norm else global_vmax
            if v_max == 0:
                v_max = 1e-15
            slice_data = t_data[:, i]
            for nx in range(2):
                for ny in range(2):
                    off = nx * self.lv[0, :2] + ny * self.lv[1, :2]
                    if use_diverging:
                        ax.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], slice_data, levels=40, cmap='bwr', vmin=-v_max, vmax=v_max)
                    elif self.show_mag and has_mag:
                        ax.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], slice_data, levels=40, cmap='hot', norm=mc.PowerNorm(self.ldos_gamma, vmin=0, vmax=v_max))
                    else:
                        ax.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], slice_data, levels=40, cmap='jet', norm=mc.PowerNorm(self.ldos_gamma, vmin=0, vmax=v_max))
            ax.scatter(m_coords_np[:, 0], m_coords_np[:, 1], color=self.m_colors[:len(m_coords_np)], s=30, edgecolors='white', zorder=5)
            title_str = f"E = {target_e:.3f} eV" if p_idx == 0 else ""
            if self.plot_level >= 3:
                ylabel_str = p_label.split()[-1] if i == 0 else ""
            else:
                ylabel_str = p_label if i == 0 else ""
            ax.set_title(title_str, fontsize=10)
            if ylabel_str:
                ax.set_ylabel(ylabel_str, fontsize=10)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

    def _on_pick(self, event):
        if event.artist == getattr(self, 'ends', None) and self.mode == 'Line':
            self.active_obj = ('end', event.ind[0])
        elif event.artist == getattr(self, 'path_pts', None) and self.mode == 'Path':
            self.active_obj = ('path_node', event.ind[0])
        elif event.artist == getattr(self, 'marks', None):
            self.active_obj = ('mark_map', event.ind[0])
        elif getattr(event, 'mouseevent', None) and event.mouseevent.inaxes == self.ax_spec:
            label = event.artist.get_label()
            if label.startswith('emarker_'):
                if self.plot_level == 0:
                    self.active_obj = ('emarker', int(label.split('_')[1]))
                return
            if self.plot_level == 0:
                if label.startswith('marker_'):
                    self.active_marker_idx = int(label.split('_')[1])
                self.plot_level = 1
            elif self.plot_level == 1:
                self.plot_level = 2
            elif self.plot_level == 2:
                self.active_element = label
                self.plot_level = 3
            elif self.plot_level == 3:
                self.active_atom = next(k for k, v in self.vesta_label_map.items() if v == label)
                self.active_element = self._get_element_by_index_helper(self.active_atom)
                self.plot_level = 4
            self._update_all()
        elif isinstance(event.artist, plt.Line2D) and event.artist.get_label().startswith('marker_'):
            self.active_obj = ('mark_dynamic', int(event.artist.get_label().split('_')[1]))
            self.active_marker_idx = int(event.artist.get_label().split('_')[1])
            self._update_all()

    def _on_press(self, event):
        if event.inaxes == self.ax_spec:
            if self.fig.canvas.manager.toolbar.mode == "" and not event.dblclick:
                hit = any(l.contains(event)[0] for l in self.ax_spec.get_lines() if l.get_picker())
                if not hit:
                    self.plot_level = max(0, self.plot_level - 1)
        if self.fig.canvas.manager.toolbar.mode == "":
            self._update_all()

    def _get_element_by_index_helper(self, a):
        curr = 0
        for idx, count in enumerate(self.atomnums):
            if a <= curr + count:
                return self.atomtypes[idx]
            curr += count
        return 'grey'

    def _on_motion(self, event):
        if self.active_obj is None or event.xdata is None:
            return
        t_obj, idx = self.active_obj
        if self.mode == 'Line':
            p_len = norm(self.p2 - self.p1)
        elif self.mode == 'Path':
            _, _, p_len, _ = self._build_path_xy()
        else:
            p_len = 1.0

        if t_obj == 'end' and self.mode == 'Line':
            if idx == 0:
                self.p1 = np.array([event.xdata, event.ydata])
            else:
                self.p2 = np.array([event.xdata, event.ydata])
        elif t_obj == 'path_node' and self.mode == 'Path':
            if event.inaxes == self.ax_map:
                self.path_nodes[idx] = np.array([event.xdata, event.ydata])
        elif t_obj == 'mark_map' and event.inaxes == self.ax_map:
            if self.mode == 'Line':
                v = self.p2 - self.p1
                v_sq = np.dot(v, v)
                if v_sq > 1e-9:
                    self.marker_ratios[idx] = np.clip(np.dot(np.array([event.xdata, event.ydata]) - self.p1, v) / v_sq, 0, 1)
            elif self.mode == 'Path':
                # Project click onto the polyline: nearest resampled path point.
                p_xy, _, _, _ = self._build_path_xy()
                d2 = (p_xy[:, 0] - event.xdata) ** 2 + (p_xy[:, 1] - event.ydata) ** 2
                self.marker_ratios[idx] = int(np.argmin(d2)) / (self.npts - 1)
            else:
                self.marker_coords[idx] = [event.xdata, event.ydata]
        elif t_obj == 'mark_dynamic' and self.mode in ('Line', 'Path'):
            if hasattr(self, 'ax_prof') and event.inaxes == self.ax_prof:
                if p_len > 1e-9:
                    self.marker_ratios[idx] = np.clip(event.xdata / p_len, 0, 1)
            elif event.inaxes in [ax for ax in [getattr(self, 'ax_ldos', None), getattr(self, 'ax_stripe', None), getattr(self, 'ax_line_topo', None)] if ax is not None]:
                if p_len > 1e-9:
                    self.marker_ratios[idx] = np.clip(event.ydata / p_len, 0, 1)
        elif t_obj == 'emarker' and event.inaxes == self.ax_spec:
            self.map_e_targets[idx] = np.clip(event.xdata, self.s_emin.val, self.s_emax.val)
            for line in self.ax_spec.get_lines():
                if line.get_label() == f'emarker_{idx}':
                    line.set_xdata([self.map_e_targets[idx], self.map_e_targets[idx]])
            self.fig.canvas.draw_idle()
            return
        self._update_all()

    def _on_rel(self, event):
        if self.active_obj is not None and self.active_obj[0] == 'emarker':
            idx = self.active_obj[1]
            target_e = self.map_e_targets[idx]
            e_idx = np.searchsorted(self.energies, target_e)
            if e_idx >= len(self.energies):
                e_idx = len(self.energies) - 1
            grid_z = cp.hstack([self.grid_xy_gpu, cp.array(self.current_z_map)[:, None]])
            t_up, t_dn, _ = self._calculate_ldos_at_points_gpu(
                grid_z, self.energies[e_idx],
                self.energies[min(e_idx + 1, len(self.energies) - 1)] + 1e-6,
                use_energy_decay=self.use_decay_ldos, preserve_orbitals=True,
                global_bias=abs(self.s_emax.val - self.s_emin.val))
            self.cached_ld_up[:, idx:idx + 1] = cp.asnumpy(t_up[:, 0:1])
            if t_dn is not None and self.cached_ld_dn is not None:
                self.cached_ld_dn[:, idx:idx + 1] = cp.asnumpy(t_dn[:, 0:1])
            self._redraw_map_slice(idx)
            self.fig.canvas.draw_idle()
        self.active_obj = None
