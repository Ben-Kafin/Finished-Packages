# -*- coding: utf-8 -*-
"""
Auto-detecting DOSCAR parser.
Handles ISPIN=1 (collinear unpolarized), ISPIN=2 (collinear polarized),
and non-collinear SOC calculations from a single interface.

Author: Benjamin Kafin
"""

import numpy as np
from enum import Enum, auto


class SpinMode(Enum):
    COLLINEAR_UNPOL = auto()
    COLLINEAR_POL = auto()
    NCL_SOC = auto()


# Column-count → SpinMode mapping (unambiguous by VASP format)
#   ISPIN=1:  3 (l-decomposed), 9 (lm-decomposed spd), 16 (lm spdf)
#   ISPIN=2:  6, 18, 32
#   NCL/SOC:  12 (l×4), 36 (lm-spd×4), 64 (lm-spdf×4)
_COL_TO_MODE = {}
for _n in (3, 9, 16):
    _COL_TO_MODE[_n] = SpinMode.COLLINEAR_UNPOL
for _n in (6, 18, 32):
    _COL_TO_MODE[_n] = SpinMode.COLLINEAR_POL
for _n in (12, 36, 64):
    _COL_TO_MODE[_n] = SpinMode.NCL_SOC


class SpinAwareDosParser:
    """
    Decoupled DOSCAR parser optimized for LDOS simulations.
    Auto-detects spin mode from the site-projected column count.

    Attributes
    ----------
    spin_mode : SpinMode
        Detected spin configuration.
    energies : ndarray, shape (nedos,)
        Energy grid relative to Fermi level.
    ef : float
        Fermi energy (eV).
    spin_up_dos : ndarray
        ISPIN=1 / NCL_SOC: total DOS, shape (atoms, nedos, n_orbitals).
        ISPIN=2: spin-up DOS, same shape.
    spin_down_dos : ndarray or None
        ISPIN=2: spin-down DOS.  Others: None.
    mag_dos : ndarray or None
        NCL_SOC only: magnetization magnitude |m| = sqrt(mx²+my²+mz²),
        shape (atoms, nedos, n_orbitals).  Others: None.
    is_polarized : bool
        True for COLLINEAR_POL (backward compat).
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.energies = None
        self.ef = 0.0
        self.spin_mode = SpinMode.COLLINEAR_UNPOL
        self.spin_up_dos = None
        self.spin_down_dos = None
        self.mag_dos = None
        self.is_polarized = False
        self._parse()

    # ------------------------------------------------------------------
    def _parse(self):
        with open(self.filepath, "r") as f:
            atomnum = int(f.readline().split()[0])
            for _ in range(4):
                f.readline()
            header = f.readline().split()
            nedos, self.ef = int(header[2]), float(header[3])

            energies_list = []
            total_dos = []
            site_dos = []

            for i in range(atomnum + 1):
                if i != 0:
                    f.readline()  # skip atom header line
                block = []
                for _ in range(nedos):
                    line = [float(x) for x in f.readline().split()]
                    if i == 0:
                        energies_list.append(line[0])
                        total_dos.append(line[1:])
                    else:
                        block.append(line[1:])
                if i > 0:
                    site_dos.append(block)

        self.energies = np.array(energies_list) - self.ef
        site_dos = np.array(site_dos)  # (atoms, nedos, num_cols)
        num_cols = site_dos.shape[2]

        # --- Auto-detect spin mode from column count ---
        self.spin_mode = _COL_TO_MODE.get(num_cols)
        if self.spin_mode is None:
            raise ValueError(
                f"Unexpected DOSCAR column count {num_cols}. "
                f"Expected one of {sorted(_COL_TO_MODE.keys())}."
            )

        # --- Branch on detected mode ---
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            self.is_polarized = True
            # VASP interleaved: s_up, s_dn, py_up, py_dn, ...
            self.spin_up_dos = site_dos[:, :, 0::2]
            self.spin_down_dos = site_dos[:, :, 1::2]
            self.mag_dos = None

        elif self.spin_mode == SpinMode.NCL_SOC:
            self.is_polarized = False
            # NCL: total, mx, my, mz per orbital (stride-4)
            self.spin_up_dos = site_dos[:, :, 0::4]   # ρ_total per orbital
            dos_mx = site_dos[:, :, 1::4]
            dos_my = site_dos[:, :, 2::4]
            dos_mz = site_dos[:, :, 3::4]
            self.mag_dos = np.sqrt(dos_mx**2 + dos_my**2 + dos_mz**2)
            self.spin_down_dos = None

        else:  # COLLINEAR_UNPOL
            self.is_polarized = False
            self.spin_up_dos = site_dos
            self.spin_down_dos = None
            self.mag_dos = None

    # ------------------------------------------------------------------
    def get_dos_for_simulator(self, spin="up"):
        """
        Returns (num_atoms, nedos, num_orbitals) array.
        For NCL_SOC with spin='mag', returns magnetization magnitude DOS.
        """
        if spin == "mag" and self.spin_mode == SpinMode.NCL_SOC:
            return self.mag_dos
        if spin == "up" or not self.is_polarized:
            return self.spin_up_dos
        return self.spin_down_dos
