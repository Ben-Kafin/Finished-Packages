# -*- coding: utf-8 -*-
"""
File: matcher_vacupot.py (Merged: Vacuum Alignment + Builder + Classifier Analysis)
Supports multiple molecule systems simultaneously.
Allows choosing the Global Zero reference system.
Uses WF_dpl_vcupot for vacuum calculation.
"""
import os
import sys

# When this file is run directly (e.g. %runfile on matcher_vacupot.py) rather
# than imported as part of the 'matcher' package, the package-relative imports
# below fail and the absolute fallbacks are used instead. Those fallbacks pull in
# builder.py, which itself imports the Zheng modules (vaspwfc, aewfc) that live in
# the repository root -- one directory ABOVE this file's 'matcher/' folder. Insert
# that repo root on sys.path so those transitive imports resolve in the standalone
# case. When run packaged (via run_MOPMatcher.py) the root is already importable,
# so this insert is a harmless no-op.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from typing import List, Optional, Tuple, Union
from pymatgen.io.vasp import Locpot
import matplotlib.pyplot as plt

# --- 1. Import all external tools properly ---
# Relative imports when loaded as the 'matcher' package; absolute fallbacks when
# this file is run directly as a standalone script (see the sys.path insert above).
try:
    from .builder import TrueBlochStateBuilder, whitener_uid_from_file
    from .vacupot_plotter import RectAEPAWColorPlotter, PlotConfig
    from .classifier import StateBehaviorClassifier
    from .wf_dpl_vacupot import calculate_work_function
except ImportError:
    from builder import TrueBlochStateBuilder, whitener_uid_from_file
    from vacupot_plotter import RectAEPAWColorPlotter, PlotConfig
    from classifier import StateBehaviorClassifier
    from wf_dpl_vacupot import calculate_work_function


def _build_worker(molecule_dirs, metal_dir, full_dir, analysis_kwargs, log_queue=None):
    """
    Module-level entry point run in a SEPARATE process to build the true Bloch
    state .h5 files. Running the builder in a child process that fully exits
    before the matcher proceeds means the OS reclaims the builder's entire memory
    footprint (large per-band coefficient arrays, beta vectors, the whitener
    eigendecomposition) before the matcher allocates anything. The two phases
    communicate ONLY through the on-disk .h5 files, so nothing large crosses the
    process boundary. Defined at module scope so it is picklable by the 'spawn'
    start method (required on Windows).

    log_queue, when given, is a multiprocessing.Queue the builder writes its
    progress lines to so the PARENT process can print them to the console. Spyder
    routes a spawned child's stdout to the (hidden) kernel window rather than the
    IPython console, so the child cannot print to the visible console directly;
    forwarding the lines to the parent, which prints them, is what makes build
    progress appear live in Spyder. A None sentinel is always sent (in finally) so
    the parent's reader loop terminates even if the build raises.
    """
    try:
        builder = TrueBlochStateBuilder(molecule_dirs, metal_dir, full_dir, **analysis_kwargs)
        if log_queue is not None:
            builder._log_queue = log_queue
        builder.build_all()
    finally:
        if log_queue is not None:
            try:
                log_queue.put(None)  # sentinel: tells the parent reader to stop
            except Exception:
                pass


def _run_build_in_subprocess(molecule_dirs, metal_dir, full_dir, analysis_kwargs):
    """
    Launch _build_worker in a child process and wait for it to finish, so the
    builder's memory is fully released back to the OS before matching begins.
    Uses an explicit 'spawn' context so the child does not inherit the parent's
    address space. If the child exits with an error, raise so the caller does not
    silently proceed without the freshly built files.

    The builder's progress lines are forwarded over a Queue and printed HERE in
    the parent process, so they appear live in the Spyder IPython console (a
    spawned child's own stdout does not). Draining happens while the child runs;
    the loop ends when the worker sends its None sentinel or the child exits.
    """
    import multiprocessing as mp
    import queue as _queue
    ctx = mp.get_context("spawn")
    log_queue = ctx.Queue()
    proc = ctx.Process(
        target=_build_worker,
        args=(molecule_dirs, metal_dir, full_dir, analysis_kwargs, log_queue),
    )
    proc.start()

    # Drain and print builder messages as they arrive. A short timeout lets the
    # loop notice if the child died without sending a sentinel (e.g. hard crash).
    done = False
    while not done:
        try:
            msg = log_queue.get(timeout=0.2)
        except _queue.Empty:
            if not proc.is_alive():
                # Child exited; drain anything still queued, then stop.
                while True:
                    try:
                        msg = log_queue.get_nowait()
                    except _queue.Empty:
                        break
                    if msg is None:
                        break
                    print(msg, flush=True)
                done = True
            continue
        if msg is None:
            done = True
        else:
            print(msg, flush=True)

    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(
            f"[MATCHER] Build subprocess failed (exit code {proc.exitcode}); "
            "the true_blochstates.h5 files were not (re)built."
        )

class RectangularTrueBlochMatcher:
    """
    Analyzes overlaps between pre-computed true Bloch states. 
    Performs vacuum-level alignment for physical energy consistency.
    Supports multiple molecule inputs.
    """
    def __init__(self, molecule_dirs, metal_dir, full_dir, **kwargs):
        self.molecule_dirs = list(molecule_dirs)
        self.metal_dir, self.full_dir = metal_dir, full_dir
        self.mol_labels = [os.path.basename(os.path.normpath(d)) for d in self.molecule_dirs]
        self.analysis_kwargs = kwargs
        
        # Configuration
        self.curv_tol = kwargs.get("curvature_tol", 2.5e-5)
        self.dipole_threshold = kwargs.get("dipole_threshold", 2.5)
        
        # --- Robust Band Window Handling ---
        bw_mols_input = kwargs.get("band_window_molecules", [None])
        if not isinstance(bw_mols_input, list):
            bw_mols_input = [bw_mols_input]
        if len(bw_mols_input) < len(self.molecule_dirs):
            diff = len(self.molecule_dirs) - len(bw_mols_input)
            bw_mols_input.extend([bw_mols_input[-1]] * diff)
            
        self.band_windows = {lbl: bw_mols_input[i] for i, lbl in enumerate(self.mol_labels)}

        # --- Per-directory Fermi-line mode ---
        # For each system, choose whether its red local-Fermi line is drawn at the
        # reported/aligned Fermi energy (False, default) or exactly midway between
        # that system's HOMO and LUMO (True). Only affects the value written to the
        # '# FERMIS:' header (i.e. the drawn line); energy alignment, color
        # assignment, and the matching/classification are unaffected.
        fm_mols_input = kwargs.get("fermi_mid_molecules", False)
        if not isinstance(fm_mols_input, list):
            fm_mols_input = [fm_mols_input] * len(self.molecule_dirs)
        if len(fm_mols_input) < len(self.molecule_dirs):
            fm_mols_input = list(fm_mols_input) + [False] * (len(self.molecule_dirs) - len(fm_mols_input))
        self.fermi_mid = {lbl: bool(fm_mols_input[i]) for i, lbl in enumerate(self.mol_labels)}
        self.fermi_mid["metal"] = bool(kwargs.get("fermi_mid_metal", False))
        self.fermi_mid["full"] = bool(kwargs.get("fermi_mid_full", False))

    @staticmethod
    def get_vacuum_potential(directory: str, curvature_tol=5e-8, dipole_threshold=0.15, min_width=10, reuse_cache=True) -> Tuple[float, bool]:
        """
        Uses WF_dpl_vcupot logic to find vacuum potential. 
        """
        cache_path = os.path.join(directory, "vacuum_potential.h5")
        
        # 1. Check Cache
        if reuse_cache and os.path.isfile(cache_path):
            try:
                with h5py.File(cache_path, "r") as data:
                    return float(data["v_vac"][()]), True
            except Exception:
                pass

        locpot_path = os.path.join(directory, "LOCPOT")
        if not os.path.exists(locpot_path):
            raise FileNotFoundError(f"LOCPOT missing in {directory} for vacuum alignment.")
        
        # 2. Load LOCPOT
        try:
            locpot = Locpot.from_file(locpot_path)
        except Exception as e:
             print(f"  [ERROR] Failed to load LOCPOT at {directory}: {e}")
             return 0.0, False

        # 3. Call External Logic
        # We pass ef=0.0 because we only care about the absolute vacuum potential (v_vac)
        # returned in the tuple, not the work function derived from EF.
        try:
            _, v_vac, _ = calculate_work_function(
                locpot, 
                ef=0.0, 
                curvature_tol=curvature_tol, 
                dipole_threshold=dipole_threshold, 
                min_width=min_width,
                plot=False,     # Disable plotting for batch mode
                verbose=False   # Disable verbose printing
            )
        except Exception as e:
            print(f"  [ERROR] Vacuum calculation failed for {os.path.basename(directory)}: {e}")
            # Fallback to simple max if complex logic fails
            v_vac = np.max(locpot.data['total'])

        # 4. Save and Return
        # Uncompressed scalar (mirrors the previous np.savez / ZIP_STORED write).
        with h5py.File(cache_path, "w") as f:
            f.create_dataset("v_vac", data=np.float64(v_vac))
        return float(v_vac), False

    @staticmethod
    def load_true_bloch(directory, bands=None):
        """
        Load the true Bloch states for one system from true_blochstates.h5.

        Returns a tuple (C, B_ortho) of the two normalized blocks (complex128),
        NOT a single concatenated array: the matcher contracts the two blocks
        separately via the block-sum overlap identity
            <psi_i | psi_j> = C_i . C_j^H  +  B_i . B_j^H
        so the combined [C | B_ortho] array is never materialized.

        Storage holds the raw (pre-normalization) C and B_ortho plus the per-band
        norms; normalization is applied here on read by dividing each band row by
        its norm, which is bit-identical to having stored the normalized arrays.

        bands : optional band selector (a slice or 1-D integer index array). When
        given, only those bands are read from disk via an HDF5 hyperslab read and
        normalized, so the full per-system array need never be resident (band-tile
        streaming). When None (default), all bands are read.

        Returns None (with a warning) if the file is missing or unreadable, so the
        caller rebuilds exactly that system.
        """
        path = os.path.join(directory, "true_blochstates.h5")
        if not os.path.isfile(path): return None
        try:
            with h5py.File(path, "r") as data:
                if bands is None:
                    C = data["C"][...]
                    B_ortho = data["B_ortho"][...]
                    norms = data["norms"][...]
                else:
                    # Hyperslab (partial) read of just the requested bands.
                    C = data["C"][bands, :]
                    B_ortho = data["B_ortho"][bands, :]
                    norms = data["norms"][bands]
            # Normalize on read (norms stored as complex128 with zero imaginary
            # part; dividing is bit-identical to dividing by the real norms).
            C = C / norms[:, None]
            B_ortho = B_ortho / norms[:, None]
            return C, B_ortho
        except Exception as e:
            print(f"[MATCHER][WARN] Corrupt/unreadable {path} "
                  f"({type(e).__name__}: {e}); it will be rebuilt.")
            return None

    @staticmethod
    def _open_bloch_reader(directory):
        """
        Open directory/true_blochstates.h5 for streaming band reads and return
        (h5file, ncoeff, northo, nbands) without loading C/B_ortho into memory.
        The caller reads bands (or band tiles) on demand via _read_bloch_bands and
        must close the returned file. Returns None if the file is missing/unreadable
        (so the caller can trigger a rebuild), mirroring load_true_bloch.
        """
        path = os.path.join(directory, "true_blochstates.h5")
        if not os.path.isfile(path):
            return None
        try:
            f = h5py.File(path, "r")
            ncoeff = int(f["C"].shape[1])
            northo = int(f["B_ortho"].shape[1])
            nbands = int(f["C"].shape[0])
            return f, ncoeff, northo, nbands
        except Exception as e:
            print(f"[MATCHER][WARN] Corrupt/unreadable {path} "
                  f"({type(e).__name__}: {e}); it will be rebuilt.")
            return None

    @staticmethod
    def _read_bloch_bands(h5file, band_idx):
        """
        Read the given band rows (a 1-D integer index array or a slice) from an
        open true_blochstates.h5, divide by the stored per-band norms (the same
        normalization load_true_bloch applies on read, bit-identical), and return
        (C, B_ortho) as complex128 arrays of shape (len(band_idx), ncoeff/northo).
        Only the requested bands are pulled off disk (HDF5 hyperslab read).

        Normalization is done IN PLACE (C /= n) rather than C = C / n so the large
        C array is not transiently duplicated (the out-of-place form holds the raw
        read AND the divided copy simultaneously, doubling peak memory for the
        widest array). The datasets are stored complex128, so the hyperslab read is
        already complex128 and no dtype copy is needed.
        """
        C = h5file["C"][band_idx, :]
        B = h5file["B_ortho"][band_idx, :]
        n = h5file["norms"][band_idx]
        C = np.ascontiguousarray(C, dtype=np.complex128)
        B = np.ascontiguousarray(B, dtype=np.complex128)
        C /= n[:, None]
        B /= n[:, None]
        return C, B

    @staticmethod
    def _plan_overlap_tiling(ncoeff, northo, n_full, comp_bands_list, metal_label=None):
        """
        Decide the memory cascade (see below) and return a plan dict describing how
        to stream the metal/full overlap so the full wavefunction arrays are never
        all resident at once. Sizing uses 2/3 of the TRUE available RAM
        (psutil.virtual_memory().available -- free + reclaimable, NOT total),
        mirroring the builder's _choose_band_tile.

        Inputs:
          ncoeff, northo : per-band widths of C and B_ortho (identical across all
                           systems -- same cell/ENCUT/k-points).
          n_full         : number of full-system bands in the band window (the
                           output axis).
          comp_bands_list: list of (label, n_bands_in_window) for every component,
                           in the order they are stacked (molecules first, then
                           metal), i.e. the combined component band axis.

        Per-band resident cost (complex128): a C row + a B_ortho row =
        16*(ncoeff+northo) bytes. The overlap/output per (component-band, full-band)
        pair is small (16 B complex + 8 B |z|^2); it is bounded by tile sizes and
        added in.

        Cascade (first that fits 2/3 available RAM wins):
          L1 ONE-SHOT  : all component bands (the whole stack) AND all full bands
                         fit -> one GEMM, everything resident, fastest path.
          L2 FULL-TILED: the whole component stack fits resident, plus a multi-band
                         (>=2) tile of the full system -> hold all components, tile
                         the full axis.
          L3 FULL-1BAND: the whole component stack fits resident, plus exactly one
                         full band -> hold all components, full one band at a time.
          L4 COMP-TILED: not even the whole component stack fits -> tile the
                         combined component stack (component stack gets RAM
                         priority), full system one band at a time. Components are
                         re-read once per full band.

        Returns dict: level, full_tile, comp_tile, avail_bytes, budget_bytes,
        per_band_bytes, n_comp_total.
        """
        try:
            import psutil
            avail_bytes = int(psutil.virtual_memory().available)
            avail_src = "psutil available"
        except Exception:
            avail_bytes = 2 * 1024**3
            avail_src = "psutil unavailable -> 2 GiB fallback"
        budget = (2.0 / 3.0) * avail_bytes

        # Per-band resident cost must count EVERY large array simultaneously alive
        # during a tile's read + stack + GEMM, not just one C row, or the chosen
        # tile will not actually fit (this is what previously OOMed). At peak a
        # component-stack tile holds: the vstacked C_stk (16*ncoeff/band) AND the
        # transient C_parts that np.vstack copies from (another 16*ncoeff/band),
        # plus B_stk (16*northo/band). A full tile holds Cf_t (16*ncoeff/band), its
        # .conj() transient inside the GEMM (another 16*ncoeff/band), plus Bf_t
        # (16*northo/band). So budget 16*(2*ncoeff + northo) per band for EACH axis
        # that is resident, which is ~2x the naive 16*(ncoeff+northo) and leaves
        # the divided/normalized arrays (done in place in _read_bloch_bands) and
        # the small GEMM output covered by the remaining 1/3 of RAM.
        per_band = 16 * (2 * int(ncoeff) + int(northo))
        n_comp_total = int(sum(nb for _lbl, nb in comp_bands_list))

        # Per-(comp_band x full_band) overlap/output element: complex S (16 B) +
        # float64 |z|^2 (8 B). Bounded by the product of the two tile sizes.
        def out_bytes(ct, ft):
            return (16 + 8) * int(ct) * int(ft)

        comp_stack_bytes = n_comp_total * per_band
        full_all_bytes = int(n_full) * per_band

        # L1: everything resident at once.
        if comp_stack_bytes + full_all_bytes + out_bytes(n_comp_total, n_full) <= budget:
            return dict(level="L1_one_shot", full_tile=int(n_full),
                        comp_tile=n_comp_total, avail_bytes=avail_bytes,
                        avail_src=avail_src, budget_bytes=budget,
                        per_band_bytes=per_band, n_comp_total=n_comp_total)

        # L2: whole component stack resident + a multi-band full tile.
        if comp_stack_bytes < budget:
            remaining = budget - comp_stack_bytes
            # full_tile from remaining, leaving room for the per-tile output.
            full_tile = int(remaining // (per_band + out_bytes(n_comp_total, 1)))
            full_tile = max(1, min(full_tile, int(n_full)))
            if full_tile >= 2:
                return dict(level="L2_full_tiled", full_tile=full_tile,
                            comp_tile=n_comp_total, avail_bytes=avail_bytes,
                            avail_src=avail_src, budget_bytes=budget,
                            per_band_bytes=per_band, n_comp_total=n_comp_total)
            # L3: whole component stack resident + exactly one full band.
            return dict(level="L3_full_1band", full_tile=1,
                        comp_tile=n_comp_total, avail_bytes=avail_bytes,
                        avail_src=avail_src, budget_bytes=budget,
                        per_band_bytes=per_band, n_comp_total=n_comp_total)

        # L4: component stack does not fit even alone -> CO-TILE both axes. Hold a
        # multi-band component-stack tile AND a multi-band full tile at once, so the
        # expensive component-stack read is amortized over many full bands instead
        # of being re-read for every single full band (the old full_tile=1 path
        # re-read the whole ~stack per full band -> pathological disk I/O). The
        # streaming loop deletes the previous comp tile before reading the next
        # (single-buffered), so the resident peak is EXACTLY, with no padding beyond
        # the 2/3 budget fraction:
        #   comp_tile*pb + full_tile*pb            (the two resident operands)
        #   + full_tile*16*ncoeff                  (Cf_t.conj() transient in the GEMM)
        #   + comp_tile*full_tile*24               (S complex + Z float64 output)
        #   + S_mf_bytes                           (persistent metal/full matrix)
        # where pb = 16*(ncoeff+northo). The budget left for the two operands after
        # reserving S_mf is split EVENLY between the two axes (both cost pb/band),
        # then full_tile takes the remainder accounting for the conj + output terms.
        pb1 = 16 * (int(ncoeff) + int(northo))           # one band, one copy (C+B)
        # n_metal: the metal component's band count. comp_bands_list is in
        # comp_labels order (molecules then metal); the metal entry is identified by
        # metal_label so S_mf_metal's persistent cost is counted exactly.
        n_metal = None
        if metal_label is not None:
            for _lbl, _nb in comp_bands_list:
                if _lbl == metal_label:
                    n_metal = int(_nb)
                    break
        if n_metal is None:                               # fallback: largest comp
            n_metal = int(max(nb for _lbl, nb in comp_bands_list))
        s_mf_bytes = n_metal * int(n_full) * 16

        budget_ops = budget - s_mf_bytes                  # left for the two operands
        if budget_ops < 2 * pb1:                          # cannot even hold 1+1 band
            budget_ops = 2 * pb1                          # degrade; clamp below
        # Even split: half the operand budget to the component axis.
        half = budget_ops / 2.0
        comp_tile = int(half // pb1)
        comp_tile = max(1, min(comp_tile, n_comp_total))
        # full_tile from the remainder, charging pb for Cf/Bf, 16*ncoeff for the
        # conj() transient, and comp_tile*24 for the per-full-band output column.
        remaining = budget_ops - comp_tile * pb1
        full_den = pb1 + 16 * int(ncoeff) + comp_tile * 24
        full_tile = int(remaining // full_den)
        full_tile = max(1, min(full_tile, int(n_full)))
        # Exact predicted peak (incl. S_mf) for the log so it can be verified.
        peak_bytes = (comp_tile * pb1 + full_tile * pb1
                      + full_tile * 16 * int(ncoeff)
                      + comp_tile * full_tile * 24 + s_mf_bytes)
        return dict(level="L4_cotiled", full_tile=full_tile, comp_tile=comp_tile,
                    avail_bytes=avail_bytes, avail_src=avail_src,
                    budget_bytes=budget, per_band_bytes=pb1,
                    n_comp_total=n_comp_total, peak_bytes=peak_bytes,
                    s_mf_bytes=s_mf_bytes)

    def _stream_overlaps_and_write(
        self, hf, ncoeff, northo, bw_f, comp_readers, comp_windows, comp_labels,
        comp_energies, e_f, metal_label, log=print,
    ):
        """
        Stream the block-sum overlaps over full-system band tiles against the
        stacked component bands, writing the per-component |z|^2 grid into the open
        results file `hf` and returning (rows, S_mf_metal):
          rows       : per-full-band "bests" summary records (same semantics as the
                       former in-RAM loop).
          S_mf_metal : (n_metal_window x n_full) complex metal/full overlap matrix
                       for the classifier (metal is one of the stacked components).

        The full wavefunction arrays are never all resident. The combined component
        band axis (molecules first, then metal -- comp_labels order) and the full
        axis are tiled by _plan_overlap_tiling (2/3 of true available RAM). One GEMM
        per (full-tile x component-stack-tile) handles every component in that stack
        tile at once; rows scatter back to per-component grids via the stack index.
        Output values match the all-in-RAM computation up to GEMM summation order.

        comp_readers : dict label -> reader tuple from _open_bloch_reader, plus the
                       special key "__full__" for the full system's reader.
        comp_windows : dict label -> 1-D band-index array, or None for all bands.
        """
        n_full = int(len(bw_f))

        # Combined component stack index. Each entry is (label, local_pos,
        # source_band): local_pos is the row's position in that component's grid
        # (0..nb-1, == output row), source_band is the actual band index to read
        # from that component's .h5. label_nb[label] is the component's band count.
        stack = []
        label_nb = {}
        for label in comp_labels:
            win = comp_windows[label]
            if win is None:
                nb = int(comp_readers[label][3])  # nbands from reader
                bands = range(nb)
            else:
                win = list(win)
                nb = len(win)
                bands = win
            label_nb[label] = nb
            for local_pos, src in enumerate(bands):
                stack.append((label, local_pos, int(src)))
        n_comp_total = len(stack)

        comp_bands_list = [(label, label_nb[label]) for label in comp_labels]
        plan = self._plan_overlap_tiling(ncoeff, northo, n_full, comp_bands_list,
                                         metal_label=metal_label)
        gib = 1024.0 ** 3
        peak_str = (f"; predicted peak={plan['peak_bytes']/gib:.2f} GiB"
                    if plan.get("peak_bytes") is not None else "")
        log(f"[MATCHER]   overlap tiling: level={plan['level']}; "
            f"avail RAM={plan['avail_bytes']/gib:.2f} GiB ({plan['avail_src']}); "
            f"budget(2/3)={plan['budget_bytes']/gib:.2f} GiB; "
            f"per-band={plan['per_band_bytes']/(1024.0**2):.3f} MiB; "
            f"full_tile={plan['full_tile']} of {n_full}; "
            f"comp_tile={plan['comp_tile']} of {n_comp_total} stacked comp bands"
            + peak_str)

        full_tile = int(plan["full_tile"])
        comp_tile = int(plan["comp_tile"])

        # Per-component |z|^2 grid datasets (nb x n_full), float64.
        grid_group = hf.create_group("grid")
        grid_ds = {}
        for label in comp_labels:
            nb = label_nb[label]
            grid_ds[label] = grid_group.create_dataset(
                label, shape=(nb, n_full), dtype="float64",
                chunks=(nb, min(full_tile, n_full)),
                compression="lzf", shuffle=True, fletcher32=True,
            )

        # Metal overlap matrix for the classifier (assembled as tiles complete).
        n_metal = label_nb[metal_label]
        S_mf_metal = np.empty((n_metal, n_full), dtype=np.complex128)

        full_reader = comp_readers["__full__"][0]
        bw_f_arr = np.asarray(bw_f)

        # Levels L1-L3 hold the whole component stack resident -> read each stack
        # tile once and cache it, reusing across all full tiles. L4 (co-tiled)
        # caches nothing (the stack does not fit), so component tiles are re-read
        # per full TILE -- and, crucially, the previous comp tile is deleted before
        # the next is read (single-buffered), so the planner's peak-RAM accounting
        # (which assumes ONE resident comp tile) matches reality and never swaps.
        hold_components = plan["level"] in ("L1_one_shot", "L2_full_tiled", "L3_full_1band")
        comp_cache = {}

        rows = []
        for f0 in range(0, n_full, full_tile):
            f1 = min(f0 + full_tile, n_full)
            full_bands = bw_f_arr[f0:f1]
            Cf_t, Bf_t = self._read_bloch_bands(full_reader, full_bands)  # (ft, ncoeff)

            # |z|^2 columns for this full tile, per component, for the bests below.
            ztile = {label: np.empty((label_nb[label], f1 - f0), dtype=np.float64)
                     for label in comp_labels}

            for c0 in range(0, n_comp_total, comp_tile):
                c1 = min(c0 + comp_tile, n_comp_total)
                if hold_components and (c0, c1) in comp_cache:
                    C_stk, B_stk = comp_cache[(c0, c1)]
                else:
                    C_stk, B_stk = self._read_stack_tile(comp_readers, stack, c0, c1)
                    if hold_components:
                        comp_cache[(c0, c1)] = (C_stk, B_stk)

                # One GEMM: stacked component bands [c0:c1] vs the full tile.
                S = C_stk @ Cf_t.conj().T + B_stk @ Bf_t.conj().T   # (c1-c0, ft)
                Z = (np.abs(S) ** 2).astype(np.float64)

                # Scatter each stack row to its component grid row (local_pos), and
                # capture metal rows for the classifier matrix.
                for srow in range(c0, c1):
                    label, local_pos, _src = stack[srow]
                    ztile[label][local_pos, :] = Z[srow - c0, :]
                    if label == metal_label:
                        S_mf_metal[local_pos, f0:f1] = S[srow - c0, :]

                # When not caching (L4), release this comp tile's large arrays
                # BEFORE the next iteration reads the next tile, so only one comp
                # tile is ever resident -- matching the single-buffered peak the
                # planner sized against. (When caching, they are retained in
                # comp_cache for reuse and must not be deleted.)
                del S, Z
                if not hold_components:
                    del C_stk, B_stk

            # Persist this full tile's columns and compute per-full-band bests.
            for label in comp_labels:
                grid_ds[label][:, f0:f1] = ztile[label]
            for jj, j in enumerate(range(f0, f1)):
                E_full = e_f[0, j]
                bests = {}
                for i, label in enumerate(comp_labels):
                    energies = comp_energies[i]
                    mags = ztile[label][:, jj]
                    i_best, ov_best, w_span = int(np.argmax(mags)), float(mags.max()), float(mags.sum())
                    E_comp = energies[i_best]
                    bests[label] = dict(idx=i_best + 1, E=E_comp, dE=E_full - E_comp,
                                        ov_best=ov_best, w_span=w_span)
                rows.append({"full_idx": j + 1, "E_full": E_full,
                             "residual": max(0., 1. - sum(b['w_span'] for b in bests.values())),
                             **bests})

            # Release this full tile's operands before the next full tile reads its
            # own, so only one full tile is resident at a time (matches the planner's
            # single-buffered peak accounting).
            del Cf_t, Bf_t, ztile

        return rows, S_mf_metal

    def _read_stack_tile(self, comp_readers, stack_rows, c0, c1):
        """
        Read stacked component-band rows [c0:c1] from their source .h5 files and
        return (C_stack, B_stack) of shape (c1-c0, ncoeff/northo), normalized. A
        stack tile may span multiple components (and thus multiple files); rows are
        read per source component in contiguous sub-runs and concatenated in stack
        order. Only the requested bands are read from disk.
        """
        # Group the requested stack rows by source component, preserving order.
        runs = []  # (label, [local_bands...]) contiguous in stack order
        cur_label, cur_bands = None, []
        for srow in range(c0, c1):
            label, _local_pos, lb = stack_rows[srow]
            if label != cur_label:
                if cur_label is not None:
                    runs.append((cur_label, cur_bands))
                cur_label, cur_bands = label, [lb]
            else:
                cur_bands.append(lb)
        if cur_label is not None:
            runs.append((cur_label, cur_bands))

        C_parts, B_parts = [], []
        for label, bands in runs:
            reader = comp_readers[label]
            C, B = self._read_bloch_bands(reader[0], np.asarray(bands))
            C_parts.append(C)
            B_parts.append(B)
        C_stack = np.vstack(C_parts) if len(C_parts) > 1 else C_parts[0]
        B_stack = np.vstack(B_parts) if len(B_parts) > 1 else B_parts[0]
        return C_stack, B_stack

    @staticmethod
    def get_fermi_energy(directory):
        """
        Retrieves the Fermi energy and returns it as 'ef'.
        Searches DOSCAR first, then falls back to OUTCAR.
        """
        doscar_path = os.path.join(directory, "DOSCAR")
        outcar_path = os.path.join(directory, "OUTCAR")
        ef = None
        # 1. Attempt to parse DOSCAR (Line 6, Index 3)
        if os.path.exists(doscar_path):
            try:
                with open(doscar_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 5:
                        # Extracts the float at index 3 of the 6th line
                        ef = float(lines[5].split()[3])
                        return ef
            except (ValueError, IndexError):
                pass
        # 2. Fallback to OUTCAR search
        if os.path.exists(outcar_path):
            try:
                with open(outcar_path, "r") as f:
                    for line in f:
                        # Matches your specific string: "Fermi energy: -2.234..."
                        if "Fermi energy:" in line:
                            # Taking the last element captures the signed numerical value
                            ef = float(line.split()[-1])
                # Returns the final converged value found in the file
                if ef is not None:
                    return ef
            except Exception as e:
                print(f"Error reading OUTCAR: {e}")
        # Default fallback if both files fail
        if ef is None:
            print(f"Warning: Could not find Fermi Energy in {directory}. Using 0.0")
            ef = 0.0
        return ef

    @staticmethod
    def read_gamma_energies_from_eigenval(directory: str):
        path = os.path.join(directory, "EIGENVAL")
        with open(path, "r") as f: lines = f.readlines()
        _, nb = [int(x) for x in lines[5].split()[1:3]]
        idx = 7
        while idx < len(lines) and not lines[idx].strip(): idx += 1
        E = np.zeros((1, nb), float)
        for ib in range(nb):
            line_index = idx + 1 + ib
            if line_index < len(lines):
                E[0, ib] = float(lines[line_index].split()[1])
        return E

    @staticmethod
    def _read_eigenval_occupancies(directory: str, k_index: int = 1):
        """
        Read occupancies from EIGENVAL for the specified k-point block (1-indexed).
        Uses the same k_index the matcher/builder uses (passed in from analysis_kwargs).
        ISPIN-aware column index:
          ISPIN=1 / NCL-SOC (3-col band lines):  occ at split()[2]
          ISPIN=2           (5-col band lines):  occ_up at split()[3], matching
                                                  read_gamma_energies_from_eigenval
                                                  which already takes E_up only.
        Returns shape (1, nb) array of occupancies for the requested k-point.
        """
        path = os.path.join(directory, "EIGENVAL")
        with open(path, "r") as f: lines = f.readlines()
        ispin = int(lines[0].split()[3])
        occ_col = 3 if ispin == 2 else 2
        _, nb = [int(x) for x in lines[5].split()[1:3]]
        idx = 7
        while idx < len(lines) and not lines[idx].strip(): idx += 1
        # Each k-point block = 1 coord line + nb band lines + 1 blank separator = nb + 2 lines.
        # Advance from the first k-point's coord line to the requested k_index (1-indexed).
        coord_line = idx + (k_index - 1) * (nb + 2)
        occ = np.zeros((1, nb), float)
        for ib in range(nb):
            line_index = coord_line + 1 + ib
            if line_index < len(lines):
                occ[0, ib] = float(lines[line_index].split()[occ_col])
        return occ

    @staticmethod
    def _homo_from_occupancies(occupancies, occ_thresh=0.1):
        """
        Determine HOMO band index from EIGENVAL occupancies at a single k-point.
        HOMO = highest band index with occupancy > occ_thresh (1-indexed).
        Threshold 0.1 catches half-filled bands (occ ~ 0.5) and rejects smearing
        noise floor; tolerant of slightly-above-1.0 smearing artifacts.
        Fallback: if no band exceeds the threshold, returns the most-occupied band.
        """
        occ = np.asarray(occupancies).ravel()
        occupied = np.where(occ > occ_thresh)[0]
        if occupied.size == 0:
            return int(np.argmax(occ)) + 1
        return int(occupied.max()) + 1

    @staticmethod
    def _homo_lumo_midpoint(e_raw_row, homo_idx_1based):
        """
        Energy midway between HOMO and LUMO (raw, unshifted), from the full-band
        energies at the matcher's k-point and the full-band HOMO index (1-based).

        Computed from the complete EIGENVAL data before any band-window / plot-range
        slicing, so a LUMO (band HOMO+1) always exists.
        """
        h = int(homo_idx_1based)
        return 0.5 * (float(e_raw_row[h - 1]) + float(e_raw_row[h]))

    def run(self, output_path: Optional[str] = None, reuse_vac_cache: bool = True, zero_reference: str = "metal"):
        """
        zero_reference: "metal", "full", or the name of a molecule (e.g. "NHC_left")
        """
        dir_map = {label: path for label, path in zip(self.mol_labels, self.molecule_dirs)}
        dir_map["metal"], dir_map["full"] = self.metal_dir, self.full_dir

        # Validate reference input
        valid_refs = ["metal", "full"] + self.mol_labels
        if zero_reference not in valid_refs:
             print(f"[ERROR] Invalid zero_reference '{zero_reference}'. Valid options: {valid_refs}. Defaulting to 'metal'.")
             zero_reference = "metal"

        # 1. Builder
        # Build phase runs in a SEPARATE process that fully exits before the
        # match phase, so the OS reclaims the builder's entire footprint before
        # the matcher allocates (lever 1). The two phases share only the on-disk
        # .h5 files, so there is no large inter-process transfer.
        #
        # Decide whether to (re)build from the SAME content-validity check the
        # builder uses (TrueBlochStateBuilder._blochstates_valid): a file is
        # eligible for reuse only if it is present, checksum-clean, shape-
        # consistent, has finite nonzero norms, and carries the current whitener's
        # provenance stamp (w_uid). We validate (rather than probe by loading,
        # which would divide C/B_ortho by norms and warn on a malformed file), so
        # an improperly written or foreign-provenance file triggers a rebuild
        # silently. If the whitener is missing/unreadable, expected_w_uid is None
        # and the builder will rebuild it (and then everything) anyway.
        whitener_path = os.path.join(self.full_dir, "whitener.h5")
        try:
            expected_w_uid = whitener_uid_from_file(whitener_path)
        except Exception:
            expected_w_uid = None
        all_valid = all(
            TrueBlochStateBuilder._blochstates_valid(path, expected_w_uid=expected_w_uid)[0]
            for path in dir_map.values()
        )
        if not self.analysis_kwargs.get("reuse_cached", False) or not all_valid:
            print("[MATCHER] Building missing .h5 files (in a separate process)...")
            _run_build_in_subprocess(self.molecule_dirs, self.metal_dir, self.full_dir, self.analysis_kwargs)

        # Validate each system's true_blochstates.h5 is present and openable
        # WITHOUT loading the (potentially tens-of-GiB) C/B_ortho arrays into RAM.
        # The wavefunctions are streamed in band tiles later (section 5), so the
        # whole-array load that previously OOMed on the full system is gone.
        for name, path in dir_map.items():
            r = self._open_bloch_reader(path)
            if r is None:
                raise FileNotFoundError(
                    f"true_blochstates.h5 for '{name}' could not be opened.")
            r[0].close()

        # 2. Vacuum & Fermi Detection
        print(f"\n[ALIGN] Aligning Vacuum Levels (Ref: Metal) & Setting Global Zero (Ref: {zero_reference})...")
        
        # A. Get Anchors (Metal Vacuum & Reference Fermi)
        v_vac_metal, cached_m = self.get_vacuum_potential(self.metal_dir, self.curv_tol, self.dipole_threshold, reuse_cache=reuse_vac_cache)
        
        # Calculate the Global Shift required to zero the requested reference
        ref_path = dir_map[zero_reference]
        v_vac_ref, _ = self.get_vacuum_potential(ref_path, self.curv_tol, self.dipole_threshold, reuse_cache=reuse_vac_cache)
        e_fermi_ref_raw = self.get_fermi_energy(ref_path)
        
        # Logic for alignment
        global_shift = -(e_fermi_ref_raw + (v_vac_metal - v_vac_ref))
        print(f"  [ALIGN] Global Shift calculated: {global_shift:+.4f} eV (Ensures {zero_reference} E_f -> 0.0 eV)")

        final_fermis = {}
        homo_indices = {}

        # B. Apply to Metal
        e_f_metal_raw = self.get_fermi_energy(self.metal_dir)
        e_raw_metal = self.read_gamma_energies_from_eigenval(self.metal_dir)
        
        shift_metal = global_shift 
        e_m_full = e_raw_metal + shift_metal

        # Occupancy-based HOMO detection (highest band with occ > 0.1 at the matcher's k_index)
        k_index = self.analysis_kwargs.get("k_index", 1)
        occ_metal = self._read_eigenval_occupancies(self.metal_dir, k_index)
        homo_indices["metal"] = self._homo_from_occupancies(occ_metal[0])

        if self.fermi_mid["metal"]:
            final_fermis["metal"] = self._homo_lumo_midpoint(e_raw_metal[0], homo_indices["metal"]) + shift_metal
        else:
            final_fermis["metal"] = e_f_metal_raw + shift_metal
        print(f"  [METAL] V_vac={v_vac_metal:.4f} | Shift={shift_metal:+.4f} eV | Final Fermi={final_fermis['metal']:+.4f} eV")

        # C. Apply to Full System
        v_vac_full, cached_f = self.get_vacuum_potential(self.full_dir, self.curv_tol, self.dipole_threshold, reuse_cache=reuse_vac_cache)
        e_f_full_raw = self.get_fermi_energy(self.full_dir)
        
        shift_full = (v_vac_metal - v_vac_full) + global_shift
        e_raw_full = self.read_gamma_energies_from_eigenval(self.full_dir)
        e_f_full = e_raw_full + shift_full
        if self.fermi_mid["full"]:
            occ_full = self._read_eigenval_occupancies(self.full_dir, k_index)
            homo_full = self._homo_from_occupancies(occ_full[0])
            final_fermis["full"] = self._homo_lumo_midpoint(e_raw_full[0], homo_full) + shift_full
        else:
            final_fermis["full"] = e_f_full_raw + shift_full
        print(f"  [FULL]  V_vac={v_vac_full:.4f} | Shift={shift_full:+.4f} eV | Final Fermi={final_fermis['full']:+.4f} eV")

        # D. Apply to Molecules
        e_ms_full = []
        
        for i, md in enumerate(self.molecule_dirs):
            label = self.mol_labels[i]
            v_vac_mol, cached_mol = self.get_vacuum_potential(md, self.curv_tol, self.dipole_threshold, reuse_cache=reuse_vac_cache)

            e_f_mol_raw = self.get_fermi_energy(md)
            e_raw_mol = self.read_gamma_energies_from_eigenval(md)
            
            # Occupancy-based HOMO detection (highest band with occ > 0.1 at the matcher's k_index)
            occ_mol = self._read_eigenval_occupancies(md, k_index)
            homo_indices[label] = self._homo_from_occupancies(occ_mol[0])
            
            shift_mol = (v_vac_metal - v_vac_mol) + global_shift
            e_ms_full.append(e_raw_mol + shift_mol)
            
            if self.fermi_mid[label]:
                final_fermis[label] = self._homo_lumo_midpoint(e_raw_mol[0], homo_indices[label]) + shift_mol
            else:
                final_fermis[label] = e_f_mol_raw + shift_mol
            print(f"  [{label}] V_vac={v_vac_mol:.4f} | Shift={shift_mol:+.4f} eV | Final Fermi={final_fermis[label]:+.4f} eV")
        
        print("[ALIGN] Alignment complete.\n")

        # 3. Band windows (index arrays only; no wavefunction data needed here).
        # The window selects bands below the vacuum level (full/metal) or the
        # caller-supplied molecule windows. Energies are sliced now; the matching
        # wavefunction rows are read on demand in section 5.
        bw_f = np.where(e_raw_full[0] < v_vac_full)[0]
        e_f = e_f_full[:, bw_f]

        bw_m = np.where(e_raw_metal[0] < v_vac_metal)[0]
        e_m = e_m_full[:, bw_m]

        e_ms = []
        comp_windows = {}
        for i, lbl in enumerate(self.mol_labels):
            bw_mol = self.band_windows.get(lbl)
            if bw_mol:
                # Normalize a slice/None molecule window to an explicit index array.
                nb_mol = e_ms_full[i].shape[1]
                idx = np.arange(nb_mol)[bw_mol]
                comp_windows[lbl] = idx
                e_ms.append(e_ms_full[i][:, bw_mol])
            else:
                comp_windows[lbl] = None
                e_ms.append(e_ms_full[i])
        comp_windows["metal"] = np.asarray(bw_m)

        comp_labels = self.mol_labels + ["metal"]
        comp_energies = [e_ms[i][0] for i in range(len(self.mol_labels))] + [e_m[0]]
        n_full = int(len(bw_f))

        results_path = os.path.join(self.full_dir, "matcher_results.h5")
        os.makedirs(self.full_dir, exist_ok=True)

        # Open streaming readers for the full system and every component (no
        # wavefunction data loaded; bands are read in tiles inside the stream).
        comp_readers = {}
        full_r = self._open_bloch_reader(self.full_dir)
        comp_readers["__full__"] = full_r
        ncoeff, northo = int(full_r[1]), int(full_r[2])
        for i, lbl in enumerate(self.mol_labels):
            comp_readers[lbl] = self._open_bloch_reader(self.molecule_dirs[i])
        comp_readers["metal"] = self._open_bloch_reader(self.metal_dir)

        classifier = StateBehaviorClassifier()
        rows = []
        try:
            with h5py.File(results_path, "w") as hf:
                # 5. Overlaps & Write (streamed). One stacked GEMM per
                # (full-tile x component-stack-tile); the per-component |z|^2 grid
                # is written to matcher_results.h5 as tiles complete, and the
                # metal/full overlap matrix is returned for the classifier. Tiling
                # is sized by _plan_overlap_tiling (2/3 of true available RAM) so
                # the whole wavefunction arrays are never resident; when everything
                # fits, it collapses to a single all-in-RAM GEMM. Output values are
                # bit-identical to the in-RAM computation up to GEMM summation order.
                rows, S_mf_metal = self._stream_overlaps_and_write(
                    hf, ncoeff, northo, bw_f, comp_readers, comp_windows,
                    comp_labels, comp_energies, e_f, metal_label="metal",
                )

                # 4. Classifier (uses the streamed metal/full overlap matrix).
                # Block-sum overlap identity: <psi_m | psi_f> = Cm.Cf^H + Bm.Bf^H,
                # here assembled across tiles into S_mf_metal (n_metal x n_full).
                E_metal_min = np.min(e_m[0])
                degenerate_indices = np.where(e_m[0] < E_metal_min + 0.05)[0]
                group_recs = []
                for metal_idx in degenerate_indices:
                    for full_idx in range(e_f.shape[1]):
                        dE = e_f[0, full_idx] - e_m[0, metal_idx]
                        ov = np.abs(S_mf_metal[metal_idx, full_idx])**2
                        if ov > 1e-6:
                            group_recs.append({'dE': dE, 'ov': ov})
                if group_recs:
                    classification_info = classifier.classify_state(group_recs)
                    print(f"[CLASSIFIER] Metal State Variance: {classification_info['variance']:.4f} eV^2")

                # --- Full results record (metadata + alignment + classifier). ---
                v_aligned = v_vac_metal + global_shift
                self._write_results_metadata(
                    hf, comp_labels, comp_energies, e_f, e_m,
                    v_vac_metal, v_vac_full, v_aligned, global_shift,
                    final_fermis, homo_indices, degenerate_indices,
                    e_raw_full, e_raw_metal,
                    classifier, group_recs, n_full=n_full,
                )
        finally:
            for r in comp_readers.values():
                try:
                    r[0].close()
                except Exception:
                    pass


        main_out = output_path or os.path.join(self.full_dir, "band_matches_rectangular.txt")
        all_out = os.path.join(self.full_dir, "band_matches_rectangular_all.txt")
        os.makedirs(os.path.dirname(main_out), exist_ok=True)

        with open(main_out, "w") as f:
            homo_str = "; ".join([f"{k}={v}" for k, v in homo_indices.items()])
            fermi_str = "; ".join([f"{k}={v:.5f}" for k, v in final_fermis.items()])

            # Single aligned vacuum potential value
            v_aligned = v_vac_metal + global_shift
            vacuum_str = f"aligned={v_aligned:.5f}"

            f.write(f"# HOMOS: {homo_str}\n")
            f.write(f"# FERMIS: {fermi_str}\n")
            f.write(f"# VACUUMS: {vacuum_str}\n") # New header line
            f.write("# full_idx E_full " + " ".join([f"| {lbl}_idx {lbl}_E {lbl}_dE {lbl}_ov_best {lbl}_w_span" for lbl in comp_labels]) + " | residual\n")

            for r in sorted(rows, key=lambda x: x['full_idx']):
                f.write(f"{r['full_idx']:<8d} {r['E_full']:.4f} " + " ".join([f"| {b['idx']:<7d} {b['E']:.4f} {b['dE']:.4f} {b['ov_best']:.5f} {b['w_span']:.5f}" for b in [r[lbl] for lbl in comp_labels]]) + f" | {r['residual']:.5f}\n")

        # _all.txt is written from the stored grid (not a Python tuple list), in
        # the identical current sort order: full_idx ascending, then component
        # label (Python string order), then comp_idx ascending. Byte-for-byte
        # identical to the previous ov_all_lines output.
        with h5py.File(results_path, "r") as hf:
            with open(all_out, "w") as f:
                f.write("# component full_idx comp_idx E_comp dE_comp ov w_span_comp\n")
                label_e = {label: comp_energies[i] for i, label in enumerate(comp_labels)}
                for j in range(1, n_full + 1):
                    E_full = e_f[0, j - 1]
                    for label in sorted(comp_labels):
                        Z_col = hf["grid"][label][:, j - 1]
                        w_span = Z_col.sum()
                        energies = label_e[label]
                        for i_pair in range(Z_col.shape[0]):
                            E_comp = energies[i_pair]
                            f.write(f"{label:<15s} {j:<8d} {i_pair+1:<8d} {E_comp:.4f} {E_full - E_comp:.4f} {Z_col[i_pair]:.5f} {w_span:.5f}\n")

        print(f"[MATCHER] Wrote vacuum-aligned analysis to '{main_out}'")
        print(f"[MATCHER] Wrote complete run record to '{results_path}'")

    def _write_results_metadata(self, hf, comp_labels, comp_energies, e_f, e_m,
                                v_vac_metal, v_vac_full, v_aligned, global_shift,
                                final_fermis, homo_indices, degenerate_indices,
                                e_raw_full, e_raw_metal,
                                classifier, group_recs, n_full):
        """
        Write the complete run record into the already-open matcher_results.h5
        file handle: the energy axes that (with the stored grid) reconstruct the
        _all output, all alignment/energy metadata, the band windows, the
        per-system shifts and Fermi values, the degenerate-index set, the run
        kwargs, and the classifier's metal-variance output. The |z|^2 overlap grid
        itself is written tiled by the caller (group 'grid'). Large arrays use
        lzf + shuffle; scalars are plain datasets.
        """
        import json

        def _arr(group, name, data, complex_ok=False):
            a = np.asarray(data)
            if a.size == 0:
                group.create_dataset(name, data=a)
                return
            group.create_dataset(name, data=a, chunks=True,
                                 compression="lzf", shuffle=True, fletcher32=True)

        meta = hf.create_group("meta")

        # Component energy axes (per component, on the aligned/global scale) and
        # the full-system energy axis: with the grid these reconstruct _all.
        en = meta.create_group("comp_energies")
        for i, label in enumerate(comp_labels):
            _arr(en, label, comp_energies[i])
        _arr(meta, "E_full", e_f[0])
        _arr(meta, "E_metal", e_m[0])

        # Alignment / vacuum / shift metadata.
        meta.create_dataset("v_vac_metal", data=np.float64(v_vac_metal))
        meta.create_dataset("v_vac_full", data=np.float64(v_vac_full))
        meta.create_dataset("v_aligned", data=np.float64(v_aligned))
        meta.create_dataset("global_shift", data=np.float64(global_shift))

        # Per-system vacuum potentials and shifts (computed the same way run()
        # computed them; recorded for reproducibility).
        vac = meta.create_group("v_vac_per_dir")
        shifts = meta.create_group("shift_per_system")
        dir_map = {label: path for label, path in zip(self.mol_labels, self.molecule_dirs)}
        dir_map["metal"], dir_map["full"] = self.metal_dir, self.full_dir
        for name, path in dir_map.items():
            vv, _ = self.get_vacuum_potential(path, self.curv_tol, self.dipole_threshold,
                                              reuse_cache=True)
            vac.create_dataset(name, data=np.float64(vv))
            shifts.create_dataset(name, data=np.float64((v_vac_metal - vv) + global_shift))

        # Final (aligned) and raw Fermi values; HOMO indices; fermi_mid flags.
        ff = meta.create_group("final_fermis")
        for k, v in final_fermis.items():
            ff.create_dataset(k, data=np.float64(v))
        hi = meta.create_group("homo_indices")
        for k, v in homo_indices.items():
            hi.create_dataset(k, data=np.int64(v))
        fm = meta.create_group("fermi_mid")
        for k, v in self.fermi_mid.items():
            fm.create_dataset(k, data=np.int64(1 if v else 0))

        # Raw gamma-point energies (full and metal) before alignment.
        _arr(meta, "e_raw_full", e_raw_full[0])
        _arr(meta, "e_raw_metal", e_raw_metal[0])

        # Degenerate metal indices used for the metal-variance classification.
        meta.create_dataset("degenerate_indices",
                            data=np.asarray(degenerate_indices, dtype=np.int64))

        # Band windows per molecule (as JSON; slice objects are not HDF5-native).
        def _bw_repr(bw):
            if bw is None:
                return None
            if isinstance(bw, slice):
                return {"type": "slice", "start": bw.start, "stop": bw.stop, "step": bw.step}
            return {"type": "index", "values": list(np.asarray(bw).tolist())}
        bw_map = {lbl: _bw_repr(self.band_windows.get(lbl)) for lbl in self.mol_labels}
        meta.create_dataset("band_windows_json", data=json.dumps(bw_map))

        # Component label order and run kwargs.
        meta.create_dataset("comp_labels_json", data=json.dumps(list(comp_labels)))
        meta.create_dataset("run_kwargs_json", data=json.dumps(self.analysis_kwargs, default=str))

        # Classifier: metal-state variance (and the per-group dE/ov it was built
        # from) -- the matcher's own classifier output.
        cls = hf.create_group("classifier")
        if group_recs:
            info = classifier.classify_state(group_recs)
            cm = cls.create_group("metal_variance")
            for key, val in info.items():
                cm.create_dataset(key, data=(val if isinstance(val, str) else np.float64(val)))
            _arr(cls, "metal_group_dE", np.array([r["dE"] for r in group_recs], dtype=float))
            _arr(cls, "metal_group_ov", np.array([r["ov"] for r in group_recs], dtype=float))


def run_match(molecule_dirs, metal_dir, full_dir, **kwargs) -> List[str]:
    matcher = RectangularTrueBlochMatcher(molecule_dirs, metal_dir, full_dir, **kwargs)
    reuse_vac_cache = kwargs.get("reuse_vac_cache", True)
    
    # Pass 'zero_reference' from kwargs, defaulting to 'metal'
    zero_reference = kwargs.get("zero_reference", "metal")
    
    matcher.run(kwargs.get("output_path"), reuse_vac_cache=reuse_vac_cache, zero_reference=zero_reference)
    main_file = kwargs.get("output_path") or os.path.join(full_dir, "band_matches_rectangular.txt")
    all_file = os.path.join(full_dir, "band_matches_rectangular_all.txt")
    return [main_file, all_file]

if __name__ == "__main__":
    run_kwargs = {
        "k_index": 1, 
        "tol_map": 1e-3, 
        "check_species": True,
        "band_window_molecules": [slice(0, 82)],
        "reuse_cached": True,       
        "reuse_vac_cache": True,    
        "curvature_tol": 5e-8, 
        "dipole_threshold": 0.15,
        
        # === CHOOSE YOUR ZERO HERE ===
        "zero_reference": "full" # Options: "metal", "full", "NHC_left" (or your molecule folder name)
    }
    
    match_files = run_match(
        #NHCAu_complex
        #molecule_dirs=[r'C:/Users/Benjamin Kafin/Documents/VASP/lone/fcc/kpoints551/NHC_c/kpoints551/kp552'],
        #metal_dir=r'C:/Users/Benjamin Kafin/Documents/VASP/lone/fcc/kpoints551/lone_adatom/552',
        #full_dir=r'C:/Users/Benjamin Kafin/Documents/VASP/lone/fcc/kpoints551/NHCAu_complex/dipole_correction/kp552',
        
        #NHCAu_fcc
        #molecule_dirs=[r'C:/Users/Benjamin Kafin/Documents/VASP/lone/fcc/kpoints551/NHC/dipole_correction/kpoints551'],
        #metal_dir=r'C:/Users/Benjamin Kafin/Documents/VASP/lone/fcc/kpoints551/adatom_surface/dpl_correction_kp551',
        #full_dir=r'C:/Users/Benjamin Kafin/Documents/VASP/lone/fcc/kpoints551/full_dpl_noEfield/kp551/no_dpl_matched_with_dpl_components',
        
        #NHC2Au_complex
        #molecule_dirs=[r'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au_smaller/ispin1/C_left', r'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au_smaller/ispin1/C_right'],
        #metal_dir=r'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au_smaller/ispin1/lone_adatom',
        #full_dir=r'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au_smaller/ispin1/NHC2Au_complex',
        
        #NHC2Au_fcc
        #molecule_dirs=[r'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au_smaller/ispin1/NHC_left', r'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au_smaller/ispin1/NHC_right'],
       # metal_dir=r'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au_smaller/ispin1/adatom_surface',
        #full_dir=r'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au_smaller/ispin1',
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

        
        **run_kwargs
    )
    main_match_file = match_files[0]
    print(f"[MAIN] Match process complete. Main output: '{main_match_file}'")
    
    print("\n--- Running Plotter ---")
    try:
        cfg = PlotConfig(
            cmap_name_simple="managua_r", 
            cmap_name_metal="vanimo_r",
            energy_range=(-11, 6), 
            shared_molecule_color=True,
            min_total_mol_wspan=0.01,
            
            # NEW COLORING PARAMETERS
            power_simple_neg=0.25,
            power_simple_pos=0.75,
            power_metal_neg=0.075,
            power_metal_pos=0.075,
            power_residual=0.5,
            
            # COLORING MODE
            pick_primary=False, 
            
            # FERMI VISUALS
            show_local_fermi=True,
            
            # NEW: VACUUM VISUALS
            show_vacuum_line=True,      # Enable the black line
            vacuum_line_color="black",
            vacuum_line_width=2,      # Thick line as requested
            vacuum_line_style="-"
        )
        
        plotter = RectAEPAWColorPlotter(cfg)
        fig, axes = plotter.plot(main_match_file, bonding=True) 
        plt.show()
    except Exception as e:
        print(f"[WARN] Plotter call failed: {e}")