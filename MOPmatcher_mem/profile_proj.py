# -*- coding: utf-8 -*-
"""
profile_proj.py — measure where the build spends time and capture ground-truth
betas, BEFORE any rewrite of the projection kernel (nonlq.proj / get_beta_njk).

It does two independent things, neither of which modifies any code and neither of
which needs the full system or a finished build:

  (1) PROFILE: load ONE system (metadata only -- lazy WAVECAR handle, no AE grid),
      then run the current per-band projection over the first N_PROFILE bands under
      cProfile. Prints the functions ranked by cumulative time, so you can see what
      fraction is actually in nonlq.proj vs. everything else. This is fast: a
      molecule directory and ~20 bands take seconds-to-minutes.

  (2) REFERENCE: compute betas for the first N_REF bands with the CURRENT code and
      save them (complex128) to a .npz. After the batched rewrite, run it again in
      "compare" mode to verify the new betas match this ground truth to tolerance.
      A subtle indexing/conjugation error in the rewrite shows up here as a
      mismatch instead of as silently-wrong wavefunctions.

Usage
-----
Edit SYSTEM_DIR / FULL_DIR below to a SMALL system (a molecule directory is ideal)
and run (green play button in Spyder), or:
    python profile_proj.py
Then, after the rewrite, set MODE = "compare" and run again to validate.
"""

import os
import sys
import time
import cProfile
import pstats
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from matcher.builder import TrueBlochStateBuilder


def _load_one_system(system_dir, full_dir, k_index=1, lsorbit=None):
    """
    Construct the builder and load just the target system's metadata (ps + ae,
    no coefficients), exactly as build_all does, so ae.get_beta_njk / _q_proj.proj
    is the real code path. full_dir only sets SOC auto-detection; no build occurs.
    """
    kw = {"k_index": k_index, "reuse_cached": True, "reuse_W": True}
    if lsorbit is not None:
        kw["lsorbit"] = lsorbit
    b = TrueBlochStateBuilder([system_dir], system_dir, full_dir, **kw)
    sysd = b.load_system(system_dir, "profile_target", load_coeffs=False)
    return b, sysd


def _compute_betas(sysd, k_index, nbands):
    """
    Run the CURRENT projection (ae.get_beta_njk -> nonlq.proj) for the first
    `nbands` bands and return an (nbands, nproj) complex128 array. This is exactly
    what _form_B_for_rows does per band, just collected here for timing/reference.
    """
    rows = []
    for ib in range(1, nbands + 1):
        cg = np.asarray(
            sysd.ps.readBandCoeff(ispin=1, ikpt=k_index, iband=ib, norm=False),
            dtype=np.complex128,
        )
        rows.append(np.asarray(sysd.ae.get_beta_njk(cg), dtype=np.complex128))
    return np.vstack(rows)


def _read_C_block(sysd, k_index, nbands):
    """Read the first `nbands` bands' raw coefficients as one (nbands, ncoeff) array."""
    rows = [
        np.asarray(
            sysd.ps.readBandCoeff(ispin=1, ikpt=k_index, iband=ib, norm=False),
            dtype=np.complex128,
        )
        for ib in range(1, nbands + 1)
    ]
    return np.vstack(rows)


def _compute_betas_batched(sysd, k_index, nbands):
    """
    Run the BATCHED projection (proj_batched.get_beta_njk_batched) on a block of
    `nbands` bands at once and return (nbands, nproj) complex128. Same physics as
    _compute_betas, computed as GEMMs instead of a per-band/per-atom loop.
    """
    from proj_batched import get_beta_njk_batched
    C_block = _read_C_block(sysd, k_index, nbands)
    return np.asarray(get_beta_njk_batched(sysd.ae, C_block), dtype=np.complex128)


def run_compare_methods(system_dir, full_dir, k_index=1, n=20, lsorbit=None):
    """
    Head-to-head: run BOTH the current per-band kernel and the batched kernel on
    the same `n` bands, verify they agree (max abs/rel error, bit-identical?), and
    time each. Neither replaces anything; this just proves equivalence + speedup.
    """
    print(f"[CMP] system: {system_dir}")
    b, sysd = _load_one_system(system_dir, full_dir, k_index, lsorbit)
    nb_total = sysd.ps._nbands
    n = min(n, nb_total)
    print(f"[CMP] total bands={nb_total}; comparing first {n} bands.")

    # --- current (per-band loop) ---
    t0 = time.perf_counter()
    beta_cur = _compute_betas(sysd, k_index, n)
    t_cur = time.perf_counter() - t0

    # --- batched (GEMM) ---
    t0 = time.perf_counter()
    beta_bat = _compute_betas_batched(sysd, k_index, n)
    t_bat = time.perf_counter() - t0

    print("")
    print(f"[CMP] current : {t_cur:9.3f} s  ({t_cur / n * 1000:8.1f} ms/band)")
    print(f"[CMP] batched : {t_bat:9.3f} s  ({t_bat / n * 1000:8.1f} ms/band)")
    if t_bat > 0:
        print(f"[CMP] speedup : {t_cur / t_bat:8.1f}x")
    print("")

    # --- correctness ---
    if beta_cur.shape != beta_bat.shape:
        print(f"[CMP] *** SHAPE MISMATCH: current={beta_cur.shape} batched={beta_bat.shape} ***")
        return
    diff = np.abs(beta_cur - beta_bat)
    denom = np.maximum(np.abs(beta_cur), 1e-30)
    print(f"[CMP] shape           : {beta_cur.shape}")
    print(f"[CMP] bit-identical   : {bool(np.array_equal(beta_cur, beta_bat))}")
    print(f"[CMP] max |abs error| : {float(diff.max()):.3e}")
    print(f"[CMP] max |rel error| : {float((diff / denom).max()):.3e}")
    print(f"[CMP] allclose(rtol=1e-9, atol=1e-12): "
          f"{bool(np.allclose(beta_cur, beta_bat, rtol=1e-9, atol=1e-12))}")


def run_profile(system_dir, full_dir, k_index=1, n_profile=20, lsorbit=None):
    print(f"[PROFILE] system: {system_dir}")
    b, sysd = _load_one_system(system_dir, full_dir, k_index, lsorbit)
    nb_total = sysd.ps._nbands
    n = min(n_profile, nb_total)
    print(f"[PROFILE] total bands={nb_total}; profiling first {n} bands.")
    print(f"[PROFILE] nplw={sysd.ps._q_proj.nplw if hasattr(sysd.ps,'_q_proj') else 'n/a'} "
          f"(reciprocal projector grid)")

    # Wall-clock for a plain sense of per-band cost.
    t0 = time.perf_counter()
    _compute_betas(sysd, k_index, n)
    dt = time.perf_counter() - t0
    print(f"[PROFILE] wall time for {n} bands: {dt:.3f} s  ({dt / n * 1000:.1f} ms/band)")
    print(f"[PROFILE] extrapolated for {nb_total} bands: {dt / n * nb_total:.1f} s")
    print("")

    # cProfile to see the ranked breakdown (where the time actually goes).
    pr = cProfile.Profile()
    pr.enable()
    _compute_betas(sysd, k_index, n)
    pr.disable()
    print("[PROFILE] Top functions by cumulative time:")
    st = pstats.Stats(pr).sort_stats("cumulative")
    st.print_stats(15)


def capture_reference(system_dir, full_dir, out_path, k_index=1, n_ref=8, lsorbit=None):
    b, sysd = _load_one_system(system_dir, full_dir, k_index, lsorbit)
    n = min(n_ref, sysd.ps._nbands)
    betas = _compute_betas(sysd, k_index, n)
    np.savez(out_path, betas=betas, system_dir=system_dir, k_index=k_index, n_ref=n)
    print(f"[REF] saved reference betas: shape={betas.shape}, dtype={betas.dtype} -> {out_path}")


def compare_reference(system_dir, full_dir, ref_path, k_index=1, lsorbit=None):
    """
    Recompute betas with whatever the current code is and compare to the saved
    reference. After the rewrite, this validates the new kernel against ground
    truth. Reports max absolute and relative error.
    """
    ref = np.load(ref_path, allow_pickle=True)
    ref_betas = ref["betas"]
    n = int(ref["n_ref"])
    b, sysd = _load_one_system(system_dir, full_dir, k_index, lsorbit)
    new_betas = _compute_betas(sysd, k_index, n)
    if new_betas.shape != ref_betas.shape:
        print(f"[COMPARE] SHAPE MISMATCH: new={new_betas.shape} vs ref={ref_betas.shape}")
        return
    diff = np.abs(new_betas - ref_betas)
    denom = np.abs(ref_betas)
    max_abs = float(diff.max())
    max_rel = float((diff / np.maximum(denom, 1e-30)).max())
    exact = bool(np.array_equal(new_betas, ref_betas))
    print(f"[COMPARE] bands={n}, shape={new_betas.shape}")
    print(f"[COMPARE] bit-identical: {exact}")
    print(f"[COMPARE] max |abs error| = {max_abs:.3e}")
    print(f"[COMPARE] max |rel error| = {max_rel:.3e}")


if __name__ == "__main__":
    # ---- EDIT THESE ----
    # A SMALL system is ideal for fast profiling (a molecule directory).
    SYSTEM_DIR = r"D:/Docs/VASP/NHC/iPr/SAM/NHC2Au/SOC/NHC_c1m1"
    FULL_DIR   = r"D:/Docs/VASP/NHC/iPr/SAM/NHC2Au/SOC"   # only for SOC auto-detect
    K_INDEX    = 1
    N_PROFILE  = 20      # bands to profile (more = steadier timing)
    N_REF      = 8       # bands to store as the validation reference
    REF_PATH   = os.path.join(_THIS_DIR, "proj_reference_betas.npz")

    MODE = "compare_methods"   # "profile" | "reference" | "compare" | "compare_methods"
    # --------------------

    if MODE == "profile":
        run_profile(SYSTEM_DIR, FULL_DIR, k_index=K_INDEX, n_profile=N_PROFILE)
        # also drop a reference now so it's ready for post-rewrite validation
        capture_reference(SYSTEM_DIR, FULL_DIR, REF_PATH, k_index=K_INDEX, n_ref=N_REF)
    elif MODE == "reference":
        capture_reference(SYSTEM_DIR, FULL_DIR, REF_PATH, k_index=K_INDEX, n_ref=N_REF)
    elif MODE == "compare":
        compare_reference(SYSTEM_DIR, FULL_DIR, REF_PATH, k_index=K_INDEX)
    elif MODE == "compare_methods":
        run_compare_methods(SYSTEM_DIR, FULL_DIR, k_index=K_INDEX, n=N_PROFILE)
    else:
        raise ValueError(f"unknown MODE: {MODE}")
