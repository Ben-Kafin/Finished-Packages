# -*- coding: utf-8 -*-
"""
mark_bands_done.py — one-time patch that adds the 'bands_done' completion marker
to existing MOPMatcher true_blochstates.h5 files that were written before the
resumable-build feature existed.

Background
----------
The builder now writes each true_blochstates.h5 incrementally and records, in a
scalar dataset 'bands_done', how many contiguous bands (counted from band 0) have
been fully written. A complete file has bands_done == nbands; a crashed/partial
file has bands_done < nbands and the builder RESUMES it from that band on the next
run. _blochstates_valid now REQUIRES bands_done == nbands to treat a file as a
usable result, so files written by the older code (which have no bands_done) must
be patched once, or they would all be considered incomplete and rebuilt.

What it does (no building, metadata only)
-----------------------------------------
For each true_blochstates.h5 path you give it:
  1. Opens the file and reads 'norms'.
  2. Determines how many leading bands are actually written. The old writer filled
     bands in ascending order in contiguous tile blocks and left the remaining
     rows as zeros, and a real Bloch band's norm is never zero, so the count of
     completed bands is the index of the FIRST zero (or non-finite) norm. If no
     such row exists, every band is written and the count is nbands (a complete
     file).
  3. Writes that count into a 'bands_done' scalar dataset.

So a complete component file gets bands_done == nbands (and will be reused), while
a half-written full-system file gets bands_done == (#written bands) and will be
resumed from there by the builder.

Provenance gate
---------------
If a whitener path is given, a file is patched only when its 'w_uid' attribute
matches the whitener's w_uid (so a file from a different run is not silently
adopted). Pass whitener_path=None to skip this check.

Usage
-----
Edit the paths below and run (green play button in Spyder), or call
mark_files(paths, whitener_path=...) from your own script.
"""

import os
import sys
import h5py
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
# Reuse the canonical whitener-id helper so the provenance check matches the
# builder/stamper exactly.
from matcher.builder import whitener_uid_from_file


def _count_written_bands(norms):
    """
    Number of leading, contiguously-written bands in a norms vector: the index of
    the first zero or non-finite entry, or len(norms) if all are valid. A real
    Bloch band has a finite nonzero norm, and the old writer filled bands in
    ascending contiguous order leaving unwritten rows as zeros, so this is an
    exact boundary between written and unwritten bands.
    """
    norms = np.asarray(norms)
    nb = norms.shape[0]
    bad = ~np.isfinite(norms) | (norms == 0)
    idx = np.nonzero(bad)[0]
    if idx.size == 0:
        return nb           # all bands written -> complete
    return int(idx[0])      # first unwritten band -> count of written bands


def mark_files(paths, whitener_path=None):
    """
    Add a 'bands_done' marker to each true_blochstates.h5 in paths. Returns
    (marked, skipped) where each entry is (path, detail).
    """
    expected_uid = None
    if whitener_path is not None:
        expected_uid = whitener_uid_from_file(whitener_path)
        print(f"[MARK] whitener w_uid = {expected_uid}")
        print("")

    marked, skipped = [], []
    for path in paths:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            print(f"[MARK] - SKIP {path}: file missing")
            skipped.append((path, "file missing"))
            continue
        try:
            with h5py.File(path, "r") as f:
                if "norms" not in f or "C" not in f:
                    print(f"[MARK] - SKIP {path}: missing norms/C dataset")
                    skipped.append((path, "missing norms/C"))
                    continue
                norms = f["norms"][...]
                nb_total = int(f["C"].shape[0])
                file_uid = f.attrs.get("w_uid", None)
                already = int(f["bands_done"][()]) if "bands_done" in f else None
        except Exception as e:
            print(f"[MARK] - SKIP {path}: unreadable ({type(e).__name__}: {e})")
            skipped.append((path, f"unreadable ({type(e).__name__})"))
            continue

        # Provenance check.
        if expected_uid is not None:
            if isinstance(file_uid, bytes):
                file_uid = file_uid.decode()
            if file_uid is None:
                print(f"[MARK] - SKIP {path}: no w_uid stamp")
                skipped.append((path, "no w_uid stamp"))
                continue
            if str(file_uid) != str(expected_uid):
                print(f"[MARK] - SKIP {path}: w_uid mismatch "
                      f"(file={str(file_uid)[:12]}, expected={str(expected_uid)[:12]})")
                skipped.append((path, "w_uid mismatch"))
                continue

        done = _count_written_bands(norms)
        with h5py.File(path, "a") as f:
            if "bands_done" in f:
                f["bands_done"][()] = np.int64(done)
            else:
                d = f.create_dataset("bands_done", shape=(), dtype="int64")
                d[()] = np.int64(done)

        status = "complete" if done == nb_total else f"partial -> will resume at band {done}"
        prev = "" if already is None else f" (was {already})"
        print(f"[MARK] + {path}: bands_done={done}/{nb_total}{prev}  [{status}]")
        marked.append((path, f"bands_done={done}/{nb_total}"))

    print("")
    print(f"[MARK] Done: {len(marked)} marked, {len(skipped)} skipped.")
    if skipped:
        print("[MARK] Skipped:")
        for p, r in skipped:
            print(f"[MARK]     {p}  --  {r}")
    return marked, skipped


if __name__ == "__main__":
    # ---- EDIT THESE ----
    base = r"D:/Docs/VASP/NHC/iPr/SAM/NHC2Au/SOC"
    whitener_path = base + r"/whitener.h5"   # set to None to skip the provenance check

    component_dirs = [
        base,                       # full system (the incomplete one)
        base + "/adatom_surface",   # metal
        base + "/NHC_c1m1",
        base + "/NHC_c1m2",
        base + "/NHC_c2m1",
        base + "/NHC_c2m2",
        base + "/NHC_c3m1",
        base + "/NHC_c3m2",
        base + "/NHC_c4m1",
        base + "/NHC_c4m2",
    ]
    paths = [os.path.join(d, "true_blochstates.h5") for d in component_dirs]
    # --------------------

    mark_files(paths, whitener_path=whitener_path)
