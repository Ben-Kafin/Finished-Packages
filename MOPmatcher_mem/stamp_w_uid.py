# -*- coding: utf-8 -*-
"""
stamp_w_uid.py — one-time provenance stamper for an existing, consistent set of
MOPMatcher .h5 files (a whitener.h5 plus the true_blochstates.h5 files that were
produced with it).

Why this exists
---------------
MOPMatcher now tags every whitener with a content-derived id (w_uid = SHA-256 of
the whitener matrix W's bytes) and stamps each true_blochstates.h5 it writes with
that id, so the builder/matcher can PROVE a component file belongs to the current
whitener's run before reusing it. Files produced before this feature carry no
stamp. This script writes the stamp onto an existing set you already trust as
coming from one run, so MOPMatcher will reuse them instead of rebuilding.

What it does
------------
1. Reads W from the given whitener.h5 and computes w_uid (the SAME function the
   builder uses, so the ids agree exactly).
2. Writes that w_uid as a 'w_uid' attribute on the whitener.h5.
3. For each given true_blochstates.h5: sanity-checks it against this whitener
   (structure, fletcher32 chunks, consistent band counts, finite nonzero norms,
   and B_ortho width == whitener rank). If it passes, writes the same w_uid as a
   'w_uid' attribute. If it FAILS a check, it is NOT stamped (so MOPMatcher will
   rebuild exactly that one) and the reason is printed.

It never rebuilds or alters the scientific data — it only adds an attribute.

Usage
-----
Edit the two paths/lists below and run it (e.g. green play button in Spyder), or
call stamp_files(whitener_path, component_paths) from your own script.
"""

import os
import sys
import h5py
import numpy as np

# Import the canonical hashing + validity logic from the package so the stamp
# computed here is byte-for-byte the same id the builder/matcher compute, and the
# sanity checks match _blochstates_valid exactly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from matcher.builder import (
    whitener_uid_from_file,
    TrueBlochStateBuilder,
)


def stamp_files(whitener_path, component_paths):
    """
    Stamp a whitener.h5 and its component true_blochstates.h5 files with the
    whitener's w_uid. Components that fail the sanity check are left unstamped.

    Returns (stamped, refused) where each is a list of (path, detail) tuples.
    """
    whitener_path = os.path.abspath(whitener_path)
    if not os.path.isfile(whitener_path):
        raise FileNotFoundError(f"whitener.h5 not found: {whitener_path}")

    # 1) Compute the canonical id from the whitener's stored W.
    w_uid = whitener_uid_from_file(whitener_path)
    print(f"[STAMP] whitener: {whitener_path}")
    print(f"[STAMP] w_uid = {w_uid}")

    # Whitener rank (retained columns) -> the width every B_ortho must have.
    with h5py.File(whitener_path, "r") as f:
        northo = int(f["W"].shape[1])
    print(f"[STAMP] whitener rank (B_ortho width) = {northo}")

    # 2) Stamp the whitener file itself.
    with h5py.File(whitener_path, "a") as f:
        f.attrs["w_uid"] = w_uid
    print("[STAMP] \u2713 stamped whitener.h5")
    print("")

    # 3) Sanity-check + stamp each component.
    stamped, refused = [], []
    for path in component_paths:
        path = os.path.abspath(path)
        directory = os.path.dirname(path)
        if not os.path.isfile(path):
            print(f"[STAMP] \u2717 REFUSED {path}: file missing")
            refused.append((path, "file missing"))
            continue
        # Validate against THIS whitener (rank + structure), but without the
        # w_uid check (these files are not yet stamped, which is the whole point).
        ok, reason = TrueBlochStateBuilder._blochstates_valid(
            directory, expected_w_uid=None, expected_northo=northo)
        if not ok:
            print(f"[STAMP] \u2717 REFUSED {path}: {reason}")
            refused.append((path, reason))
            continue
        with h5py.File(path, "a") as f:
            f.attrs["w_uid"] = w_uid
        print(f"[STAMP] \u2713 stamped {path}")
        stamped.append((path, "ok"))

    # 4) Summary.
    print("")
    print(f"[STAMP] Done: {len(stamped)} stamped, {len(refused)} refused.")
    if refused:
        print("[STAMP] Refused files (MOPMatcher will rebuild these):")
        for p, r in refused:
            print(f"[STAMP]     {p}  --  {r}")
    return stamped, refused


if __name__ == "__main__":
    # ---- EDIT THESE ----
    # Path to the whitener produced by the run you are adopting.
    whitener_path = r"D:/Docs/VASP/NHC/iPr/SAM/NHC2Au/SOC/whitener.h5"

    # All component true_blochstates.h5 files from that same run. Include the
    # full system, the metal, and every molecule directory's file. (Any file you
    # omit simply won't be stamped, so MOPMatcher would rebuild it.)
    base = r"D:/Docs/VASP/NHC/iPr/SAM/NHC2Au/SOC"
    component_dirs = [
        base,                       # full system
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
    component_paths = [os.path.join(d, "true_blochstates.h5") for d in component_dirs]
    # --------------------

    stamp_files(whitener_path, component_paths)
