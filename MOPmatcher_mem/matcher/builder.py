# -*- coding: utf-8 -*-
"""
Created on Wed Oct 8 21:18:18 2025
Updated for NCL/SOC auto-detection.

@author: Benjamin Kafin
"""

import os
import re
import h5py
import hashlib
import numpy as np
from typing import Dict, List, Optional, Sequence
from scipy.sparse import issparse, identity, coo_matrix, block_diag as sp_block_diag
from scipy.linalg import block_diag as la_block_diag
from scipy.optimize import linear_sum_assignment
from ase.io import read as ase_read
from joblib import Parallel, delayed
from collections import defaultdict

from vaspwfc import vaspwfc
from aewfc import vasp_ae_wfc


def whitener_uid_from_W(W):
    """
    Canonical provenance id for a whitener: the SHA-256 hex digest of the
    whitener matrix's raw bytes (C-contiguous, complex128). This is the single
    definition of a run's identity. It is content-derived, so it both uniquely
    identifies the build AND detects corruption of W: any change to W's bytes
    changes the digest. The stamping script and the builder both call this on the
    W read back from whitener.h5, so the ids they compute agree exactly (W
    round-trips through HDF5 losslessly).
    """
    Wc = np.ascontiguousarray(W, dtype=np.complex128)
    return hashlib.sha256(Wc.tobytes()).hexdigest()


def whitener_uid_from_file(whitener_h5_path):
    """
    Read W from a whitener.h5 and return its canonical w_uid (see
    whitener_uid_from_W). Raises if the file or its 'W' dataset is unreadable.
    """
    with h5py.File(whitener_h5_path, "r") as f:
        W = f["W"][...]
    return whitener_uid_from_W(W)


class _ae_wfc_no_grid(vasp_ae_wfc):
    """
    vasp_ae_wfc variant that skips building the all-electron real-space grid.

    The MOPM builder uses only get_qijs() (PAW augmentation overlap from the
    POTCAR radial integrals) and get_beta_njk() (PS-space projector inner
    products via self._q_proj). Neither touches the AE real-space grid that
    vasp_ae_wfc.set_aecut() constructs (G-vectors, phase factors, ylm, SBT core
    terms). For a large slab that grid's per-atom phase array can be tens of GiB.

    vasp_ae_wfc.__init__ calls set_aecut() as its final step; overriding it to a
    no-op leaves every other initialized quantity (_q_proj, _pawpp,
    _element_idx, ...) intact while never allocating the grid. The real-space
    methods (get_ae_wfc / get_dipole_mat / get_moment_mat / get_ae_norm) are not
    used here; if ever needed, call the parent set_aecut() explicitly to build it.
    """
    def set_aecut(self, aecut=-4):
        return


def detect_ncl_from_incar(directory):
    """
    Auto-detect non-collinear / SOC calculation from INCAR.
    Returns True if LSORBIT=.TRUE. or LNONCOLLINEAR=.TRUE. found.
    Falls back to False if INCAR is missing.
    """
    incar_path = os.path.join(directory, "INCAR")
    if not os.path.isfile(incar_path):
        return False
    try:
        with open(incar_path, "r") as f:
            text = f.read().upper()
        # Strip comments
        text = re.sub(r"[!#].*", "", text)
        for tag in ("LSORBIT", "LNONCOLLINEAR"):
            pattern = rf"{tag}\s*=\s*\.?\s*(TRUE|T)\b"
            if re.search(pattern, text):
                return True
    except Exception:
        pass
    return False


class SystemData:
    """A simple data container for system properties."""
    def __init__(self, name: str, directory: str):
        self.name = name
        self.directory = directory
        self.ps: vaspwfc = None
        self.ae: vasp_ae_wfc = None
        self.atoms = None
        self.C_by_k: Dict[int, np.ndarray] = {}
        self.kpoints: List[int] = []
        self.nspins: int = 1
        self.gamma_energies: Optional[np.ndarray] = None
        self.ch_per_atom: List[int] = []
        self.nproj_total: int = 0


class TrueBlochStateBuilder:
    """
    Handles the computationally expensive task of building and saving
    true Bloch state wavefunctions (true_blochstates.h5 files) from VASP outputs.
    Auto-detects NCL/SOC from INCAR files.
    """
    def __init__(self, molecule_dirs, metal_dir, full_dir, **kwargs):
        self.molecule_dirs = molecule_dirs
        self.metal_dir = metal_dir
        self.full_dir = full_dir
        self.k_index = kwargs.get("k_index", 1)
        self.tol_map = kwargs.get("tol_map", 1e-3)
        self.check_species = kwargs.get("check_species", True)
        # Reuse a cached whitener (whitener.h5 in full_dir) when valid, mirroring
        # reuse_cached for the Bloch-state .h5 files.
        self.reuse_W = kwargs.get("reuse_W", True)
        # When True, build_all skips systems whose true_blochstates.h5 already
        # loads cleanly and rebuilds only the missing/corrupt ones. When False,
        # all systems are rebuilt (matches the matcher's reuse_cached semantics).
        self.reuse_cached = kwargs.get("reuse_cached", False)

        # Auto-detect NCL from INCAR, allow kwarg override
        self.lsorbit = kwargs.get("lsorbit", None)
        if self.lsorbit is None:
            self.lsorbit = detect_ncl_from_incar(full_dir)
        if self.lsorbit:
            print(f"[BUILDER] NCL/SOC detected (lsorbit=True)")

    def _log(self, msg):
        """
        Emit a progress message to (1) the parent process via the log queue, if
        one was attached, so it prints live in the console (a spawned child's own
        stdout is not shown in Spyder's IPython console); (2) stdout, for non-
        Spyder / non-subprocess use; and (3) the per-run build logfile
        (full_dir/build_progress.log), flushed immediately so the file reflects
        progress live and a crash leaves everything up to that point on disk. The
        queue and the logfile handle may both be absent (e.g. standalone callers),
        in which case only stdout is written.
        """
        q = getattr(self, "_log_queue", None)
        if q is not None:
            try:
                q.put(msg)
            except Exception:
                pass
        print(msg)
        logf = getattr(self, "_logf", None)
        if logf is not None:
            try:
                logf.write(msg + "\n")
                logf.flush()
            except Exception:
                pass

    def build_all(self):
        # Open a fresh per-run progress logfile in full_dir (overwrite each run),
        # closed in the finally block so a crash mid-build still leaves a complete
        # post-mortem of everything written up to the failure.
        self._logf = open(os.path.join(self.full_dir, "build_progress.log"), "w")
        try:
            self._build_all_inner()
        finally:
            try:
                self._logf.close()
            finally:
                self._logf = None

    def _build_all_inner(self):
        self._log("[BUILDER] Starting generation of true Bloch states...")

        # ============================ DECISION PHASE ============================
        # Resolve the whitener ONCE, scan every system against it, and print the
        # full plan (what will be reused, what will be rebuilt and why) BEFORE
        # building anything. No component WAVECAR is loaded in this phase -- only
        # the full system's metadata (needed for the whitener) and the existing
        # .h5 files (read for validity) are touched.

        # Full system metadata only (lazy WAVECAR handle, no AE real-space grid).
        # It is the reference every component maps to, so it stays resident for
        # the whole build; it is the only system kept in memory across the loop.
        full_system = self.load_system(self.full_dir, "full", load_coeffs=False)

        # Resolve the single whitener W and its provenance id in ONE call (no
        # separate cache probe, so any cache-mismatch warning is logged once).
        # reuse_W lets a valid cached whitener be loaded; otherwise it is built
        # from full's Q. from_cache reports which happened, for the plan message.
        W, w_uid, whitener_from_cache = self._load_or_build_whitener(full_system)
        if whitener_from_cache:
            self._log(f"[BUILDER] Using cached whitener (w_uid={w_uid[:12]}).")
        else:
            self._log(f"[BUILDER] No valid cached whitener; built from scratch "
                      f"(w_uid={w_uid[:12]}).")

        northo = W.shape[1]  # retained whitener rank; every B_ortho must match.

        # System roster in processing order: full first, then metal, then the
        # molecules. (name, directory, kind) -- kind drives how it is loaded.
        roster = [("full", self.full_dir, "full"),
                  ("metal", self.metal_dir, "metal")]
        roster += [(os.path.basename(os.path.normpath(md)), md, "mol")
                   for md in self.molecule_dirs]

        # Decide reuse vs rebuild for every system up front.
        #   - A freshly built whitener invalidates ALL prior files (their stamp
        #     cannot match the new w_uid), so everything is rebuilt. We still run
        #     the per-file check so the printed reason is accurate.
        #   - reuse_cached off forces a full rebuild regardless.
        to_reuse, to_build = [], []
        for name, directory, kind in roster:
            if not self.reuse_cached:
                to_build.append((name, directory, kind, "reuse_cached disabled"))
                continue
            ok, reason = self._blochstates_valid(
                directory, expected_w_uid=w_uid, expected_northo=northo)
            if ok:
                to_reuse.append((name, directory, kind))
            else:
                to_build.append((name, directory, kind, reason))

        # ----- Print the full plan before doing any work -----
        if whitener_from_cache:
            self._log("[BUILDER] Checking existing true_blochstates.h5 files against "
                      f"whitener (w_uid={w_uid[:12]})...")
        else:
            self._log("[BUILDER] Whitener was rebuilt -- all systems will be regenerated.")
        self._log(f"[BUILDER] Valid / will reuse ({len(to_reuse)}):")
        if to_reuse:
            for name, _d, _k in to_reuse:
                self._log(f"[BUILDER]     {name}")
        else:
            self._log("[BUILDER]     (none)")
        self._log(f"[BUILDER] Will rebuild ({len(to_build)}):")
        if to_build:
            for name, _d, _k, reason in to_build:
                self._log(f"[BUILDER]     {name}  --  {reason}")
        else:
            self._log("[BUILDER]     (none)")
        self._log(f"[BUILDER] Plan: {len(to_reuse)} reuse, {len(to_build)} rebuild. "
                  "Starting build...")

        # ============================ EXECUTION PHASE ===========================
        # Build only the systems in to_build, in roster order (full first), one
        # system at a time. Full is already resident; each component is loaded,
        # processed, and released (WAVECAR handle closed) before the next, so at
        # most one component's heavy objects are in memory alongside full + W.

        # Channel counts/atoms of the full system, used to map every component.
        full_ch = full_system.ch_per_atom

        for name, directory, kind, _reason in to_build:
            if kind == "full":
                sys_data = full_system  # already loaded; do not reload
                T = identity(sum(full_ch), format="csr")
                if self.lsorbit:
                    T = identity(2 * sum(full_ch), format="csr")
                release_after = False  # full is reused as the mapping reference
            else:
                self._log(f"[BUILDER] Loading system '{name}'...")
                sys_data = self.load_system(directory, name, load_coeffs=False)
                amap = self.map_atoms_by_coords(sys_data.atoms, full_system.atoms)
                T = self.build_T_injection(full_ch, sys_data.ch_per_atom, amap)
                if self.lsorbit:
                    T = sp_block_diag([T, T], format="csr")
                release_after = True

            self._log(f"[BUILDER] Processing system: {name}")
            self._process_and_save(sys_data, T, directory, W, w_uid=w_uid)
            self._log(f"[BUILDER] \u2713 Wrote true_blochstates.h5 for '{name}' "
                      f"(stamped w_uid={w_uid[:12]}).")

            if release_after:
                # Close the WAVECAR file handle and drop the heavy objects so only
                # one component is resident at a time.
                try:
                    sys_data.ps._wfc.close()
                except Exception:
                    pass
                sys_data.ps = None
                sys_data.ae = None
                del sys_data
                self._log(f"[BUILDER] Released system '{name}'.")

        for name, _d, _k in to_reuse:
            self._log(f"[BUILDER] Reused existing true_blochstates.h5 for '{name}'.")

        self._log("[BUILDER] All true Bloch state .h5 files have been generated.")

    @staticmethod
    def _blochstates_valid(directory, expected_w_uid=None, expected_northo=None):
        """
        Check whether directory/true_blochstates.h5 is present, intact, usable,
        and (if expected_w_uid is given) provably from the current whitener's run.

        Returns a tuple (ok: bool, reason: str). reason is "" when ok is True, and
        otherwise a short human-readable explanation for the console/log so the
        user sees WHY a file is being rebuilt.

        Checks, in order:
          - file exists
          - contains C, B_ortho, norms
          - every chunk passes its fletcher32 checksum (read raises on bit-rot)
          - has a bands_done marker equal to the band count (completeness): a
            missing marker or bands_done < nbands means a partial/resumable build,
            reported invalid so it is not used as a finished result
          - C, B_ortho, norms share the same band count (catches partial writes)
          - every per-band norm is finite and nonzero (the matcher divides by
            norms on read; zero/non-finite would yield inf/nan)
          - if expected_northo is given: B_ortho width == whitener rank (a file
            built against a differently-ranked whitener is incompatible)
          - if expected_w_uid is given: the file's stored 'w_uid' attribute equals
            it (provenance: the file belongs to the current whitener's run). A
            missing stamp or a mismatch is invalid, so a fresh whitener forces a
            rebuild of every component and stale/foreign files are never reused.
        """
        path = os.path.join(directory, "true_blochstates.h5")
        if not os.path.isfile(path):
            return False, "file missing"
        try:
            with h5py.File(path, "r") as f:
                expected = {"C", "B_ortho", "norms"}
                if not expected.issubset(set(f.keys())):
                    return False, "missing C/B_ortho/norms dataset"
                dC = f["C"]
                dB = f["B_ortho"]
                # Shapes/attrs are metadata (no array data loaded) -- safe even for
                # multi-GiB files that cannot be held in RAM.
                cshape = dC.shape
                bshape = dB.shape
                nshape = f["norms"].shape
                file_uid = f.attrs.get("w_uid", None)
                bands_done = int(f["bands_done"][()]) if "bands_done" in f else None
                nbands = int(cshape[0])

                # norms is one value per band (tiny); read it whole. This both
                # triggers its chunk checksums and gives the finite/nonzero check.
                norms = np.asarray(f["norms"][...])

                # Verify every chunk's fletcher32 checksum by READING THE DATA, but
                # in bounded band-tiles so the whole (possibly tens-of-GiB) array is
                # never resident. Reading a tile forces HDF5 to verify the checksum
                # of every chunk it touches (raising on corruption), so scanning all
                # bands tile-by-tile gives the same corruption coverage as reading
                # the whole array at once -- the HDF5 analogue of zipfile testzip()
                # -- without the memory blow-up. The read result is discarded; only
                # the act of reading (and its checksum verification) matters. Tile
                # is sized to ~256 MiB worth of bands across both C and B_ortho.
                per_band_bytes = max(1, dC.dtype.itemsize * int(cshape[1])
                                     + dB.dtype.itemsize * int(bshape[1]))
                scan_tile = max(1, int((256 * 1024**2) // per_band_bytes))
                for b0 in range(0, nbands, scan_tile):
                    b1 = min(b0 + scan_tile, nbands)
                    # Touch the data so fletcher32 is checked for these chunks.
                    dC[b0:b1, :]
                    dB[b0:b1, :]
            # Completeness: a fully built file records bands_done == nbands. A
            # missing marker or bands_done < nbands means the file is a partial
            # (resumable) build and must NOT be treated as a usable result.
            if bands_done is None:
                return False, "no bands_done marker (incomplete or pre-resume file)"
            if bands_done != nbands:
                return False, (f"incomplete: bands_done={bands_done} of "
                               f"{nbands} bands")
            # Shapes must be mutually consistent: one norm per band, and C and
            # B_ortho must have that same number of bands (rows). Catches partial
            # writes that still checksum per-chunk.
            if not (cshape[0] == bshape[0] == nshape[0]):
                return False, (f"shape mismatch (C={cshape[0]}, "
                               f"B_ortho={bshape[0]}, norms={nshape[0]} bands)")
            # The loader divides by norms, so every band norm must be finite and
            # strictly nonzero. A real Bloch state cannot have zero norm; a zero
            # or non-finite value marks the file as improperly written.
            if not np.all(np.isfinite(norms)):
                return False, "norms contain non-finite values"
            if np.any(norms == 0):
                return False, "norms contain zero"
            # B_ortho width must match the current whitener's retained rank.
            if expected_northo is not None and bshape[1] != int(expected_northo):
                return False, (f"B_ortho width {bshape[1]} != "
                               f"whitener rank {int(expected_northo)}")
            # Provenance: stamp must be present and match the current whitener.
            if expected_w_uid is not None:
                if file_uid is None:
                    return False, "no w_uid stamp"
                if isinstance(file_uid, bytes):
                    file_uid = file_uid.decode()
                if str(file_uid) != str(expected_w_uid):
                    return False, (f"w_uid mismatch (file={str(file_uid)[:12]}, "
                                   f"expected={str(expected_w_uid)[:12]})")
            return True, ""
        except Exception as e:
            return False, f"unreadable/corrupt ({type(e).__name__})"

    def _load_or_build_whitener(self, full_system, cache_only=False):
        """
        Build the whitener W from the full system's PAW overlap Q, or load it
        from a cached 'whitener.h5' in the full directory when reuse_W is set
        and the cache is valid. Only one W is ever produced and it is applied to
        every system, so the cross-system AE overlaps stay consistent.

        Returns (W, w_uid, from_cache): the whitener matrix, its canonical
        provenance id (SHA-256 of W's bytes; see whitener_uid_from_W), and a bool
        that is True when W was loaded from a valid cache and False when built
        from scratch. Every system built with this W is stamped with this w_uid,
        and reuse requires the stamp to match.

        If cache_only=True, return (W, w_uid, True) from a valid cache or
        (None, None, False) WITHOUT building or saving.

        A cached whitener is accepted only if (a) its fingerprint (SOC flag,
        dimension, per-atom channel counts, Q checksum) matches the current full
        system AND (b) the SHA-256 of its stored W equals its stored 'w_uid'
        attribute -- so a corrupted W (whose bytes no longer hash to the recorded
        id) is rejected even though the cheap fingerprint still matches.

        For SOC/NCL the doubled beta vectors require a doubled whitener. Since
        Q_doubled = block_diag(Q, Q), its exact matrix square root is
        block_diag(W_block, W_block); the (2N)x(2N) eigendecomposition is never
        formed.
        """
        # Undoubled PAW overlap for the full system (Hermitian-symmetrized).
        Q_block = 0.5 * (full_system.ae.get_qijs() + full_system.ae.get_qijs().getH())

        # Cheap fingerprint of what W depends on: SOC flag, channel dimension,
        # per-atom channel counts, and a checksum of Q's entries. Q (the PAW
        # augmentation overlap) depends only on the elements/POTCAR, not on
        # geometry, so this fully identifies a valid cache.
        Qcsr = Q_block.tocsr()
        n_block = int(Qcsr.shape[0])
        q_checksum = float(np.sum(np.abs(Qcsr.data)))
        ch_per_atom = np.asarray(full_system.ch_per_atom, dtype=np.int64)
        lsorbit_flag = 1 if self.lsorbit else 0

        cache_path = os.path.join(self.full_dir, "whitener.h5")

        if self.reuse_W and os.path.isfile(cache_path):
            try:
                with h5py.File(cache_path, "r") as cached:
                    cached_ch = cached["ch_per_atom"][...]
                    fingerprint_ok = (
                        int(cached["lsorbit"][()]) == lsorbit_flag
                        and int(cached["n_block"][()]) == n_block
                        and tuple(cached_ch.shape) == tuple(ch_per_atom.shape)
                        and np.array_equal(cached_ch, ch_per_atom)
                        and abs(float(cached["q_checksum"][()]) - q_checksum) <= 1e-6 * (1.0 + abs(q_checksum))
                    )
                    W_cached = cached["W"][...]
                    stored_uid = cached.attrs.get("w_uid", None)
                if fingerprint_ok:
                    computed_uid = whitener_uid_from_W(W_cached)
                    if stored_uid is None:
                        self._log("[BUILDER][WARN] Cached whitener has no w_uid stamp; rebuilding.")
                    else:
                        if isinstance(stored_uid, bytes):
                            stored_uid = stored_uid.decode()
                        if str(stored_uid) == computed_uid:
                            return W_cached, computed_uid, True
                        self._log("[BUILDER][WARN] Cached whitener W does not match its w_uid "
                                  "(corrupted); rebuilding.")
                else:
                    self._log("[BUILDER][WARN] Cached whitener does not match the current full "
                              "system (SOC flag / dimension / Q checksum); rebuilding.")
            except Exception as e:
                self._log(f"[BUILDER][WARN] Failed to read cached whitener ({e}); rebuilding.")

        if cache_only:
            return None, None, False

        # Build: eigendecompose the single (undoubled) block.
        self._log("[BUILDER] Building whitener (eigendecomposing PAW overlap)...")
        W_block, qinfo = self.build_whitener(Q_block)
        if self.lsorbit:
            W = la_block_diag(W_block, W_block)
        else:
            W = W_block
        # Promote to complex128 so the returned W, the cache-reuse return
        # (cached["W"][...]), and the stored dataset all share one dtype. The
        # whitener is real (eigenvectors of the real-symmetric PAW overlap Q), so
        # this is a zero-imaginary promotion; B @ W is bit-identical either way.
        W = W.astype("complex128")
        # Canonical provenance id, computed from the exact bytes about to be
        # written (W is contiguous complex128 here, and round-trips through HDF5
        # losslessly, so reading W back and re-hashing yields the same id).
        w_uid = whitener_uid_from_W(W)
        self._log(f"[BUILDER] Whitener built: rank={qinfo['rank']}"
                  + (" (per spin block)" if self.lsorbit else "")
                  + f" (w_uid={w_uid[:12]})")

        with h5py.File(cache_path, "w") as f:
            # Provenance stamp identifying this whitener build.
            f.attrs["w_uid"] = w_uid
            # Large array: complex128, lzf + shuffle + fletcher32 (lossless).
            f.create_dataset("W", data=W, chunks=True,
                             compression="lzf", shuffle=True, fletcher32=True)
            # Fingerprint scalars (plain datasets; fletcher32 requires chunking
            # and is not applicable to scalars). A mismatch only forces a rebuild.
            f.create_dataset("lsorbit", data=np.int64(lsorbit_flag))
            f.create_dataset("n_block", data=np.int64(n_block))
            f.create_dataset("q_checksum", data=np.float64(q_checksum))
            # 1-D fingerprint index array (channel counts): integer metadata used
            # only for the cache-validity comparison, kept as int64.
            f.create_dataset("ch_per_atom", data=ch_per_atom, chunks=True,
                             compression="lzf", shuffle=True, fletcher32=True)
        return W, w_uid, False

    def _read_coeff_slice(self, sys_data):
        """
        Read and stack the plane-wave coefficients for all bands at k_index for
        one system, returning an (nbands, ncoeff) complex array. For SOC,
        ncoeff = 2*nplw (stacked spinor). Called per system so the large
        coefficient array is not held for every system simultaneously.
        """
        rows = []
        for i in range(1, sys_data.ps._nbands + 1):
            coeff = sys_data.ps.readBandCoeff(ispin=1, ikpt=self.k_index, iband=i, norm=False)
            rows.append(np.asarray(coeff, dtype=np.complex128))
        return np.vstack(rows)

    def _choose_band_tile(self, ncoeff, nproj, northo, nbands):
        """
        Pick how many bands to process at once so peak builder RAM stays bounded.
        The estimate counts EVERY array simultaneously alive inside one tile
        iteration (see _process_and_save), not just one row, so the budget
        fraction means what it says. Per band, at peak, the loop holds:
          C_rows            16*ncoeff   (raw coefficients, complex128)
          C_rows.conj()     16*ncoeff   (einsum temporary for the norm)
          B_native          16*nproj    (projector inner products)
          B_lifted          16*nproj    (lifted to full channel space)
          B_ortho           16*northo   (B_lifted @ W, whitener-rank wide)
          B_ortho.conj()    16*northo   (einsum temporary for the norm)
          vstack transient  16*ncoeff   (C_rows assembled from a per-band list)
        i.e. bytes_per_band = 16*(3*ncoeff + 2*nproj + 2*northo). (norms is
        nbands-length and negligible.) The tile uses up to 2/3 of currently
        available memory against this realistic per-band cost, leaving ~1/3 free
        for other transient copies and other processes. Falls back to a
        conservative 2 GiB budget if psutil is unavailable (e.g. not on the
        target Windows host).
        """
        try:
            import psutil
            avail_bytes = int(psutil.virtual_memory().available)
            avail_src = "psutil available"
        except Exception:
            avail_bytes = 2 * 1024**3  # conservative 2 GiB fallback
            avail_src = "psutil unavailable -> 2 GiB fallback"
        bytes_per_band = max(1, 16 * (3 * int(ncoeff) + 2 * int(nproj) + 2 * int(northo)))
        budget = (2.0 / 3.0) * avail_bytes
        tile = int(budget // bytes_per_band)
        tile = max(1, min(tile, int(nbands)))
        gib = 1024.0 ** 3
        mib = 1024.0 ** 2
        self._log(
            f"[BUILDER]   tiling: avail RAM={avail_bytes / gib:.2f} GiB ({avail_src}); "
            f"budget(2/3)={budget / gib:.2f} GiB; per-band={bytes_per_band / mib:.2f} MiB "
            f"(ncoeff={int(ncoeff)}, nproj={int(nproj)}, northo={int(northo)}); "
            f"tile={tile} bands of {int(nbands)}"
        )
        return tile

    @staticmethod
    def _form_B_for_rows(ae, C_rows):
        """
        Projector inner products for a block of bands (rows of C_rows), returning
        an (nrows, nproj) complex array. Uses the batched (GEMM) projection
        (proj_batched.get_beta_njk_batched), which computes the whole block in a
        few matrix multiplies instead of a per-band/per-atom Python loop. The
        result is numerically identical to looping ae.get_beta_njk per band (only
        the floating-point summation order differs, at the ~1e-12 level), and it
        does NOT modify Zheng's paw.py/aewfc.py -- it only reads the projector
        arrays those already built and reuses ae._q_proj.
        """
        from proj_batched import get_beta_njk_batched
        return np.ascontiguousarray(
            np.asarray(get_beta_njk_batched(ae, C_rows), dtype=np.complex128)
        )

    def _process_and_save(self, sys_data, T, out_dir, W, w_uid=None):
        """
        Build and save one system's true Bloch states, streaming over band tiles
        so the full (nbands x 2*nplw) coefficient matrix is never resident. Each
        tile is read, projected, lifted, orthogonalized (B @ W) and its per-band
        norms computed, then the raw C rows, the B_ortho rows and the norms are
        written straight into the on-disk datasets. The stored values are
        bit-identical to the whole-array path: C is stored raw, and B_ortho /
        norms are computed per band (row-independent), so tiling changes nothing
        but peak memory.

        w_uid, when given, is written as a 'w_uid' attribute stamping this file as
        belonging to the current whitener's run, so reuse can be proven later.
        """
        nbands = sys_data.ps._nbands
        # ncoeff (per-band coefficient length) from band 1; nproj (channel count,
        # SOC-doubled) from W's row dimension, which the B rows must match.
        first = np.asarray(
            sys_data.ps.readBandCoeff(ispin=1, ikpt=self.k_index, iband=1, norm=False),
            dtype=np.complex128,
        )
        ncoeff = first.shape[0]
        # B_ortho = (B @ T^H) @ W has width W.shape[1] (the whitener's retained
        # rank), NOT W.shape[0] (the channel dimension); the whitener is
        # rectangular (U[:, keep] * sqrt(w[keep])). Use the rank for the stored
        # dataset width. nproj (channel count) is used only for the memory-tile
        # size estimate below.
        nproj = W.shape[0]
        northo = W.shape[1]
        tile = self._choose_band_tile(ncoeff, nproj, northo, nbands)

        # Crash-hardening: build into a temporary file in the SAME directory and
        # atomically rename it to the final name only when fully written. The
        # matcher only ever reads the final 'true_blochstates.h5', so it never
        # sees a partial: a killed build leaves the in-progress data in the .tmp
        # (a valid, resumable HDF5 thanks to the per-tile flush below) while the
        # final file is either absent or the last complete version. os.replace is
        # atomic on the same volume, so even a kill during the rename yields either
        # the old file or the new one, never a corrupt mix. (This is why writing
        # straight to the final name previously left an unopenable file when a run
        # was killed mid-write.)
        final_path = os.path.join(out_dir, "true_blochstates.h5")
        tmp_path = final_path + ".tmp"

        # Decide fresh build vs resume from the TEMP file. A resumable partial is
        # an existing .tmp with the matching schema (nbands, ncoeff, northo), the
        # SAME whitener stamp (w_uid), and a bands_done marker with
        # 0 < bands_done < nbands. Each tile writes a contiguous band block [b0:b1]
        # to all three datasets at once in ascending order, so the written region
        # is always bands [0:bands_done]; resuming re-tiles only the remaining
        # [bands_done:nbands], producing a file bit-identical to a single run
        # (per-band rows are independent). A stale/mismatched .tmp (wrong w_uid,
        # wrong schema, or unreadable) is discarded by starting fresh ("w").
        resume_from = 0
        if os.path.isfile(tmp_path):
            try:
                with h5py.File(tmp_path, "r") as f:
                    have = {"C", "B_ortho", "norms"}.issubset(set(f.keys()))
                    bd = int(f["bands_done"][()]) if "bands_done" in f else None
                    f_uid = f.attrs.get("w_uid", None)
                    if isinstance(f_uid, bytes):
                        f_uid = f_uid.decode()
                    schema_ok = (
                        have
                        and f["C"].shape == (nbands, ncoeff)
                        and f["B_ortho"].shape == (nbands, northo)
                        and f["norms"].shape == (nbands,)
                    )
                stamp_ok = (w_uid is None) or (f_uid is not None and str(f_uid) == str(w_uid))
                if schema_ok and stamp_ok and bd is not None and 0 < bd < nbands:
                    resume_from = bd
            except Exception:
                resume_from = 0  # unreadable/corrupt .tmp -> start fresh

        if resume_from > 0:
            self._log(f"[BUILDER]   resuming '{os.path.basename(os.path.normpath(out_dir))}' "
                      f"from band {resume_from}/{nbands}")
            fmode = "r+"
        else:
            fmode = "w"

        with h5py.File(tmp_path, fmode) as f:
            if fmode == "w":
                # Provenance stamp tying this file to the whitener that built it.
                # Written at creation so a partial file proves which whitener it
                # was started with; completeness is tracked separately by
                # bands_done (== nbands only when fully written).
                if w_uid is not None:
                    f.attrs["w_uid"] = w_uid
                dC = f.create_dataset(
                    "C", shape=(nbands, ncoeff), dtype="complex128",
                    chunks=(1, ncoeff), compression="lzf", shuffle=True, fletcher32=True,
                )
                dB = f.create_dataset(
                    "B_ortho", shape=(nbands, northo), dtype="complex128",
                    chunks=(1, northo), compression="lzf", shuffle=True, fletcher32=True,
                )
                dN = f.create_dataset(
                    "norms", shape=(nbands,), dtype="complex128",
                    chunks=True, compression="lzf", shuffle=True, fletcher32=True,
                )
                # Completion marker: number of contiguous bands written from 0.
                dDone = f.create_dataset("bands_done", shape=(), dtype="int64")
                dDone[()] = 0
                f.flush()
            else:
                dC = f["C"]
                dB = f["B_ortho"]
                dN = f["norms"]
                dDone = f["bands_done"]

            for b0 in range(resume_from, nbands, tile):
                b1 = min(b0 + tile, nbands)
                # Read this tile's raw coefficients (one band per row).
                C_rows = np.vstack([
                    np.asarray(
                        sys_data.ps.readBandCoeff(ispin=1, ikpt=self.k_index, iband=ib + 1, norm=False),
                        dtype=np.complex128,
                    )
                    for ib in range(b0, b1)
                ])
                # Projector inner products -> lift to full channel space -> B @ W.
                B_native = self._form_B_for_rows(sys_data.ae, C_rows)
                B_lifted = self.lift_B(B_native, T)
                B_ortho = B_lifted @ W
                # Per-band AE norm from the two blocks separately (no [C | B_ortho]
                # materialization); identical to fuse_true_bloch_rr's einsum.
                norms = np.sqrt(
                    (np.einsum("ij,ij->i", C_rows.conj(), C_rows)
                     + np.einsum("ij,ij->i", B_ortho.conj(), B_ortho)).real
                ).astype("complex128")

                dC[b0:b1, :] = C_rows
                dB[b0:b1, :] = B_ortho
                dN[b0:b1] = norms
                # Mark this contiguous block complete and flush, so a crash leaves
                # a consistent "bands [0:b1] done" state in the .tmp that a later
                # run resumes from.
                dDone[()] = b1
                f.flush()
                del C_rows, B_native, B_lifted, B_ortho, norms

        # Fully written: atomically promote the temp file to the final name the
        # matcher reads. os.replace overwrites any existing final file in one
        # atomic step on the same volume.
        os.replace(tmp_path, final_path)

    def load_system(self, directory, name, load_coeffs=True):
        sys = SystemData(name, directory)

        # Auto-detect NCL for this specific directory too
        dir_ncl = detect_ncl_from_incar(directory)
        use_lsorbit = self.lsorbit or dir_ncl

        sys.ps = vaspwfc(os.path.join(directory, "WAVECAR"), lsorbit=use_lsorbit)
        sys.nspins = sys.ps._nspin

        # Per-band coefficients are read lazily via _read_coeff_slice so only one
        # system's coefficients are resident at a time. Kept optional so that
        # standalone callers of load_system still get C_by_k populated.
        if load_coeffs:
            sys.C_by_k[self.k_index] = self._read_coeff_slice(sys)

        sys.atoms = ase_read(os.path.join(directory, "POSCAR"))
        sys.ae = _ae_wfc_no_grid(sys.ps, poscar=os.path.join(directory, "POSCAR"),
                             potcar=os.path.join(directory, "POTCAR"))
        sys.ch_per_atom = [sys.ae._pawpp[it].lmmax for it in sys.ae._element_idx]
        self._log(f"[BUILDER] Loaded system '{name}' "
                  f"({len(sys.atoms)} atoms, {sys.ps._nbands} bands).")
        return sys

    @staticmethod
    def map_atoms_by_coords(comp_atoms, full_atoms, tol=1e-3, check_species=True):
        comp_frac = comp_atoms.get_scaled_positions(wrap=True)
        full_frac = full_atoms.get_scaled_positions(wrap=True)
        cell = full_atoms.cell.array
        mapping = -np.ones(len(comp_atoms), dtype=int)
        if check_species:
            comp_by, full_by = defaultdict(list), defaultdict(list)
            for i, s in enumerate(comp_atoms.get_chemical_symbols()):
                comp_by[s].append(i)
            for j, s in enumerate(full_atoms.get_chemical_symbols()):
                full_by[s].append(j)
            for s, idx_c in comp_by.items():
                idx_f = full_by.get(s, [])
                if len(idx_f) < len(idx_c):
                    raise ValueError(f"Full system has fewer '{s}' atoms than component.")
                D = TrueBlochStateBuilder._pairwise_min_image_dists_frac(comp_frac[idx_c], full_frac[idx_f], cell)
                rows, cols = linear_sum_assignment(np.where(D > tol, 1e6, D))
                for r, c in zip(rows, cols):
                    if D[r, c] > tol:
                        raise ValueError(f"No match found for '{s}' atom within tolerance.")
                    mapping[idx_c[r]] = idx_f[c]
        else:
            D = TrueBlochStateBuilder._pairwise_min_image_dists_frac(comp_frac, full_frac, cell)
            rows, cols = linear_sum_assignment(np.where(D > tol, 1e6, D))
            for r, c in zip(rows, cols):
                if D[r, c] > tol:
                    raise ValueError("No atomic match found within tolerance.")
                mapping[r] = c
        return mapping

    @staticmethod
    def _pairwise_min_image_dists_frac(comp_frac, full_frac, cell):
        df = comp_frac[:, None, :] - full_frac[None, :, :]
        df -= np.round(df)
        dcart = np.einsum("...j,ij->...i", df, cell)
        return np.linalg.norm(dcart, axis=-1)

    @staticmethod
    def build_T_injection(full_ch_per_atom, comp_ch_per_atom, atom_map_comp_to_full):
        full_ch, comp_ch = np.asarray(full_ch_per_atom), np.asarray(comp_ch_per_atom)
        off_full = np.concatenate(([0], np.cumsum(full_ch[:-1])))
        off_comp = np.concatenate(([0], np.cumsum(comp_ch[:-1])))
        rows, cols, data = [], [], []
        for a_comp, a_full in enumerate(atom_map_comp_to_full):
            if full_ch[a_full] != comp_ch[a_comp]:
                raise ValueError("Projector channel mismatch.")
            rf0, cf0 = int(off_full[a_full]), int(off_comp[a_comp])
            for i in range(full_ch[a_full]):
                rows.append(rf0 + i)
                cols.append(cf0 + i)
                data.append(1.0)
        return coo_matrix((data, (rows, cols)), shape=(full_ch.sum(), comp_ch.sum())).tocsr()

    @staticmethod
    def build_whitener(Q, tol=None):
        Qm = Q.toarray() if issparse(Q) else np.asarray(Q)
        Qm = 0.5 * (Qm + Qm.conj().T)
        w, U = np.linalg.eigh(Qm)
        if tol is None:
            tol = max(1e-10, 1e-8 * float(w.max() if w.size else 1.0))
        keep = (w > tol)
        if not np.any(keep):
            raise ValueError("Q matrix is not positive definite; cannot build whitener.")
        W = U[:, keep] * np.sqrt(w[keep])[None, :]
        return W, {"rank": int(keep.sum())}

    @staticmethod
    def form_B_for_slice(ae, Cslice):
        # Batched (GEMM) projection over all rows at once; numerically identical to
        # the former per-band loop (summation-order differences only, ~1e-12).
        # Reads only the arrays Zheng's code already built (via ae._q_proj); does
        # not modify paw.py/aewfc.py. (Legacy whole-array path; the resumable
        # builder uses _form_B_for_rows, which is also batched.)
        from proj_batched import get_beta_njk_batched
        return np.ascontiguousarray(
            np.asarray(get_beta_njk_batched(ae, Cslice), dtype=np.complex128)
        )

    @staticmethod
    def lift_B(B_comp, T):
        return B_comp @ T.conj().T

    @staticmethod
    def fuse_true_bloch_rr(C, B, W):
        B_ortho = B @ W
        # Per-band AE norm from the two blocks separately, so the combined
        # [C | B_ortho] array is never allocated (avoids duplicating the PW
        # block). Mathematically identical to normalizing hstack([C, B_ortho]).
        norms = np.sqrt(
            (np.einsum("ij,ij->i", C.conj(), C)
             + np.einsum("ij,ij->i", B_ortho.conj(), B_ortho)).real
        )
        # Return the PRE-division C and B_ortho together with the per-band norms.
        # Normalization is applied at read time in the matcher's loader (dividing
        # each band's row by its norm), which is bit-identical to dividing here
        # and composes cleanly with band-tiling. C and B are left unmodified.
        return C, B_ortho, norms

    @staticmethod
    def save_true_bloch(directory, C, B_ortho, norms):
        # Store raw (pre-normalization) C and B_ortho plus the per-band norms.
        # Band-axis chunking with one band per chunk (chunk shape (1, ncoeff))
        # so the matcher can read arbitrary band ranges (hyperslab reads) without
        # materializing the whole dataset; the coefficient axis is never split,
        # so no coefficient contraction is ever divided across chunks. Large
        # arrays use lzf + shuffle + fletcher32 (all lossless).
        path = os.path.join(directory, "true_blochstates.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "C", data=C.astype("complex128"), chunks=(1, C.shape[1]),
                compression="lzf", shuffle=True, fletcher32=True,
            )
            f.create_dataset(
                "B_ortho", data=B_ortho.astype("complex128"), chunks=(1, B_ortho.shape[1]),
                compression="lzf", shuffle=True, fletcher32=True,
            )
            f.create_dataset(
                "norms", data=norms.astype("complex128"), chunks=True,
                compression="lzf", shuffle=True, fletcher32=True,
            )
