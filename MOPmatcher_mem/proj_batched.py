# -*- coding: utf-8 -*-
"""
proj_batched.py — batched (GEMM) reimplementation of the PAW reciprocal-space
projection, kept SEPARATE from Zheng's paw.py so the original is untouched and the
two can be compared head-to-head before anything is replaced.

Faithful to nonlq.proj (paw.py) -- same math, no conjugation:

    current, per band, per atom i (element type t):
        beta_i[c] = sum_G  cptwf[G] * crexp[G,i] * qproj[t][c,G] * iL[t][c]

    where qproj[t] (lmmax_t x nplw) and iL[t]=cqfak[t] (lmmax_t) depend only on the
    element type, and crexp[:,i] (nplw) is the per-atom structure phase.

Batched form. For a block of bands C (nbands x nplw), per atom i of type t define
the effective projector

    P_i[c,G] = qproj[t][c,G] * iL[t][c] * crexp[G,i]        (lmmax_t x nplw)

then

    beta_block_i = C @ P_i.T                                 (nbands x lmmax_t)

a single GEMM over all bands at once, and the per-atom blocks are concatenated
along the channel axis to give (nbands x nproj). This replaces the nbands*natoms
Python-loop reductions in nonlq.proj with natoms GEMMs (one per atom, each over the
whole band block), which is what BLAS is built for. P_i is built on the fly per
atom (~lmmax_t x nplw, a few hundred MB at most) so the full nproj x nplw projector
is never materialized.

This module only READS the arrays nonlq already built in __init__
(qproj, crexp, cqfak, element_idx, natoms, nplw, _lgam); it never rebuilds them and
never imports or modifies paw.py.
"""

import numpy as np


def proj_batched(nlq, C_block):
    """
    Batched single-spinor projection. nlq is an existing paw.nonlq instance;
    C_block is (nbands, nplw) complex of plane-wave coefficients for a band block.
    Returns (nbands, nproj) where nproj = sum_i lmmax_{type(i)} -- the per-atom
    beta blocks concatenated along the channel axis, in atom order, exactly as
    nonlq.proj concatenates them.

    No conjugation is applied to C_block or the projector, matching nonlq.proj's
    plain np.sum(cptwf * crexp * qproj * iL).
    """
    C_block = np.ascontiguousarray(C_block)
    if C_block.ndim == 1:
        C_block = C_block[None, :]
    nbands = C_block.shape[0]
    assert C_block.shape[1] == nlq.nplw, \
        f"plane-wave count mismatch: C_block has {C_block.shape[1]}, nlq.nplw={nlq.nplw}"

    # qp * iL depends only on the element TYPE (not the atom), so build it once per
    # type instead of rebuilding qp*iL[:,None] for every atom. Only the per-atom
    # structure phase crexp[:,i] then varies inside the loop. The math is identical
    # to nonlq.proj; this just removes the redundant per-atom array construction.
    qp_iL = [qp * iL[:, None]                        # (lmmax_t, nplw)
             for qp, iL in zip(nlq.qproj, nlq.cqfak)]

    blocks = []
    for iatom in range(nlq.natoms):
        ntype = nlq.element_idx[iatom]
        phase = nlq.crexp[:, iatom]                 # (nplw,), complex
        # Effective projector for this atom: P_i[c,G] = (qp*iL)[c,G] * phase[G].
        P_i = qp_iL[ntype] * phase[None, :]
        # beta_block = C @ P_i.T  -> (nbands, lmmax_t). Plain bilinear sum over G,
        # no conjugation, identical to sum_G cptwf[G]*P_i[c,G].
        blocks.append(C_block @ P_i.T)

    beta = np.concatenate(blocks, axis=1)           # (nbands, nproj)
    if nlq._lgam:
        beta = beta.real.astype(C_block.dtype)
    return beta


def get_beta_njk_batched(ae, C_block):
    """
    SOC-aware batched projection mirroring aewfc.vasp_ae_wfc.get_beta_njk, but for
    a BLOCK of bands. C_block is (nbands, ncoeff) where ncoeff == 2*nplw for SOC
    (stacked spinor [up | dn]) or nplw otherwise. Uses ae._q_proj (the same nonlq
    the current code uses). Returns (nbands, nproj) where for SOC nproj is doubled
    (beta_up concatenated with beta_dn), exactly as get_beta_njk does per band.
    """
    C_block = np.ascontiguousarray(C_block)
    if C_block.ndim == 1:
        C_block = C_block[None, :]
    q = ae._q_proj
    if ae._pswfc._lsoc:
        nplw = q.nplw
        beta_up = proj_batched(q, C_block[:, :nplw])
        beta_dn = proj_batched(q, C_block[:, nplw:])
        return np.concatenate([beta_up, beta_dn], axis=1)
    else:
        return proj_batched(q, C_block)
