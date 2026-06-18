#!/usr/bin/env python3
"""
make_subsets.py
================

Given a directory containing a complete VASP calculation of an Au slab with
one or more Au adatoms and one or more NHC molecules (1,3-diisopropyl-
benzimidazol-2-ylidene; 2N + 13C + 18H per molecule) bound to those adatoms,
auto-detect the molecules and adatoms by composition + connectivity (Cartesian,
PBC-aware) and create one subdirectory per subset, each containing a complete,
self-consistent VASP input set: POSCAR, INCAR, KPOINTS, POTCAR, job.sh.

Subdirectories created inside calc_dir:
    adatom_surface/                -- all Au atoms (slab + every adatom)
    lone_adatom_<k>/               -- one per adatom
    NHC_<k>/                       -- one per NHC molecule (33 atoms)
    NHC<m>Au_complex_<k>/          -- one per complex (1 adatom + its NHCs)
                                      (m = number of NHCs in complex)

Numbering <k> is assigned by sorting the relevant entity's Cartesian centroid
lexicographically by (x, y, z).

Per-file rules in each created subdirectory:
    POSCAR   : the subset, atoms in ascending global index order, lattice
               copied, species/count line filtered to species present
               (preserving original Au-N-C-H order), Selective Dynamics +
               Direct, SD flags copied verbatim from input.
    KPOINTS  : byte-for-byte copy of input KPOINTS.
    POTCAR   : concatenation of the input POTCAR's per-species blocks for
               the species in the subset, in the same order as the subset's
               species line.
    INCAR    : copy of input INCAR with the MAGMOM line rewritten -- each
               n*X group's n is replaced by the subset atom count, X is
               preserved verbatim. Trailing comments preserved.
    job.sh   : copy of input job.sh with the SLURM job-name directive
               (`#SBATCH --job-name=`) rewritten so the job name equals the
               subdirectory name. Trailing comments preserved.

Detection algorithm (locked, validated against reference data):
    1. Validate Selective Dynamics, lattice volume >= 10 A^3, species set
       contains Au/N/C/H, and N/C/H counts divide cleanly by 2/13/18 to the
       same NHC count.
    2. Build min-image Cartesian bond graph over N+C+H atoms greedily:
       sort all candidate pairs by distance, add bond if
       dist < BOND_FACTOR * (r_a + r_b) AND both atoms below VALENCE_MAX.
    3. Connected components = NHCs. Each must be exactly 2N + 13C + 18H.
    4. Carbene C in each NHC = unique C bonded to both N atoms.
    5. For each NHC, fit backbone plane via SVD on 2N + 6 benzene-ring-C
       atoms. Bound adatom = unique Au with
          (a) min-image Au-C dist < BOND_FACTOR * (r_C + r_Au), AND
          (b) perp dist to backbone plane < AU_PLANARITY_TOL.
    6. Group: complexes = adatom + all NHCs bound to it.

All detection gates and supporting-file rewriters hard-fail on violations
with explicit diagnostics. No fallbacks, no auto-fixes.

Helpers parse_poscar, write_poscar, frac_to_cart, cart_to_frac,
unwrap_cluster, find_benzene_ring, backbone_plane_normal are lifted verbatim
from flatten_dimers_9_.py.

Usage: see the if __name__ == "__main__" block at the bottom.
"""

import os
import re
import shutil

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Tunable constants
# ═══════════════════════════════════════════════════════════════════

BOND_FACTOR = 1.3                                          # bond cutoff = factor * (r_a + r_b)
R_COV = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'Au': 1.36}      # covalent radii, Å
VALENCE_MAX = {'H': 1, 'C': 4, 'N': 3}                     # hard valence caps
AU_PLANARITY_TOL = 1.0                                   # Å, perp dist Au -> NHC backbone plane

NHC_COMPOSITION = {'N': 2, 'C': 13, 'H': 18}               # per-NHC atom counts


# ═══════════════════════════════════════════════════════════════════
# POSCAR I/O   (lifted verbatim from flatten_dimers_9_.py)
# ═══════════════════════════════════════════════════════════════════

def parse_poscar(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    comment = lines[0].rstrip("\n")
    scale = float(lines[1].strip())

    lattice = np.zeros((3, 3))
    for i in range(3):
        lattice[i] = [float(x) for x in lines[2 + i].split()]
    lattice *= scale

    species_names = lines[5].split()
    species_counts = [int(x) for x in lines[6].split()]
    natoms = sum(species_counts)

    line_idx = 7
    selective_dynamics = False
    if lines[line_idx].strip()[0].upper() == "S":
        selective_dynamics = True
        line_idx += 1

    coord_type = lines[line_idx].strip()
    line_idx += 1

    frac_coords = np.zeros((natoms, 3))
    sd_flags = []
    for i in range(natoms):
        parts = lines[line_idx + i].split()
        frac_coords[i] = [float(parts[j]) for j in range(3)]
        if selective_dynamics and len(parts) >= 6:
            sd_flags.append(parts[3:6])

    tail_start = line_idx + natoms
    tail_lines = lines[tail_start:]

    return {
        "comment": comment,
        "scale": scale,
        "lattice": lattice,
        "species_names": species_names,
        "species_counts": species_counts,
        "selective_dynamics": selective_dynamics,
        "coord_type": coord_type,
        "frac_coords": frac_coords,
        "sd_flags": sd_flags,
        "tail_lines": tail_lines,
    }


def write_poscar(filepath, data):
    with open(filepath, "w", newline="") as f:
        f.write(data["comment"] + "\n")
        f.write("   1.00000000000000\n")
        for i in range(3):
            f.write(
                f"  {data['lattice'][i, 0]:20.16f}"
                f"  {data['lattice'][i, 1]:20.16f}"
                f"  {data['lattice'][i, 2]:20.16f}\n"
            )
        f.write("   " + "   ".join(data["species_names"]) + "\n")
        f.write("   " + "   ".join(str(c) for c in data["species_counts"]) + "\n")
        if data["selective_dynamics"]:
            f.write("Selective dynamics\n")
        f.write("Direct\n")
        for i in range(len(data["frac_coords"])):
            x, y, z = data["frac_coords"][i]
            line = f"  {x:20.16f}  {y:20.16f}  {z:20.16f}"
            if data["selective_dynamics"] and i < len(data["sd_flags"]):
                line += "   " + "   ".join(data["sd_flags"][i])
            f.write(line + "\n")
        for line in data["tail_lines"]:
            f.write(line)


def frac_to_cart(frac, lattice):
    return frac @ lattice


def cart_to_frac(cart, lattice):
    return cart @ np.linalg.inv(lattice)


# ═══════════════════════════════════════════════════════════════════
# Geometry helpers   (lifted verbatim from flatten_dimers_9_.py)
# ═══════════════════════════════════════════════════════════════════

def unwrap_cluster(frac_coords, lattice, ref_idx=0):
    """Unwrap fractional coords so all atoms are near the reference atom.
    Returns Cartesian coordinates with PBC artifacts removed."""
    ref = frac_coords[ref_idx]
    delta = frac_coords - ref
    delta -= np.round(delta)
    return frac_to_cart(ref + delta, lattice)


def find_benzene_ring(c_cart):
    """Find the 6-membered carbon ring via C–C bond adjacency."""
    from itertools import combinations
    n = len(c_cart)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(c_cart[i] - c_cart[j])
            if 1.2 <= d <= 1.6:
                adj[i].add(j)
                adj[j].add(i)
    for ring in combinations(range(n), 6):
        ring_set = set(ring)
        if all(len(adj[i] & ring_set) == 2 for i in ring):
            return list(ring)
    raise RuntimeError("No 6-membered carbon ring found")


def backbone_plane_normal(n_cart, ring_c_cart):
    """Best-fit plane normal through 2 N + 6 benzene ring C atoms via SVD.
    Normal is pinned to point toward +z (vacuum side)."""
    pts = np.vstack([n_cart, ring_c_cart])
    _, _, Vt = np.linalg.svd(pts - pts.mean(axis=0))
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    return normal


# ═══════════════════════════════════════════════════════════════════
# NHC / adatom detection
# ═══════════════════════════════════════════════════════════════════

def min_image_cart_disp(frac_a, frac_b, lattice):
    """Min-image Cartesian displacement vector b - a."""
    d = frac_b - frac_a
    d -= np.round(d)
    return d @ lattice


def build_bond_graph(indices, frac, lattice, species_of):
    """Valence-aware greedy bond builder over the given atom indices.

    For every pair of atoms in `indices`, compute min-image Cartesian distance.
    Sort all pairs by distance ascending. Add a bond if
        dist < BOND_FACTOR * (r_a + r_b)
        AND both endpoints are below VALENCE_MAX[species].

    Real intramolecular bonds (~1.0-1.6 Å) get added first and saturate
    valences before longer intermolecular contacts are considered, which
    correctly rejects e.g. cross-methyl H...H contacts.

    Returns: dict {atom_idx -> set of bonded atom indices}.
    """
    pairs = []
    n = len(indices)
    for ii in range(n):
        i = indices[ii]
        for jj in range(ii + 1, n):
            j = indices[jj]
            disp = min_image_cart_disp(frac[i], frac[j], lattice)
            d = np.linalg.norm(disp)
            r_sum = R_COV[species_of[i]] + R_COV[species_of[j]]
            if d < BOND_FACTOR * r_sum:
                pairs.append((d, i, j))
    pairs.sort()

    bonds = {a: set() for a in indices}
    valence = {a: 0 for a in indices}
    for _, i, j in pairs:
        if (valence[i] < VALENCE_MAX[species_of[i]]
                and valence[j] < VALENCE_MAX[species_of[j]]):
            bonds[i].add(j)
            bonds[j].add(i)
            valence[i] += 1
            valence[j] += 1
    return bonds


def connected_components(indices, bonds):
    """BFS over the bond graph; return list of components (each a sorted
    list of atom indices)."""
    visited = set()
    components = []
    index_set = set(indices)
    for start in indices:
        if start in visited:
            continue
        comp = []
        stack = [start]
        while stack:
            a = stack.pop()
            if a in visited:
                continue
            visited.add(a)
            comp.append(a)
            for nbr in bonds[a]:
                if nbr not in visited and nbr in index_set:
                    stack.append(nbr)
        components.append(sorted(comp))
    return components


def find_carbene_c(component, bonds, species_of):
    """The unique C in `component` bonded to both N atoms via `bonds`.
    Hard-fail if 0 or >1."""
    n_atoms = [a for a in component if species_of[a] == 'N']
    c_atoms = [a for a in component if species_of[a] == 'C']
    assert len(n_atoms) == 2, "find_carbene_c expects component with 2 N"
    n0, n1 = n_atoms
    cand = [c for c in c_atoms if (n0 in bonds[c] and n1 in bonds[c])]
    if len(cand) != 1:
        comp_1based = [a + 1 for a in component]
        raise RuntimeError(
            f"NHC component (1-based atoms {comp_1based}) has "
            f"{len(cand)} carbene-C candidates (C bonded to both N atoms), "
            f"expected exactly 1. Candidates (1-based): {[c+1 for c in cand]}"
        )
    return cand[0]


def find_bound_adatom(carbene_idx, n_indices, c_indices, all_atoms,
                      au_indices, frac, lattice):
    """For one NHC, find the Au atom that is (a) within Au-C bond cutoff of
    the carbene C and (b) within AU_PLANARITY_TOL of the backbone plane.
    Hard-fail if not exactly 1 candidate.

    Returns (au_global_idx, dist_to_carbene, perp_to_plane).
    """
    nhc_frac = frac[all_atoms]
    nhc_cart_uw = unwrap_cluster(nhc_frac, lattice, ref_idx=0)
    local_of = {a: i for i, a in enumerate(all_atoms)}

    n_cart_uw = np.array([nhc_cart_uw[local_of[a]] for a in n_indices])
    c_cart_uw = np.array([nhc_cart_uw[local_of[a]] for a in c_indices])

    ring_local_in_c = find_benzene_ring(c_cart_uw)
    ring_cart_uw = c_cart_uw[ring_local_in_c]

    bn = backbone_plane_normal(n_cart_uw, ring_cart_uw)
    plane_pt = np.vstack([n_cart_uw, ring_cart_uw]).mean(axis=0)

    carbene_cart_uw = nhc_cart_uw[local_of[carbene_idx]]

    au_c_cutoff = BOND_FACTOR * (R_COV['C'] + R_COV['Au'])
    candidates = []
    for au in au_indices:
        disp = min_image_cart_disp(frac[carbene_idx], frac[au], lattice)
        au_cart_uw = carbene_cart_uw + disp
        dist = np.linalg.norm(au_cart_uw - carbene_cart_uw)
        if dist < au_c_cutoff:
            perp = abs(np.dot(au_cart_uw - plane_pt, bn))
            if perp < AU_PLANARITY_TOL:
                candidates.append((au, dist, perp))

    if len(candidates) != 1:
        diag = ", ".join(f"Au #{a+1} (d={d:.3f} Å, perp={p:.3f} Å)"
                         for a, d, p in candidates) or "(none)"
        raise RuntimeError(
            f"NHC at carbene C #{carbene_idx+1} (1-based) has "
            f"{len(candidates)} Au candidates passing both gates "
            f"(distance < {au_c_cutoff:.3f} Å, perp < {AU_PLANARITY_TOL} Å), "
            f"expected exactly 1. Candidates: {diag}"
        )
    return candidates[0]


def detect_subsystems(full):
    """Detect NHCs, adatoms, and complexes in the parsed POSCAR `full`.

    Returns a dict with:
        'nhcs'      : list of dicts {atoms, n, c, h, carbene, adatom,
                                     _au_dist, _au_perp}
        'adatoms'   : sorted list of unique bound Au global indices
        'complexes' : dict {adatom_idx -> [nhc_indices_into_'nhcs']}
        'all_au'    : list of all Au global indices
    """
    species_names = full["species_names"]
    species_counts = full["species_counts"]
    frac = full["frac_coords"]
    lattice = full["lattice"]

    if not full["selective_dynamics"]:
        raise RuntimeError("Input POSCAR must have Selective Dynamics.")

    vol = abs(np.linalg.det(lattice))
    if vol < 10.0:
        raise RuntimeError(f"Lattice volume {vol:.3f} Å³ < 10 Å³ — sanity fail.")

    for sp in ('Au', 'N', 'C', 'H'):
        if sp not in species_names:
            raise RuntimeError(f"Species {sp!r} not found in POSCAR.")

    sp_range = {}
    off = 0
    for sp, ct in zip(species_names, species_counts):
        sp_range[sp] = (off, off + ct)
        off += ct

    n_n = species_counts[species_names.index('N')]
    n_c = species_counts[species_names.index('C')]
    n_h = species_counts[species_names.index('H')]
    if (n_n % NHC_COMPOSITION['N'] != 0
            or n_c % NHC_COMPOSITION['C'] != 0
            or n_h % NHC_COMPOSITION['H'] != 0):
        raise RuntimeError(
            f"Composition counts N={n_n}, C={n_c}, H={n_h} not divisible "
            f"by per-NHC counts {NHC_COMPOSITION}."
        )
    n_nhc_expected = n_n // NHC_COMPOSITION['N']
    if (n_c // NHC_COMPOSITION['C'] != n_nhc_expected
            or n_h // NHC_COMPOSITION['H'] != n_nhc_expected):
        raise RuntimeError(
            f"Inconsistent NHC count from counts: "
            f"N implies {n_n // NHC_COMPOSITION['N']}, "
            f"C implies {n_c // NHC_COMPOSITION['C']}, "
            f"H implies {n_h // NHC_COMPOSITION['H']}."
        )

    species_of = {}
    for sp, (lo, hi) in sp_range.items():
        for a in range(lo, hi):
            species_of[a] = sp

    nch_indices = (list(range(*sp_range['N']))
                   + list(range(*sp_range['C']))
                   + list(range(*sp_range['H'])))
    bonds = build_bond_graph(nch_indices, frac, lattice, species_of)

    components = connected_components(nch_indices, bonds)
    if len(components) != n_nhc_expected:
        raise RuntimeError(
            f"Found {len(components)} connected components in N+C+H graph, "
            f"expected {n_nhc_expected}."
        )

    nhcs = []
    for comp in components:
        n_in = [a for a in comp if species_of[a] == 'N']
        c_in = [a for a in comp if species_of[a] == 'C']
        h_in = [a for a in comp if species_of[a] == 'H']
        if (len(n_in) != NHC_COMPOSITION['N']
                or len(c_in) != NHC_COMPOSITION['C']
                or len(h_in) != NHC_COMPOSITION['H']):
            raise RuntimeError(
                f"Component (1-based atoms {[a+1 for a in comp]}) has "
                f"{len(n_in)}N + {len(c_in)}C + {len(h_in)}H, "
                f"expected {NHC_COMPOSITION['N']}N + "
                f"{NHC_COMPOSITION['C']}C + {NHC_COMPOSITION['H']}H."
            )
        carbene = find_carbene_c(comp, bonds, species_of)
        nhcs.append({
            'atoms':   sorted(comp),
            'n':       sorted(n_in),
            'c':       sorted(c_in),
            'h':       sorted(h_in),
            'carbene': carbene,
            'adatom':  None,
        })

    au_indices = list(range(*sp_range['Au']))
    for nhc in nhcs:
        au, dist, perp = find_bound_adatom(
            nhc['carbene'], nhc['n'], nhc['c'], nhc['atoms'],
            au_indices, frac, lattice,
        )
        nhc['adatom'] = au
        nhc['_au_dist'] = dist
        nhc['_au_perp'] = perp

    adatom_set = sorted({nhc['adatom'] for nhc in nhcs})
    complexes = {au: [] for au in adatom_set}
    for i, nhc in enumerate(nhcs):
        complexes[nhc['adatom']].append(i)

    return {
        'nhcs':      nhcs,
        'adatoms':   adatom_set,
        'complexes': complexes,
        'all_au':    au_indices,
    }


# ═══════════════════════════════════════════════════════════════════
# Subset POSCAR writer
# ═══════════════════════════════════════════════════════════════════

def subset_species_order(atom_list, full):
    """Return the species names present in the subset, in the same order
    they appear in the full POSCAR's species line."""
    sp_range = {}
    off = 0
    for sp, ct in zip(full['species_names'], full['species_counts']):
        sp_range[sp] = (off, off + ct)
        off += ct
    order = []
    for sp in full['species_names']:
        lo, hi = sp_range[sp]
        if any(lo <= a < hi for a in atom_list):
            order.append(sp)
    return order


def write_subset_poscar(out_path, atom_list, full):
    """Write a POSCAR containing only the given atom indices (0-based).
    Atoms appear in ascending global index order. Species/count line is
    filtered to species present in the subset, preserving original order.
    SD flags copied verbatim. Line-0 (comment) is blank.
    """
    atom_list = sorted(atom_list)
    species_names = full["species_names"]
    species_counts = full["species_counts"]

    sp_range = {}
    off = 0
    for sp, ct in zip(species_names, species_counts):
        sp_range[sp] = (off, off + ct)
        off += ct

    sub_species = []
    sub_counts = []
    for sp in species_names:
        lo, hi = sp_range[sp]
        n_in = sum(1 for a in atom_list if lo <= a < hi)
        if n_in > 0:
            sub_species.append(sp)
            sub_counts.append(n_in)

    sub_frac = np.array([full["frac_coords"][a] for a in atom_list])
    sub_sd = [full["sd_flags"][a] for a in atom_list]

    sub_data = {
        "comment": "",
        "scale": 1.0,
        "lattice": full["lattice"],
        "species_names": sub_species,
        "species_counts": sub_counts,
        "selective_dynamics": True,
        "coord_type": "Direct",
        "frac_coords": sub_frac,
        "sd_flags": sub_sd,
        "tail_lines": [],
    }
    write_poscar(out_path, sub_data)


# ═══════════════════════════════════════════════════════════════════
# Supporting-file rewriters: INCAR (MAGMOM), POTCAR, job.sh
# ═══════════════════════════════════════════════════════════════════

def _is_active_magmom_line(line):
    """True iff `line` is an active (uncommented) MAGMOM = ... line."""
    stripped = line.lstrip()
    if stripped.startswith('#') or stripped.startswith('!'):
        return False
    return re.match(r'^MAGMOM\s*=', stripped, re.IGNORECASE) is not None


def _split_line_ending(s):
    """Return (body_without_newline, newline_str)."""
    if s.endswith('\r\n'):
        return s[:-2], '\r\n'
    if s.endswith('\n'):
        return s[:-1], '\n'
    if s.endswith('\r'):
        return s[:-1], '\r'
    return s, ''


def rewrite_magmom_line(line, n_subset):
    """Rewrite a single MAGMOM line: replace each n*X group's n with
    n_subset, preserve X verbatim, preserve trailing comment.

    Hard-fail if:
      - any token doesn't match 'n*X' (n positive integer)
      - groups have inconsistent n in the input
    """
    body, nl = _split_line_ending(line)

    eq_idx = body.find('=')
    if eq_idx < 0:
        raise RuntimeError(f"MAGMOM line has no '=' separator: {line!r}")
    lhs = body[:eq_idx + 1]
    rhs = body[eq_idx + 1:]

    comment_idx = -1
    for c in ('#', '!', '('):
        idx = rhs.find(c)
        if idx >= 0 and (comment_idx < 0 or idx < comment_idx):
            comment_idx = idx

    if comment_idx >= 0:
        value_part = rhs[:comment_idx]
        comment_part = rhs[comment_idx:]
    else:
        value_part = rhs
        comment_part = ''

    value_stripped = value_part.strip()
    leading_ws = value_part[:len(value_part) - len(value_part.lstrip())]
    trailing_ws = value_part[len(value_part.rstrip()):]

    if not value_stripped:
        raise RuntimeError(f"MAGMOM value is empty: {line!r}")

    tokens = value_stripped.split()
    new_tokens = []
    n_values = []
    for tok in tokens:
        m = re.match(r'^(\d+)\*(.+)$', tok)
        if not m:
            raise RuntimeError(
                f"MAGMOM token {tok!r} doesn't match 'n*X' pattern. "
                f"Full line: {line!r}"
            )
        n_values.append(int(m.group(1)))
        new_tokens.append(f"{n_subset}*{m.group(2)}")

    if len(set(n_values)) != 1:
        raise RuntimeError(
            f"MAGMOM groups in input have inconsistent n values {n_values}. "
            f"Full line: {line!r}"
        )

    new_value = ' '.join(new_tokens)
    return f"{lhs}{leading_ws}{new_value}{trailing_ws}{comment_part}{nl}"


def rewrite_incar_text(incar_text, n_subset):
    """Find the unique active MAGMOM line in `incar_text` and rewrite it.
    Hard-fail if 0 or >1 active MAGMOM lines."""
    lines = incar_text.splitlines(keepends=True)
    active = [i for i, l in enumerate(lines) if _is_active_magmom_line(l)]
    if len(active) == 0:
        raise RuntimeError("No active MAGMOM line found in INCAR.")
    if len(active) > 1:
        raise RuntimeError(
            f"Multiple active MAGMOM lines found at lines "
            f"{[i+1 for i in active]} (1-based)."
        )
    idx = active[0]
    lines[idx] = rewrite_magmom_line(lines[idx], n_subset)
    return ''.join(lines)


def parse_potcar(potcar_text):
    """Parse POTCAR into a dict {element_symbol: block_text}, preserving
    block bytes verbatim. Block boundaries: lines containing 'End of Dataset'.
    Element symbol is the second whitespace-separated token of the block's
    first non-blank line (e.g. 'PAW_PBE Au_d 06Sep2000' -> 'Au').

    Hard-fail on duplicate element blocks or trailing non-whitespace content.
    """
    lines = potcar_text.splitlines(keepends=True)
    blocks = {}
    block_start = 0
    for i, line in enumerate(lines):
        if 'End of Dataset' in line:
            block_lines = lines[block_start:i + 1]
            block_text = ''.join(block_lines)
            first_nonblank = next((l for l in block_lines if l.strip()), '')
            tokens = first_nonblank.split()
            if len(tokens) < 2:
                raise RuntimeError(
                    f"POTCAR block first line not parseable: {first_nonblank!r}"
                )
            species_label = tokens[1]
            element = species_label.split('_')[0]
            if element in blocks:
                raise RuntimeError(
                    f"POTCAR contains duplicate block for element {element!r}."
                )
            blocks[element] = block_text
            block_start = i + 1

    if not blocks:
        raise RuntimeError("POTCAR contains no 'End of Dataset' markers.")

    if block_start < len(lines):
        remaining = ''.join(lines[block_start:])
        if remaining.strip():
            raise RuntimeError(
                "POTCAR has trailing non-whitespace content after last block."
            )

    return blocks


def build_subset_potcar(blocks, species_order):
    """Concatenate POTCAR blocks for `species_order` (in that exact order).
    Hard-fail if any required species is missing."""
    parts = []
    for sp in species_order:
        if sp not in blocks:
            raise RuntimeError(
                f"POTCAR is missing block for species {sp!r}. "
                f"Available: {sorted(blocks.keys())}"
            )
        parts.append(blocks[sp])
    return ''.join(parts)


# Match: optional leading whitespace, '#', optional whitespace, 'SBATCH',
#        whitespace, '--job-name=', non-whitespace value, rest.
_JOBNAME_RE = re.compile(r'^(\s*#\s*SBATCH\s+--job-name=)(\S+)(.*)$')


def rewrite_jobsh_jobname(jobsh_text, new_name):
    """Find the unique '#SBATCH --job-name=' line in `jobsh_text` and
    rewrite the job name to `new_name`. Trailing content (e.g. inline
    comments) is preserved. Hard-fail if 0 or >1 such lines."""
    lines = jobsh_text.splitlines(keepends=True)
    matches = []
    for i, line in enumerate(lines):
        body, _ = _split_line_ending(line)
        if _JOBNAME_RE.match(body):
            matches.append(i)

    if len(matches) == 0:
        raise RuntimeError("No '#SBATCH --job-name=' line found in job.sh.")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple '#SBATCH --job-name=' lines found at lines "
            f"{[i+1 for i in matches]} (1-based)."
        )

    i = matches[0]
    body, nl = _split_line_ending(lines[i])
    m = _JOBNAME_RE.match(body)
    new_body = m.group(1) + new_name + m.group(3)
    lines[i] = new_body + nl
    return ''.join(lines)


# ═══════════════════════════════════════════════════════════════════
# Top-level orchestration
# ═══════════════════════════════════════════════════════════════════

def lex_key(v, ndp=6):
    """Lex sort key for a Cartesian vector at given decimal precision
    (1e-6 Å tie-break tolerance)."""
    return (round(float(v[0]), ndp),
            round(float(v[1]), ndp),
            round(float(v[2]), ndp))


REQUIRED_FILES = ('POSCAR', 'INCAR', 'KPOINTS', 'POTCAR', 'job.sh')


def run(calc_dir, overwrite):
    """Detect subsystems in calc_dir/POSCAR and write one subdirectory per
    subset, each containing POSCAR, INCAR, KPOINTS, POTCAR, job.sh.

    Parameters
    ----------
    calc_dir : str
        Path to the VASP calculation directory containing the input files.
    overwrite : bool
        If True, existing subdirectories with the target names are removed
        before writing. If False, hard-fail if any target subdirectory exists.
    """
    if not os.path.isdir(calc_dir):
        raise RuntimeError(f"calc_dir is not a directory: {calc_dir}")

    paths = {fn: os.path.join(calc_dir, fn) for fn in REQUIRED_FILES}
    for fn, p in paths.items():
        if not os.path.isfile(p):
            raise RuntimeError(f"Required input file missing: {p}")

    # Parse POSCAR and detect subsystems
    full = parse_poscar(paths['POSCAR'])
    det = detect_subsystems(full)

    nhcs = det['nhcs']
    adatoms = det['adatoms']
    complexes = det['complexes']
    all_au = det['all_au']

    frac = full["frac_coords"]
    lattice = full["lattice"]
    cart = frac_to_cart(frac, lattice)

    # Centroids and sort orders
    nhc_centroids = []
    for nhc in nhcs:
        nhc_cart_uw = unwrap_cluster(frac[nhc['atoms']], lattice, ref_idx=0)
        nhc_centroids.append(nhc_cart_uw.mean(axis=0))

    nhc_order     = sorted(range(len(nhcs)),
                           key=lambda i: lex_key(nhc_centroids[i]))
    adatom_order  = sorted(adatoms, key=lambda au: lex_key(cart[au]))
    complex_order = sorted(adatoms, key=lambda au: lex_key(cart[au]))

    # Read auxiliary inputs once
    with open(paths['INCAR'], 'r') as f:
        incar_text = f.read()
    with open(paths['POTCAR'], 'r') as f:
        potcar_text = f.read()
    with open(paths['job.sh'], 'r') as f:
        jobsh_text = f.read()
    potcar_blocks = parse_potcar(potcar_text)

    # Build the list of subsets to write
    subsets = []
    subsets.append(('adatom_surface', list(all_au)))
    for k, au in enumerate(adatom_order, 1):
        subsets.append((f'lone_adatom_{k}', [au]))

    # NHC naming:
    #   single-complex case  -> NHC_<k>            (k = global NHC sort index)
    #   multi-complex case   -> NHC_c<X>m<Y>       where X = complex index
    #                                              (matches NHC<m>Au_complex_<X>),
    #                                              Y = within-complex molecule index
    #                                              (assigned in global NHC sort order
    #                                              restricted to that complex).
    if len(complex_order) > 1:
        complex_index_of_adatom = {au: idx + 1
                                   for idx, au in enumerate(complex_order)}
        within_complex_count = {au: 0 for au in complex_order}
        for nhc_i in nhc_order:
            au = nhcs[nhc_i]['adatom']
            within_complex_count[au] += 1
            x = complex_index_of_adatom[au]
            y = within_complex_count[au]
            subsets.append((f'NHC_c{x}m{y}', list(nhcs[nhc_i]['atoms'])))
    else:
        for k, nhc_i in enumerate(nhc_order, 1):
            subsets.append((f'NHC_{k}', list(nhcs[nhc_i]['atoms'])))
    for k, au in enumerate(complex_order, 1):
        nhc_idxs = complexes[au]
        m = len(nhc_idxs)
        atoms = [au]
        for nhc_i in nhc_idxs:
            atoms.extend(nhcs[nhc_i]['atoms'])
        subsets.append((f'NHC{m}Au_complex_{k}', atoms))

    # Pre-flight: when overwrite=False, hard-fail if any target exists,
    # before writing anything (so we either commit all or none).
    if not overwrite:
        existing = [name for name, _ in subsets
                    if os.path.exists(os.path.join(calc_dir, name))]
        if existing:
            raise RuntimeError(
                f"Subdirectories already exist (overwrite=False): {existing}"
            )

    written = []
    for subfolder_name, atom_list in subsets:
        subfolder = os.path.join(calc_dir, subfolder_name)
        if os.path.exists(subfolder):
            shutil.rmtree(subfolder)
        os.makedirs(subfolder)

        # POSCAR
        write_subset_poscar(os.path.join(subfolder, 'POSCAR'),
                            atom_list, full)

        # KPOINTS — byte-for-byte copy
        shutil.copyfile(paths['KPOINTS'], os.path.join(subfolder, 'KPOINTS'))

        # POTCAR — filtered, in subset species order
        sp_order = subset_species_order(atom_list, full)
        new_potcar = build_subset_potcar(potcar_blocks, sp_order)
        with open(os.path.join(subfolder, 'POTCAR'), 'w', newline='') as f:
            f.write(new_potcar)

        # INCAR — MAGMOM rewritten to match subset atom count
        new_incar = rewrite_incar_text(incar_text, len(atom_list))
        with open(os.path.join(subfolder, 'INCAR'), 'w', newline='') as f:
            f.write(new_incar)

        # job.sh — job name set to subfolder name
        new_jobsh = rewrite_jobsh_jobname(jobsh_text, subfolder_name)
        with open(os.path.join(subfolder, 'job.sh'), 'w', newline='') as f:
            f.write(new_jobsh)

        written.append((subfolder_name, len(atom_list), sp_order))

    # Summary
    print(f"Input directory: {calc_dir}")
    print(f"Detected: {len(nhcs)} NHC(s), {len(adatoms)} adatom(s), "
          f"{len(complexes)} complex(es).")
    for i in nhc_order:
        nhc = nhcs[i]
        print(f"  NHC: carbene C #{nhc['carbene']+1}, "
              f"adatom Au #{nhc['adatom']+1}, "
              f"d(Au-C)={nhc['_au_dist']:.3f} Å, "
              f"perp(Au→backbone)={nhc['_au_perp']:.3f} Å")
    print(f"\nWrote {len(written)} subdirectories:")
    for name, n_atoms, sp_order in written:
        print(f"  {name}/  ({n_atoms} atoms; species: {' '.join(sp_order)})")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    calc_dir  = 'C:/Users/Benjamin Kafin/Documents/VASP/SAM/NHC2Au'
    overwrite = True

    run(calc_dir, overwrite)