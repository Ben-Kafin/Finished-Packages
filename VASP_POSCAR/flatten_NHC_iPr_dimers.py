#!/usr/bin/env python3
"""
Flatten NHC–Au–NHC dimers on Au(111):

  Per-NHC operations (Op 1):
    a) Backbone plane from 2 N + 6 benzene ring C, pinned +z.
    b) Planarize carbene C into backbone plane (ncn_planar_angle).
    c) Flap each iPr wingtip into backbone plane (wingtip_flap).
    d) Twist each iPr wingtip H toward Au in plane (wingtip_twist).
    e) Tilt entire NHC about carbene C (tilt_angle).
  Per-dimer operations:
    Op 2: Adjust bend via backbone directions in vertical plane (bend_angle).
    Op 3: Restore original z-gap to slab top.

  All angle inputs: 180° = ideal planar / symmetric.
  Offsets from 180°: >180° moves toward +z (vacuum), <180° toward -z (slab).
  The wingtip twist base position (H toward Au) is selected automatically;
  its offset also follows the +z / -z convention.

Species block ordering (sequential within each type):
    Per NHC:  2 N, 13 C, 18 H  (33 atoms)
    Per dimer: 2 NHCs = 4 N, 26 C, 36 H  (66 atoms)
    4 dimers total → 16 N, 104 C, 144 H
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# POSCAR I/O
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
    with open(filepath, "w") as f:
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
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════

def unwrap_cluster(frac_coords, lattice, ref_idx=0):
    """Unwrap fractional coords so all atoms are near the reference atom.
    Returns Cartesian coordinates with PBC artifacts removed."""
    ref = frac_coords[ref_idx]
    delta = frac_coords - ref
    delta -= np.round(delta)
    return frac_to_cart(ref + delta, lattice)


def rodrigues(points, axis, angle_rad, pivot):
    """Rotate points about axis through pivot by angle_rad (radians).
    Accepts (N,3) array or single (3,) vector."""
    k = axis / np.linalg.norm(axis)
    single = (points.ndim == 1)
    pts = np.atleast_2d(points)
    p = pts - pivot
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    p_rot = p * c + np.cross(k, p) * s + k * (p @ k)[:, None] * (1 - c)
    result = p_rot + pivot
    return result[0] if single else result


def find_carbene_c(c_cart, au_cart):
    """Index of the C atom closest to Au (= carbene carbon)."""
    return np.argmin(np.linalg.norm(c_cart - au_cart, axis=1))


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


def angle_vec_to_plane(vec, normal):
    """Signed angle (degrees) between a vector and a plane defined by
    its normal. Positive = vec has a component along +normal."""
    v_hat = vec / np.linalg.norm(vec)
    n_hat = normal / np.linalg.norm(normal)
    return np.degrees(np.arcsin(np.clip(np.dot(v_hat, n_hat), -1, 1)))


def signed_bend_angle(bd1_proj, bd2_proj, plane_normal):
    """Angle (degrees) between two projected backbone directions.

    < 180 = backbones slope downward (V shape, bent toward surface).
    = 180 = backbones flat (anti-parallel, constant z).
    > 180 = backbones slope upward past flat (inverted V).

    Uses average z-component of backbone directions to determine
    which side of 180° the angle is on: negative z = tips below
    carbene level = bent toward surface = < 180°.
    """
    cos_val = np.dot(bd1_proj, bd2_proj) / (
        np.linalg.norm(bd1_proj) * np.linalg.norm(bd2_proj))
    unsigned = np.degrees(np.arccos(np.clip(cos_val, -1, 1)))
    avg_z = (bd1_proj[2] + bd2_proj[2]) / 2.0
    if avg_z > 0:
        return 360.0 - unsigned
    return unsigned


# ═══════════════════════════════════════════════════════════════════
# Rotation-to-plane helpers
# ═══════════════════════════════════════════════════════════════════

def angle_to_plane(target_pos, pivot, axis, plane_point, plane_normal):
    """Smallest rotation angle (radians) around *axis* through *pivot*
    that places *target_pos* in the plane (plane_point, plane_normal).

    Handles the general case where the pivot may not lie in the plane.
    """
    axis_hat = axis / np.linalg.norm(axis)
    v = target_pos - pivot
    v_par = np.dot(v, axis_hat) * axis_hat
    v_perp = v - v_par

    c_val = np.dot(pivot + v_par - plane_point, plane_normal)
    a = np.dot(v_perp, plane_normal)
    b = np.dot(np.cross(axis_hat, v_perp), plane_normal)

    R = np.sqrt(a**2 + b**2)
    if R < 1e-12:
        return 0.0

    phi = np.arctan2(b, a)

    if abs(c_val) > R * (1 + 1e-8):
        # Cannot reach plane; minimize out-of-plane residual
        return (phi + np.pi) if c_val > 0 else phi

    delta = np.arccos(np.clip(-c_val / R, -1, 1))
    theta1 = phi + delta
    theta2 = phi - delta
    # Normalize to [-pi, pi]
    theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
    theta2 = (theta2 + np.pi) % (2 * np.pi) - np.pi
    return theta1 if abs(theta1) <= abs(theta2) else theta2


def angle_to_plane_toward(target_pos, pivot, axis, plane_point,
                          plane_normal, toward_pos):
    """Like angle_to_plane, but when two solutions exist, pick the one
    that puts the rotated target_pos on the side of *toward_pos*
    relative to the pivot (projected into the backbone plane)."""
    axis_hat = axis / np.linalg.norm(axis)
    v = target_pos - pivot
    v_par = np.dot(v, axis_hat) * axis_hat
    v_perp = v - v_par
    cross_term = np.cross(axis_hat, v_perp)

    c_val = np.dot(pivot + v_par - plane_point, plane_normal)
    a = np.dot(v_perp, plane_normal)
    b = np.dot(cross_term, plane_normal)

    R = np.sqrt(a**2 + b**2)
    if R < 1e-12:
        return 0.0

    phi = np.arctan2(b, a)

    if abs(c_val) > R * (1 + 1e-8):
        return (phi + np.pi) if c_val > 0 else phi

    delta = np.arccos(np.clip(-c_val / R, -1, 1))
    theta1 = phi + delta
    theta2 = phi - delta

    def pos_at(theta):
        return (pivot + v_par
                + np.cos(theta) * v_perp
                + np.sin(theta) * cross_term)

    toward_dir = toward_pos - pivot
    toward_dir = toward_dir - np.dot(toward_dir, plane_normal) * plane_normal

    d1 = np.dot(pos_at(theta1) - pivot, toward_dir)
    d2 = np.dot(pos_at(theta2) - pivot, toward_dir)

    return theta1 if d1 >= d2 else theta2


def planar_offset_z(target_pos, pivot, axis, plane_point, bn, offset_deg):
    """Rotate target to backbone plane, then offset by offset_deg.

    Convention:  offset > 0  moves target toward +bn (+z / vacuum).
                 offset < 0  moves target toward -bn (-z / slab).
    """
    theta_flat = angle_to_plane(target_pos, pivot, axis, plane_point, bn)
    if abs(offset_deg) < 1e-10:
        return theta_flat

    axis_hat = axis / np.linalg.norm(axis)
    v = target_pos - pivot
    v_perp = v - np.dot(v, axis_hat) * axis_hat
    a = np.dot(v_perp, bn)
    b = np.dot(np.cross(axis_hat, v_perp), bn)

    # d(oop)/dtheta at theta_flat: positive means +theta moves toward +bn
    d_oop = -a * np.sin(theta_flat) + b * np.cos(theta_flat)

    offset_rad = np.radians(offset_deg)
    if d_oop < 0:
        offset_rad = -offset_rad
    return theta_flat + offset_rad


def planar_offset_toward(target_pos, pivot, axis, plane_point, bn,
                         toward_pos, offset_deg):
    """Rotate target to backbone plane (H pointing toward toward_pos),
    then offset by offset_deg.

    The toward_pos logic selects which of the two in-plane solutions
    to use as the base.  The offset convention is the same as
    planar_offset_z:
        offset > 0  moves target toward +bn (+z / vacuum).
        offset < 0  moves target toward -bn (-z / slab).
    """
    theta_flat = angle_to_plane_toward(target_pos, pivot, axis,
                                       plane_point, bn, toward_pos)
    if abs(offset_deg) < 1e-10:
        return theta_flat

    axis_hat = axis / np.linalg.norm(axis)
    v = target_pos - pivot
    v_perp = v - np.dot(v, axis_hat) * axis_hat
    a = np.dot(v_perp, bn)
    b = np.dot(np.cross(axis_hat, v_perp), bn)

    # d(oop)/dtheta at theta_flat: positive means +theta → +z
    d_oop = -a * np.sin(theta_flat) + b * np.cos(theta_flat)

    offset_rad = np.radians(offset_deg)
    if d_oop < 0:
        offset_rad = -offset_rad
    return theta_flat + offset_rad


def measure_planar_angle_z(target_pos, pivot, axis, plane_point, bn):
    """Measure the equivalent user-convention angle for a point
    relative to a backbone plane.  180 = in plane, >180 = toward +z.

    Inverts the planar_offset_z convention so the returned value
    matches the input parameter that would reproduce this position.
    """
    theta_flat = angle_to_plane(target_pos, pivot, axis, plane_point, bn)

    axis_hat = axis / np.linalg.norm(axis)
    v = target_pos - pivot
    v_perp = v - np.dot(v, axis_hat) * axis_hat
    a = np.dot(v_perp, bn)
    b = np.dot(np.cross(axis_hat, v_perp), bn)
    d_oop = -a * np.sin(theta_flat) + b * np.cos(theta_flat)

    if d_oop >= 0:
        return 180.0 - np.degrees(theta_flat)
    else:
        return 180.0 + np.degrees(theta_flat)


# ═══════════════════════════════════════════════════════════════════
# iPr group identification
# ═══════════════════════════════════════════════════════════════════

def identify_ipr_groups(c_cart, n_cart, h_cart, carb_idx, ring_idx):
    """Identify the two iPr groups (one per N atom) in an NHC.

    nhc_cart layout: [N0, N1, C0..C12, H0..H17]
      c_cart = nhc_cart[2:15]    -> nhc index = 2 + c_local
      h_cart = nhc_cart[15:33]   -> nhc index = 15 + h_local

    Returns list of 2 dicts with keys:
        n_local      - index into n_cart (0 or 1)
        cb_local     - index into c_cart for Cb (ring C bonded to N)
        cw_local     - index into c_cart for Cw (iPr CH)
        hw_local     - index into h_cart for H bonded to Cw
        ch3_c_local  - [2] indices into c_cart for CH3 carbons
        ch3_h_local  - [[3],[3]] indices into h_cart per CH3
        flap_nhc     - nhc_cart indices for flap group (Cw + all subs)
        twist_nhc    - nhc_cart indices for twist group (subs only)
    """
    ring_set = set(ring_idx)
    groups = []

    for ni in range(2):
        n_pos = n_cart[ni]

        # C atoms bonded to this N (distance < 1.6 A)
        bonded_c = [ci for ci in range(len(c_cart))
                    if np.linalg.norm(c_cart[ci] - n_pos) < 1.6]

        cb, cw = None, None
        for ci in bonded_c:
            if ci == carb_idx:
                continue
            if ci in ring_set:
                cb = ci
            else:
                cw = ci

        assert cb is not None and cw is not None, (
            f"Failed to identify Cb/Cw for N{ni}")
        cw_pos = c_cart[cw]

        # H bonded to Cw (distance < 1.15 A)
        hw = None
        for hi in range(len(h_cart)):
            if np.linalg.norm(h_cart[hi] - cw_pos) < 1.15:
                hw = hi
                break
        assert hw is not None, f"No H found bonded to Cw for N{ni}"

        # CH3 C bonded to Cw
        ch3_c = [ci for ci in range(len(c_cart))
                 if ci not in ring_set and ci != cw and ci != carb_idx
                 and np.linalg.norm(c_cart[ci] - cw_pos) < 1.65]
        assert len(ch3_c) == 2, (
            f"Expected 2 CH3 carbons for N{ni}, found {len(ch3_c)}")

        # H atoms per CH3
        used_h = {hw}
        ch3_h = []
        for ci in ch3_c:
            hs = []
            for hi in range(len(h_cart)):
                if hi in used_h:
                    continue
                if np.linalg.norm(h_cart[hi] - c_cart[ci]) < 1.15:
                    hs.append(hi)
            assert len(hs) == 3, (
                f"Expected 3 H on CH3 C[{ci}], found {len(hs)}")
            ch3_h.append(hs)
            used_h.update(hs)

        # Build nhc_cart index lists
        flap_nhc = [2 + cw, 15 + hw]
        twist_nhc = [15 + hw]
        for i, ci in enumerate(ch3_c):
            flap_nhc.append(2 + ci)
            twist_nhc.append(2 + ci)
            for hi in ch3_h[i]:
                flap_nhc.append(15 + hi)
                twist_nhc.append(15 + hi)

        groups.append({
            "n_local": ni,
            "cb_local": cb,
            "cw_local": cw,
            "hw_local": hw,
            "ch3_c_local": ch3_c,
            "ch3_h_local": ch3_h,
            "flap_nhc": flap_nhc,
            "twist_nhc": twist_nhc,
        })

    return groups


# ═══════════════════════════════════════════════════════════════════
# Main logic
# ═══════════════════════════════════════════════════════════════════

def run(poscar_in, poscar_out, tilt_angle, bend_angle,
        ncn_planar_angle, wingtip_flap, wingtip_twist,
        sd_enable, sd_z_height):
    """
    Parameters
    ----------
    poscar_in : str
        Path to input POSCAR.
    poscar_out : str
        Path to output POSCAR.
    tilt_angle : float
        180 = Au in backbone plane (flat). <180 = tighter bend.
    bend_angle : float
        180 = both NHC backbones flat (constant z). <180 = more bent.
        Measured via backbone directions in the vertical bisecting plane.
    ncn_planar_angle : float
        180 = carbene C exactly in backbone plane.
    wingtip_flap : float
        180 = N-Cw bond in backbone plane.
    wingtip_twist : float
        180 = Cw-H bond in backbone plane, H pointing toward Au.
    sd_enable : bool
        If True, write selective dynamics flags based on sd_z_height.
    sd_z_height : float
        Cartesian z threshold (A). Atoms with z > sd_z_height get
        T T T; atoms at or below get F F F.
    """
    data = parse_poscar(poscar_in)
    lattice = data["lattice"]
    frac = data["frac_coords"].copy()
    sc = data["species_counts"]
    n_au = sc[0]

    # -- Cartesian coordinates --
    cart = frac_to_cart(frac, lattice)

    # -- Auto-detect dimer count from atom counts --
    n_n, n_c, n_h = sc[1], sc[2], sc[3]
    assert n_n % 4 == 0 and n_c % 26 == 0 and n_h % 36 == 0, (
        f"Atom counts N={n_n}, C={n_c}, H={n_h} not divisible by "
        f"per-dimer counts (4, 26, 36)")
    n_dimers = n_n // 4
    assert n_c // 26 == n_dimers and n_h // 36 == n_dimers, (
        f"Inconsistent dimer count: N->{n_dimers}, C->{n_c//26}, H->{n_h//36}")

    # -- Identify Au adatoms (n_dimers highest-z) --
    au_z = cart[:n_au, 2]
    adatom_local = sorted(np.argsort(au_z)[::-1][:n_dimers])
    adatom_cart = cart[adatom_local]
    z_slab_top = np.max(np.delete(au_z, adatom_local))

    print(f"Species: {dict(zip(data['species_names'], sc))}")
    print(f"Detected {n_dimers} dimer(s)")
    print(f"Slab top z: {z_slab_top:.3f} A")
    for i, idx in enumerate(adatom_local):
        print(f"  Adatom {i}: Au[{idx}], z = {adatom_cart[i, 2]:.3f} A")

    # -- Index offsets --
    n_off = n_au
    c_off = n_au + sc[1]
    h_off = c_off + sc[2]

    new_cart = cart.copy()

    for k in range(n_dimers):
        print(f"\n{'='*60}")
        print(f"DIMER {k}")
        print(f"{'='*60}")

        # -- Build per-NHC global index lists --
        nhcs = []
        for m in range(2):
            nhcs.append({
                "n_g": list(range(n_off + 4*k + 2*m,
                                  n_off + 4*k + 2*(m+1))),
                "c_g": list(range(c_off + 26*k + 13*m,
                                  c_off + 26*k + 13*(m+1))),
                "h_g": list(range(h_off + 36*k + 18*m,
                                  h_off + 36*k + 18*(m+1))),
            })
            nhcs[m]["all_g"] = (nhcs[m]["n_g"]
                                + nhcs[m]["c_g"]
                                + nhcs[m]["h_g"])

        dimer_mol_g = nhcs[0]["all_g"] + nhcs[1]["all_g"]

        # -- Assign adatom by proximity to dimer centroid --
        dimer_frac = frac[dimer_mol_g]
        dimer_cart_uw = unwrap_cluster(dimer_frac, lattice, ref_idx=0)
        centroid_xy = dimer_cart_uw[:, :2].mean(axis=0)
        dists = np.linalg.norm(adatom_cart[:, :2] - centroid_xy, axis=1)
        ak = np.argmin(dists)
        au_pos = adatom_cart[ak].copy()
        au_global = adatom_local[ak]
        print(f"  Adatom {ak} (Au[{au_global}]) at "
              f"({au_pos[0]:.3f}, {au_pos[1]:.3f}, {au_pos[2]:.3f})")

        # -- Record original min gap --
        all_dimer_g = dimer_mol_g + [au_global]
        min_z_before = min(cart[i, 2] for i in all_dimer_g)
        min_gap = min_z_before - z_slab_top
        print(f"  Original min gap to slab: {min_gap:.3f} A")

        # -- Per-NHC operations --
        internal_target = tilt_angle - 180.0
        carbene_positions = []
        backbone_directions = []

        for m in range(2):
            nd = nhcs[m]

            # Unwrap this NHC relative to its first N atom
            nhc_frac = frac[nd["all_g"]]
            nhc_cart = unwrap_cluster(nhc_frac, lattice, ref_idx=0)

            # Subsets (slices into nhc_cart)
            n_cart = nhc_cart[0:2]
            c_cart = nhc_cart[2:15]
            h_cart = nhc_cart[15:33]

            # ---- Step 1: find backbone ----
            carb_idx = find_carbene_c(c_cart, au_pos)
            ring_idx = find_benzene_ring(c_cart)
            bn = backbone_plane_normal(n_cart, c_cart[ring_idx])
            plane_pt = n_cart.mean(axis=0)

            print(f"\n  NHC {m}:")
            print(f"    Benzene ring: {len(ring_idx)} C atoms")

            # ---- Step 2: carbene planarization ----
            old_carb = c_cart[carb_idx].copy()
            nn_axis = n_cart[1] - n_cart[0]
            carb_before = measure_planar_angle_z(old_carb, n_cart[0],
                                                 nn_axis, plane_pt, bn)
            theta_carb = planar_offset_z(old_carb, n_cart[0], nn_axis,
                                         plane_pt, bn,
                                         ncn_planar_angle - 180.0)
            nhc_cart[2 + carb_idx] = rodrigues(old_carb, nn_axis,
                                               theta_carb, n_cart[0])
            c_cart = nhc_cart[2:15]

            carb_after = measure_planar_angle_z(c_cart[carb_idx],
                                                n_cart[0], nn_axis,
                                                plane_pt, bn)
            print(f"    Carbene planar: {carb_before:.2f}"
                  f" -> {carb_after:.2f} deg"
                  f"  (target {ncn_planar_angle:.1f})")

            # ---- Step 3 & 4: wingtip alignment ----
            ipr_groups = identify_ipr_groups(c_cart, n_cart, h_cart,
                                            carb_idx, ring_idx)
            for gi, grp in enumerate(ipr_groups):
                n_pos = n_cart[grp["n_local"]]
                cb_pos = c_cart[grp["cb_local"]]
                cw_pos = c_cart[grp["cw_local"]]

                # Record before-flap in user degrees
                flap_axis = n_pos - cb_pos
                flap_before = measure_planar_angle_z(
                    cw_pos, n_pos, flap_axis, plane_pt, bn)
                twist_hw_before = measure_planar_angle_z(
                    h_cart[grp["hw_local"]], cw_pos,
                    cw_pos - n_pos, plane_pt, bn)

                # ---- Step 3: flap around Cb->N axis ----
                theta_flap = planar_offset_z(cw_pos, n_pos, flap_axis,
                                             plane_pt, bn,
                                             wingtip_flap - 180.0)

                flap_idx = np.array(grp["flap_nhc"])
                nhc_cart[flap_idx] = rodrigues(nhc_cart[flap_idx],
                                               flap_axis, theta_flap,
                                               n_pos)
                c_cart = nhc_cart[2:15]
                h_cart = nhc_cart[15:33]

                flap_after = measure_planar_angle_z(
                    c_cart[grp["cw_local"]], n_pos, flap_axis,
                    plane_pt, bn)

                # ---- Step 4: twist around N->Cw axis ----
                cw_pos = c_cart[grp["cw_local"]]
                twist_axis = cw_pos - n_pos
                hw_pos = h_cart[grp["hw_local"]]
                theta_twist = planar_offset_toward(
                    hw_pos, cw_pos, twist_axis, plane_pt, bn,
                    au_pos, wingtip_twist - 180.0)

                twist_idx = np.array(grp["twist_nhc"])
                nhc_cart[twist_idx] = rodrigues(nhc_cart[twist_idx],
                                                twist_axis, theta_twist,
                                                cw_pos)
                c_cart = nhc_cart[2:15]
                h_cart = nhc_cart[15:33]

                twist_after = measure_planar_angle_z(
                    h_cart[grp["hw_local"]], cw_pos, twist_axis,
                    plane_pt, bn)
                print(f"    Wingtip {gi} (N{grp['n_local']}): "
                      f"flap {flap_before:.2f} -> {flap_after:.2f} "
                      f"(target {wingtip_flap:.1f}), "
                      f"twist {twist_hw_before:.2f} -> {twist_after:.2f} "
                      f"(target {wingtip_twist:.1f}) deg")

            # ---- Step 5: tilt ----
            carb_pos = c_cart[carb_idx]
            v_au = au_pos - carb_pos
            current_tilt = angle_vec_to_plane(v_au, bn)

            print(f"    d(C-Au) = {np.linalg.norm(v_au):.3f} A")
            print(f"    Current tilt: {current_tilt + 180.0:.2f} deg")
            print(f"    Target tilt:  {tilt_angle:.1f} deg")

            delta_tilt = current_tilt - internal_target
            if abs(delta_tilt) > 0.01:
                rot_axis = np.cross(v_au, bn)
                rot_axis_n = np.linalg.norm(rot_axis)
                if rot_axis_n < 1e-10:
                    print("    WARNING: Au-C and normal parallel, "
                          "cannot define tilt axis. Skipping.")
                else:
                    rot_axis = rot_axis / rot_axis_n
                    nhc_cart = rodrigues(nhc_cart, rot_axis,
                                        np.radians(delta_tilt),
                                        carb_pos)

                    # Verify
                    n_new = nhc_cart[0:2]
                    c_new = nhc_cart[2:15]
                    bn_new = backbone_plane_normal(n_new,
                                                   c_new[ring_idx])
                    new_tilt = angle_vec_to_plane(
                        au_pos - c_new[carb_idx], bn_new)
                    print(f"    New tilt: {new_tilt + 180.0:.2f} deg")
            else:
                print("    Already at target -- skipping tilt.")

            # ---- Store results for bend calculation ----
            new_cart[nd["all_g"]] = nhc_cart
            carb_final = nhc_cart[2 + carb_idx]
            ring_centroid = nhc_cart[
                2 + np.array(ring_idx)].mean(axis=0)
            carbene_positions.append(carb_final)
            backbone_directions.append(ring_centroid - carb_final)
            nhcs[m]["carb_idx_in_c"] = carb_idx
            nhcs[m]["ring_idx"] = ring_idx

        # -- Op 2: adjust bend in the vertical plane --
        c1 = carbene_positions[0]
        c2 = carbene_positions[1]
        h_dir = (c1 - c2).copy()
        h_dir[2] = 0.0
        h_dir = h_dir / np.linalg.norm(h_dir)
        z_hat = np.array([0.0, 0.0, 1.0])
        plane_normal = np.cross(z_hat, h_dir)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        bd1 = backbone_directions[0]
        bd2 = backbone_directions[1]
        bd1_proj = bd1 - np.dot(bd1, plane_normal) * plane_normal
        bd2_proj = bd2 - np.dot(bd2, plane_normal) * plane_normal

        current_bend = signed_bend_angle(bd1_proj, bd2_proj, plane_normal)
        print(f"\n  Current bend (backbone dirs): {current_bend:.2f} deg")
        print(f"  Target bend:  {bend_angle:.1f} deg")

        delta_bend = bend_angle - current_bend
        if abs(delta_bend) > 0.01:
            half_delta = np.radians(delta_bend / 2.0)
            rot_axis = np.cross(bd1_proj, bd2_proj)
            rot_axis_n = np.linalg.norm(rot_axis)
            if rot_axis_n < 1e-10:
                print("  WARNING: backbone directions parallel, "
                      "skipping bend.")
            else:
                rot_axis = rot_axis / rot_axis_n

                # Empirically determine which rotation direction
                # opens the angle: try a tiny rotation on the backbone
                # directions and check if the measured angle increases.
                eps = 1e-4
                t1 = rodrigues(bd1_proj, rot_axis, -eps, np.zeros(3))
                t2 = rodrigues(bd2_proj, rot_axis, +eps, np.zeros(3))
                test_bend = signed_bend_angle(t1, t2, plane_normal)
                if test_bend < current_bend:
                    # Positive eps closes the angle, so our sign
                    # convention is inverted — flip half_delta.
                    half_delta = -half_delta

                signs = [-1.0, +1.0]
                for m in range(2):
                    nd = nhcs[m]
                    nhc_cart = new_cart[nd["all_g"]]
                    nhc_cart = rodrigues(nhc_cart, rot_axis,
                                        signs[m] * half_delta, au_pos)
                    new_cart[nd["all_g"]] = nhc_cart

                # Verify
                ci0 = nhcs[0]["carb_idx_in_c"]
                ci1 = nhcs[1]["carb_idx_in_c"]
                ri0 = nhcs[0]["ring_idx"]
                ri1 = nhcs[1]["ring_idx"]
                nc1 = new_cart[nhcs[0]["c_g"][ci0]]
                nc2 = new_cart[nhcs[1]["c_g"][ci1]]
                rc1 = new_cart[
                    np.array(nhcs[0]["c_g"])[ri0]].mean(axis=0)
                rc2 = new_cart[
                    np.array(nhcs[1]["c_g"])[ri1]].mean(axis=0)
                nbd1 = rc1 - nc1
                nbd2 = rc2 - nc2
                nbd1_p = nbd1 - np.dot(nbd1, plane_normal) * plane_normal
                nbd2_p = nbd2 - np.dot(nbd2, plane_normal) * plane_normal
                new_bend = signed_bend_angle(nbd1_p, nbd2_p, plane_normal)
                print(f"  New bend: {new_bend:.2f} deg")
        else:
            print("  Already at target -- skipping bend.")

        # -- Op 3: restore min z-gap --
        min_z_after = min(new_cart[i, 2] for i in all_dimer_g)
        z_shift = (z_slab_top + min_gap) - min_z_after
        for i in all_dimer_g:
            new_cart[i, 2] += z_shift
        min_z_final = min(new_cart[i, 2] for i in all_dimer_g)
        print(f"\n  Z-shift: {z_shift:.3f} A")
        print(f"  Final min gap: {min_z_final - z_slab_top:.3f} A "
              f"(target: {min_gap:.3f} A)")

    # -- Convert back to fractional, wrap molecules, write --
    new_frac = cart_to_frac(new_cart, lattice)
    new_frac[n_off:] = new_frac[n_off:] % 1.0
    data["frac_coords"] = new_frac

    # -- Selective dynamics by z-height --
    if sd_enable:
        data["selective_dynamics"] = True
        natoms = len(new_frac)
        sd_flags = []
        n_free, n_fixed = 0, 0
        for i in range(natoms):
            if new_cart[i, 2] > sd_z_height:
                sd_flags.append(["T", "T", "T"])
                n_free += 1
            else:
                sd_flags.append(["F", "F", "F"])
                n_fixed += 1
        data["sd_flags"] = sd_flags
        print(f"\nSelective dynamics: z > {sd_z_height:.3f} A -> T T T")
        print(f"  Free: {n_free}, Fixed: {n_fixed}")

    write_poscar(poscar_out, data)
    print(f"\nWrote: {poscar_out}")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    poscar_in  = 'C:/Users/Benjamin Kafin/Documents/VASP/lone/dblflt_flat/CONTCAR'
    poscar_out = 'C:/Users/Benjamin Kafin/Documents/VASP/lone/dblflt_flat/POSCAR.vasp'

    tilt_angle       = 187.5  # 180 = Au in backbone plane. <180 = tighter.
    bend_angle       = 195.0  # 180 = flat backbones. <180 = more bent.
    ncn_planar_angle = 190 # 180 = carbene in plane. >180 = toward +z.
    wingtip_flap     = 191  # 180 = N-Cw in plane. >180 = Cw toward +z.
    wingtip_twist    = 192 # 180 = Cw-H in plane toward Au. >180 = H toward +z.

    sd_enable   = True   # Write selective dynamics flags.
    sd_z_height = 1.0    # A -- atoms with z > this are free (T T T).

    run(poscar_in, poscar_out, tilt_angle, bend_angle,
        ncn_planar_angle, wingtip_flap, wingtip_twist,
        sd_enable, sd_z_height)
