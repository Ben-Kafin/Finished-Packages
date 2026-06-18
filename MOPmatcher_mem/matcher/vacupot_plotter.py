from __future__ import annotations
import os
import json
import h5py
from types import SimpleNamespace
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple#, Union

import numpy as np
import matplotlib.pyplot as plt

try:
    import mplcursors
    HAS_MPLCURSORS = True
except Exception:
    HAS_MPLCURSORS = False

# Relative imports when loaded as the 'matcher' package; absolute fallbacks when
# the package is run directly as a standalone script (e.g. %runfile on
# matcher_vacupot.py, which imports this module). matcher_vacupot.py inserts the
# repo root on sys.path before importing this, so the absolute forms resolve.
try:
    from .classifier import StateBehaviorClassifier
    from .builder import detect_ncl_from_incar
except ImportError:
    from classifier import StateBehaviorClassifier
    from builder import detect_ncl_from_incar


def _read_rect_txt_delimited(path: str) -> Dict[str, Any]:
    """
    Parses matcher output. Retains metadata support for HOMOS, FERMIS, and VACUUMS.
    """
    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"homos": {}, "fermis": {}, "vacuums": {}}
    
    if not os.path.isfile(path):
        return {"rows": [], "meta": meta}
    
    with open(path, "r") as f:
        lines = f.readlines()

    header_line = ""
    for line in lines:
        s = line.strip()
        if s.startswith("# HOMOS:"):
            parts = s.replace("# HOMOS:", "").strip().split(";")
            for p in parts:
                if "=" in p:
                    k, v = p.split("=")
                    try: meta["homos"][k.strip()] = int(v.strip())
                    except ValueError: pass
        elif s.startswith("# FERMIS:"):
            parts = s.replace("# FERMIS:", "").strip().split(";")
            for p in parts:
                if "=" in p:
                    k, v = p.split("=")
                    try: meta["fermis"][k.strip()] = float(v.strip())
                    except ValueError: pass
        elif s.startswith("# VACUUMS:"):
            parts = s.replace("# VACUUMS:", "").strip().split(";")
            for p in parts:
                if "=" in p:
                    k, v = p.split("=")
                    try: meta["vacuums"][k.strip()] = float(v.strip())
                    except ValueError: pass
        elif s.startswith("# full_idx") or s.startswith("#full_idx"):
            header_line = s
            break
    
    if not header_line:
        return {"rows": [], "meta": meta}

    header_blocks = [b.strip() for b in header_line.lstrip('# ').split("|")]
    component_labels = []
    for block in header_blocks[1:-1]:
        label = block.split('_idx')[0].strip()
        component_labels.append(label)

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"): continue
        
        blocks = [b.strip() for b in s.split("|")]
        if len(blocks) != len(header_blocks): continue

        try:
            full_fields = blocks[0].split()
            rec = {"full_idx": int(full_fields[0]), "E_full": float(full_fields[1])}
            
            for i, label in enumerate(component_labels):
                comp_fields = blocks[i + 1].split()
                rec[label] = {
                    "idx": int(comp_fields[0]), "E": float(comp_fields[1]),
                    "dE": float(comp_fields[2]), "ov_best": float(comp_fields[3]),
                    "w_span": float(comp_fields[4])
                }
            rec["residual"] = float(blocks[-1])
            rows.append(rec)
        except (ValueError, IndexError): continue
            
    return {"rows": rows, "meta": meta}


def _read_ov_all(path: str) -> Tuple[Dict[int, Dict[str, List[Dict[str, float]]]], Dict[str, List[Tuple[int, float]]]]:
    by_full = defaultdict(lambda: defaultdict(list))
    comp_idx_E = defaultdict(dict)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 7: continue
            comp = parts[0]
            try:
                full_idx, comp_idx = int(parts[1]), int(parts[2])
                E_comp, dE_comp = float(parts[3]), float(parts[4])
                ov, w_span = float(parts[5]), float(parts[6])
            except Exception: continue
            rec = dict(comp_idx=comp_idx, E=E_comp, dE=dE_comp, ov=ov, w_span=w_span)
            by_full[full_idx][comp].append(rec)
            comp_idx_E[comp].setdefault(comp_idx, E_comp)
    comp_pairs = {comp: sorted(d.items(), key=lambda x: x[1]) for comp, d in comp_idx_E.items()}
    return by_full, comp_pairs


def _read_results_h5(results_path: str):
    """
    Reconstruct the same structures the two .txt readers produce, but from
    matcher_results.h5: returns ({"rows", "meta"}, (by_full, comp_pairs)).

    The .txt files store energies/overlaps at fixed print precision (%.4f for
    energies, %.5f for overlaps/w_span/residual, %.5f for the aligned vacuum and
    Fermis). The plotters were written to consume those rounded values, so the
    values reconstructed here are rounded to the SAME precision; otherwise the
    full-precision floats would shift colors/classifications in the last digit.
    The grid is read tile-free here (whole datasets); callers that need bounded
    memory at plot time can adapt, but the matcher already wrote a chunked grid.
    """
    with h5py.File(results_path, "r") as hf:
        comp_labels = json.loads(hf["meta"]["comp_labels_json"][()])
        E_full = hf["meta"]["E_full"][...]
        comp_energies = {l: hf["meta"]["comp_energies"][l][...] for l in comp_labels}
        n_full = int(E_full.shape[0])

        homos = {k: int(hf["meta"]["homo_indices"][k][()]) for k in hf["meta"]["homo_indices"]}
        fermis = {k: round(float(hf["meta"]["final_fermis"][k][()]), 5)
                  for k in hf["meta"]["final_fermis"]}
        vac = round(float(hf["meta"]["v_aligned"][()]), 5)
        meta = {"homos": homos, "fermis": fermis, "vacuums": {"aligned": vac}}

        grid = {l: hf["grid"][l][...] for l in comp_labels}

    rows = []
    by_full = defaultdict(lambda: defaultdict(list))
    comp_idx_E = defaultdict(dict)
    for j in range(n_full):
        E_full_j = round(float(E_full[j]), 4)
        bests = {}
        for label in comp_labels:
            Z = grid[label][:, j]
            energies = comp_energies[label]
            ib = int(np.argmax(Z))
            ob = round(float(Z.max()), 5)
            ws = round(float(Z.sum()), 5)
            Ec = round(float(energies[ib]), 4)
            dE = round(float(E_full[j] - energies[ib]), 4)
            bests[label] = {"idx": ib + 1, "E": Ec, "dE": dE, "ov_best": ob, "w_span": ws}
        resid = round(max(0.0, 1.0 - sum(float(grid[l][:, j].sum()) for l in comp_labels)), 5)
        rows.append({"full_idx": j + 1, "E_full": E_full_j, **bests, "residual": resid})
        for label in comp_labels:
            Z = grid[label][:, j]
            energies = comp_energies[label]
            ws = round(float(Z.sum()), 5)
            for ip in range(Z.shape[0]):
                Ec = round(float(energies[ip]), 4)
                dEc = round(float(E_full[j] - energies[ip]), 4)
                ov = round(float(Z[ip]), 5)
                by_full[j + 1][label].append(dict(comp_idx=ip + 1, E=Ec, dE=dEc, ov=ov, w_span=ws))
                comp_idx_E[label].setdefault(ip + 1, Ec)
    comp_pairs = {comp: sorted(d.items(), key=lambda x: x[1]) for comp, d in comp_idx_E.items()}
    return {"rows": rows, "meta": meta}, (dict(by_full), comp_pairs)
@dataclass
class PlotConfig:
    cmap_name_simple: str = "managua_r"
    cmap_name_metal: str = "vanimo_r"
    power_simple_neg: float = 0.25
    power_simple_pos: float = 0.75
    power_metal_neg: float = 0.075
    power_metal_pos: float = 0.075
    power_residual: float = 2.0
    figsize: Tuple[float, float] = (8.0, 3.0)
    lw_stick: float = 2.0
    xlabel: str = "Energy (eV)"
    ylabel: str = "Normalized"
    
    show_fermi_line: bool = True
    fermi_line_style: str = ":"
    fermi_line_color: str = "k"
    show_local_fermi: bool = True
    local_fermi_style: str = "--"
    local_fermi_color: str = "red"

    show_vacuum_line: bool = True
    vacuum_line_color: str = "black"
    vacuum_line_width: float = 3.5 
    vacuum_line_style: str = "-"
    
    annotate_on_hover: bool = True
    interactive: bool = True
    shared_molecule_color: bool = False
    energy_range: Optional[Tuple[float, float]] = None
    title_full: str = "Full system"
    pick_primary: Any = "blended" 
    min_total_mol_wspan: float = 0.025
    soc_pair_thresh_eV: float = 0.001


class RectAEPAWColorPlotter:
    def __init__(self, config: Optional[PlotConfig] = None):
        self.cfg = config or PlotConfig()
        self._artists_by_comp: Dict[str, List[Any]] = {}
        self._hover_map_comp: Dict[str, Dict[Any, str]] = {} 
        self._cursor_by_comp: Dict[str, Any] = {}
        self._artists_f: List[Any] = []
        self._hover_map_f: Dict[Any, str] = {}
        self._cursor_f = None
        self._cursor_by_axes: Dict[Any, Any] = {}

    def _get_cmap(self, name: str):
        try: return plt.get_cmap(name)
        except Exception: return plt.get_cmap("viridis")

    def _build_colors_rank_pivot(self, pairs: List[Tuple[int, float]], cmap_name: str,
                                 center_idx: Optional[int], power_neg: float, power_pos: float) -> Dict[int, Tuple]:
        if not pairs: return {}
        ordered = sorted(pairs, key=lambda t: t[1])
        idxs = [idx for idx, _ in ordered]
        n = len(ordered)
        pivot = idxs.index(center_idx) if (center_idx is not None and center_idx in idxs) else n // 2
        neg_count = max(pivot, 1); pos_count = max(n - pivot - 1, 1)
        cmap = self._get_cmap(cmap_name)
        def warp(r): return abs(r) ** (power_neg if r < 0 else power_pos)
        colors = {}
        for i, (idx, _) in enumerate(ordered):
            r = (i - pivot) / neg_count if i <= pivot else (i - pivot) / pos_count
            v = 0.5 + 0.5 * np.sign(r) * warp(r)
            colors[idx] = cmap(np.clip(v, 0.0, 1.0))
        return colors

    def _build_colors_with_soc_pairing(self, pairs: List[Tuple[int, float]], cmap_name: str,
                                       center_idx: Optional[int], power_neg: float, power_pos: float,
                                       is_soc: bool, thresh: float) -> Dict[int, Tuple]:
        if not is_soc or not pairs:
            return self._build_colors_rank_pivot(pairs, cmap_name, center_idx, power_neg, power_pos)
        ordered = sorted(pairs, key=lambda t: t[1])
        n = len(ordered)

        # Resolve pivot energy: center_idx's energy in ordered, else middle state
        pivot_E: Optional[float] = None
        if center_idx is not None:
            for idx, E in ordered:
                if idx == center_idx:
                    pivot_E = E
                    break
        if pivot_E is None:
            pivot_E = ordered[n // 2][1]

        # Walk sorted list; assign reps so each pair adopts the closer-to-pivot member's slot
        idx_to_rep: Dict[int, int] = {}
        i = 0
        while i < n:
            idx_i, E_i = ordered[i]
            if i + 1 < n and (ordered[i + 1][1] - E_i) < thresh:
                idx_j, E_j = ordered[i + 1]
                if center_idx is not None and center_idx in (idx_i, idx_j):
                    rep = center_idx
                elif E_j <= pivot_E:
                    rep = idx_j  # both below pivot; upper member is closer
                elif E_i >= pivot_E:
                    rep = idx_i  # both above pivot; lower member is closer
                else:
                    # pair straddles pivot without containing center_idx
                    rep = idx_i if abs(E_i - pivot_E) <= abs(E_j - pivot_E) else idx_j
                idx_to_rep[idx_i] = rep
                idx_to_rep[idx_j] = rep
                i += 2
            else:
                idx_to_rep[idx_i] = idx_i
                i += 1

        # Color the FULL original distribution; pair members look up their rep's color
        base_colors = self._build_colors_rank_pivot(pairs, cmap_name, center_idx, power_neg, power_pos)
        return {orig_idx: base_colors[rep] for orig_idx, rep in idx_to_rep.items()}

    def _mix_component_color(self, recs: List[Dict[str, float]], base_colors: Dict[int, Any], default=(0.4, 0.4, 0.4, 1.0)):
        if not recs: return default
        num = np.zeros(3); denom = 0.0
        for r in recs:
            idx, w = int(r["comp_idx"]), max(float(r.get("ov", 0.0)), 0.0)
            c = base_colors.get(idx, default)
            num[:3] += w * np.array(c[:3], dtype=float)
            denom += w
        if denom <= 0: return default
        return (*(num / denom), 1.0)

    def load(self, path: str):
        return _read_rect_txt_delimited(path)

    def plot(self, path: str, ax: Optional[plt.Axes] = None, bonding: bool = False):
        self._cursor_by_axes.clear()
        P = self._prepare(path, bonding, write_summaries=True)

        n_comp_axes = len(P.mol_labels) + (1 if P.metal_present else 0)
        total_rows = max(1, n_comp_axes) + 1
        figsize = (self.cfg.figsize[0], max(3, 1.5 * total_rows))
        fig, axes = plt.subplots(total_rows, 1, sharex=True, figsize=figsize)
        axes = [axes] if total_rows == 1 else list(axes)
        comp_axes, ax_f = axes[:n_comp_axes], axes[-1]

        for i, comp_label in enumerate(P.comp_iter_order):
            self._draw_component_axis(
                comp_axes[i], comp_label,
                P.comp_pairs.get(comp_label, []),
                P.component_colors.get(comp_label, {}),
                P.component_class_maps.get(comp_label, {}),
                P.known_fermis.get(comp_label), P.v_aligned)

        self._draw_full_axis(
            ax_f, P.rows, P.by_full, P.comp_iter_order, P.component_colors,
            P.classifier, P.mol_labels, P.known_fermis, P.v_aligned)

        self._wire_global_click(fig)
        fig.tight_layout()
        return fig, axes

    def _prepare(self, path: str, bonding: bool, write_summaries: bool = True):
        """
        Load a single MOPMatcher run's matcher output and build the per-component
        color maps and classification maps. Returns a SimpleNamespace bundle.

        Reads from matcher_results.h5 (the complete run record) when it is present
        in the run directory, reconstructing the same rows / meta / by_full /
        comp_pairs structures the .txt readers produce (at the .txt print
        precision). Falls back to parsing the .txt outputs when the .h5 is absent
        (e.g. an older run). With write_summaries=True (plot's default) the
        behavior summary files are written exactly as before; with
        write_summaries=False (used by the combined plotter) no files are
        written.
        """
        results_path = os.path.join(os.path.dirname(path) or ".", "matcher_results.h5")
        if os.path.isfile(results_path):
            data, (by_full, comp_pairs) = _read_results_h5(results_path)
            rows, meta = data["rows"], data["meta"]
        else:
            data = self.load(path)
            rows, meta = data["rows"], data["meta"]
            ov_all_path = os.path.join(os.path.dirname(path), "band_matches_rectangular_all.txt")
            by_full, comp_pairs = _read_ov_all(ov_all_path) if os.path.isfile(ov_all_path) else ({}, {})

        known_homos, known_fermis = meta.get("homos", {}), meta.get("fermis", {})
        known_vacuums = meta.get("vacuums", {})
        v_aligned = known_vacuums.get("aligned")

        classifier = StateBehaviorClassifier()
        output_dir = os.path.dirname(path) or "."
        is_soc = detect_ncl_from_incar(output_dir)
        if write_summaries:
            class_maps, class_maps_bonding = classifier.classify_and_write_summaries(by_full, output_dir)
        else:
            class_maps, class_maps_bonding = classifier.classify_maps(by_full)
        component_class_maps = class_maps_bonding if bonding else class_maps

        comp_labels_all = list(comp_pairs.keys())
        metal_present = "metal" in comp_labels_all
        mol_labels = [lbl for lbl in comp_labels_all if lbl.lower() != "metal"]

        component_colors: Dict[str, Dict[int, Tuple]] = {}

        def get_center_idx(lbl, pairs):
            if lbl in known_homos: return int(known_homos[lbl])
            occupied = [(idx, E) for idx, E in pairs if E <= 0]
            return max(occupied, key=lambda x: x[1])[0] if occupied else None

        if self.cfg.shared_molecule_color and mol_labels:
            ref_label = mol_labels[0]
            shared_map = {int(idx): float(E) for lbl in mol_labels for idx, E in comp_pairs.get(lbl, [])}
            shared_pairs = sorted(shared_map.items(), key=lambda t: t[1])
            center = get_center_idx(ref_label, shared_pairs)
            shared_colors = self._build_colors_with_soc_pairing(shared_pairs, self.cfg.cmap_name_simple, center, self.cfg.power_simple_neg, self.cfg.power_simple_pos, is_soc, self.cfg.soc_pair_thresh_eV)
            for lbl in mol_labels: component_colors[lbl] = shared_colors
        else:
            for lbl in mol_labels:
                pairs = comp_pairs.get(lbl, [])
                center = get_center_idx(lbl, pairs)
                component_colors[lbl] = self._build_colors_with_soc_pairing(pairs, self.cfg.cmap_name_simple, center, self.cfg.power_simple_neg, self.cfg.power_simple_pos, is_soc, self.cfg.soc_pair_thresh_eV)

        if metal_present:
            pairs = comp_pairs.get("metal", [])
            center = get_center_idx("metal", pairs)
            component_colors["metal"] = self._build_colors_with_soc_pairing(pairs, self.cfg.cmap_name_metal, center, self.cfg.power_metal_neg, self.cfg.power_metal_pos, is_soc, self.cfg.soc_pair_thresh_eV)

        comp_iter_order = (["metal"] if metal_present else []) + mol_labels

        return SimpleNamespace(
            rows=rows, by_full=by_full, comp_pairs=comp_pairs,
            known_homos=known_homos, known_fermis=known_fermis, v_aligned=v_aligned,
            is_soc=is_soc, classifier=classifier,
            class_maps=class_maps, class_maps_bonding=class_maps_bonding,
            component_class_maps=component_class_maps, component_colors=component_colors,
            metal_present=metal_present, mol_labels=mol_labels, comp_iter_order=comp_iter_order,
            bonding=bonding,
        )

    @staticmethod
    def _dual_hover_text(comp_idx, E_global, left_info, right_info, left_title, right_title):
        """
        Two-column ("twice as wide") hover body for the shared-molecule row.
        Left column = secondary run (e.g. complex, classified on the shifted /
        global energy axis); right column = this axis' run (e.g. full). The
        per-branch fields are identical to the standard single-column hover.
        """
        def body(info):
            ms, var = info.get('mean_shift', 0.0), info.get('variance', 0.0)
            return [f"up    E={info.get('E_plus',0):+.3f}, I={info.get('I_plus',0):.3f}",
                    f"zero  E={info.get('E_zero',0):+.3f}, I={info.get('I_zero',0):.3f}",
                    f"down  E={info.get('E_minus',0):+.3f}, I={info.get('I_minus',0):.3f}",
                    f"mean shift {ms:+.3f} eV",
                    f"variance   {var:.3f} eV^2"]
        left, right = body(left_info), body(right_info)
        colw = max([len(left_title)] + [len(s) for s in left])
        head = f"{left_title:<{colw}}   {right_title}"
        lines = [f"{l:<{colw}}   {r}" for l, r in zip(left, right)]
        return f"band {comp_idx}, E {E_global:+.3f} eV\n" + head + "\n" + "\n".join(lines)

    def _draw_component_axis(self, axc, comp_label, pairs, colors_map, class_map,
                             local_fermi, v_aligned, shift=0.0, title=None,
                             key=None, dual=None):
        """
        Draw one component axis. Extracted verbatim from the original plot()
        per-component loop body; with the default arguments (shift=0.0,
        title=None, key=None, dual=None) it reproduces the original single-run
        rendering exactly.

        shift : added to every drawn energy and to the displayed absolute
                energy in the hover (energy differences are not shifted).
        title : axis title (defaults to comp_label).
        key   : bookkeeping key for the per-axis artist/hover/cursor dicts
                (defaults to comp_label); pass a unique key when the same
                comp_label appears on more than one axis (e.g. two runs).
        dual  : when provided, render a two-column ("twice as wide") hover.
                Keys: 'class_map' (secondary/left map keyed by comp_idx, already
                on this axis' energy scale), 'left_title', 'right_title'. The
                axis' own class_map is used for the right column.
        """
        k = key if key is not None else comp_label
        if self.cfg.show_fermi_line: axc.axvline(0.0, color=self.cfg.fermi_line_color, linestyle=self.cfg.fermi_line_style, alpha=0.7)

        artists, hover_map = self._artists_by_comp.setdefault(k, []), self._hover_map_comp.setdefault(k, {})
        artists.clear(); hover_map.clear()

        for comp_idx, E in pairs:
            line = axc.vlines(E + shift, 0, 1, color=colors_map.get(comp_idx, "black"), linewidth=self.cfg.lw_stick)
            artists.append(line)
            if dual is not None:
                hover_map[line] = self._dual_hover_text(
                    comp_idx, E + shift,
                    dual.get("class_map", {}).get(comp_idx, {}),
                    class_map.get(comp_idx, {}),
                    dual.get("left_title", "left"),
                    dual.get("right_title", "right"))
            else:
                info = class_map.get(comp_idx, {})
                ms, var = info.get('mean_shift', 0.0), info.get('variance', 0.0)
                body = (f"up    E={info.get('E_plus',0):+.3f}, I={info.get('I_plus',0):.3f}\n"
                        f"zero  E={info.get('E_zero',0):+.3f}, I={info.get('I_zero',0):.3f}\n"
                        f"down  E={info.get('E_minus',0):+.3f}, I={info.get('I_minus',0):.3f}\n"
                        f"mean shift {ms:+.3f} eV\n" f"variance   {var:.3f} eV^2")
                hover_map[line] = f"{comp_label} band {comp_idx}, E {E + shift:+.3f} eV\n{body}"

        if self.cfg.show_local_fermi and local_fermi is not None:
            axc.axvline(local_fermi + shift, color=self.cfg.local_fermi_color, linestyle=self.cfg.local_fermi_style, alpha=0.9, zorder=10)

        if self.cfg.show_vacuum_line and v_aligned is not None:
            axc.axvline(v_aligned + shift, color=self.cfg.vacuum_line_color, linewidth=self.cfg.vacuum_line_width,
                        linestyle=self.cfg.vacuum_line_style, zorder=10)

        axc.set_ylabel(self.cfg.ylabel); axc.set_title(title if title is not None else comp_label)
        if self.cfg.annotate_on_hover and HAS_MPLCURSORS and artists:
            cur = mplcursors.cursor(artists, hover=True); cur.enabled = False
            self._cursor_by_comp[k] = cur; self._cursor_by_axes[axc] = cur
            _mono = dual is not None
            @cur.connect("add")
            def _on_add_comp(sel, hmap=hover_map, mono=_mono):
                if (txt := hmap.get(sel.artist)):
                    sel.annotation.set_text(txt)
                    if mono: sel.annotation.set_fontfamily("monospace")

    def _draw_full_axis(self, ax_f, rows, by_full, comp_iter_order, component_colors,
                        classifier, mol_labels, known_fermis, v_aligned,
                        shift=0.0, title=None, xlabel=True):
        """
        Draw the full-system axis. Extracted verbatim from the original plot()
        full-system block; with the default arguments (shift=0.0, title=None,
        xlabel=True) it reproduces the original single-run rendering exactly.

        shift  : added to every drawn full-system energy, to the displayed
                 absolute energies in the hover (E_full and each component's E),
                 and to the local-fermi / vacuum lines. Energy differences (dE),
                 overlaps, residuals, and the z-metric (which derive only from
                 dE / ov) are NOT shifted, so segment colors, tiering and sort
                 order are identical to the unshifted case.
        title  : axis title (defaults to cfg.title_full).
        xlabel : whether to set the x-axis label (only the bottom axis of a
                 shared-x stack should; plot() always passes True).

        Uses local artist/hover dictionaries (mirrored onto self._artists_f /
        self._hover_map_f at the end for backward compatibility) so the hover
        closure captures a per-call map; this lets a stack contain more than
        one full-system axis with independent hovers.
        """
        metal_segments_to_plot, molecule_segments_to_plot = [], []
        RES_T = 0.02
        C_GREY = np.array([1.0, 1.0, 1.0])

        for rec in rows:
            E_full, full_idx = float(rec["E_full"]), int(rec["full_idx"])
            res = rec.get("residual", 0.0)
            w_res = np.clip(((res - RES_T) / (1.0 - RES_T)) ** self.cfg.power_residual, 0.0, 1.0) if res > RES_T else 0.0

            comps = by_full.get(full_idx, {})
            all_wspans = {lbl: rec.get(lbl, {}).get('w_span', 0.0) for lbl in comp_iter_order}
            total_mol_wspan = sum(w for lbl, w in all_wspans.items() if lbl != "metal")

            comp_lines = [f"{lbl}: idx {int(top['idx'])}, E {top['E'] + shift:+.3f}, dE {top['dE']:+.3f}, ov {top['ov_best']:.4f}, w_span {top['w_span']:.4f}"
                          for lbl in comp_iter_order if (top := rec.get(lbl))]
            hover_text = f"full_idx {full_idx}\nE_full {E_full + shift:+.3f}\n" + "\n".join(comp_lines) + f"\nresidual {res:.5f}"

            # Full weighted energy shift for the current E_full state across ALL components
            pooled_recs = [r for lbl in comp_iter_order for r in comps.get(lbl, [])]
            state_classification = classifier.classify_state(pooled_recs)
            full_weighted_z_metric = abs(state_classification.get("mean_shift", 0.0))

            if self.cfg.pick_primary is True:
                winner_lbl = max(all_wspans, key=all_wspans.get)
                w_info = rec.get(winner_lbl, {})
                if winner_lbl == "metal":
                    color = self._mix_component_color(comps.get("metal", []), component_colors.get("metal", {}))
                    color = tuple((1.0 - w_res) * np.array(color[:3]) + w_res * C_GREY)
                    metal_segments_to_plot.append((full_weighted_z_metric, E_full, 0.0, 1.0, color, hover_text))
                else:
                    z_metric = abs(w_info.get('dE', 0.0))
                    color = component_colors.get(winner_lbl, {}).get(int(w_info['idx']), (0.4, 0.4, 0.4))
                    color = tuple((1.0 - w_res) * np.array(color[:3]) + w_res * C_GREY)
                    molecule_segments_to_plot.append((z_metric, E_full, 0.0, 1.0, color, hover_text))

            elif self.cfg.pick_primary == "blended":
                final_rgb, total_ov_sum = np.zeros(3), 0.0
                for lbl in comp_iter_order:
                    for r in comps.get(lbl, []):
                        ov = r.get('ov', 0.0)
                        if ov > 1e-8:
                            c = component_colors.get(lbl, {}).get(int(r['comp_idx']), (0.4, 0.4, 0.4))
                            final_rgb += np.array(c[:3]) * ov
                            total_ov_sum += ov
                blend_col = tuple(final_rgb / total_ov_sum) if total_ov_sum > 0 else (0.4, 0.4, 0.4)
                blend_col = tuple((1.0 - w_res) * np.array(blend_col[:3]) + w_res * C_GREY)

                if total_mol_wspan >= self.cfg.min_total_mol_wspan:
                    mol_dEs = [abs(rec[lbl]['dE']) for lbl in mol_labels if lbl in rec]
                    z_metric = max(mol_dEs) if mol_dEs else 0.0
                    molecule_segments_to_plot.append((z_metric, E_full, 0.0, 1.0, blend_col, hover_text))
                else:
                    metal_segments_to_plot.append((full_weighted_z_metric, E_full, 0.0, 1.0, blend_col, hover_text))

            else:
                final_rgb, total_ov_sum = np.zeros(3), 0.0
                for lbl in comp_iter_order:
                    for r in comps.get(lbl, []):
                        ov = r.get('ov', 0.0)
                        if ov > 1e-8:
                            c = component_colors.get(lbl, {}).get(int(r['comp_idx']), (0.4, 0.4, 0.4))
                            final_rgb += np.array(c[:3]) * ov
                            total_ov_sum += ov
                global_blend_col = tuple(final_rgb / total_ov_sum) if total_ov_sum > 0 else (0.4, 0.4, 0.4)
                global_blend_col = tuple((1.0 - w_res) * np.array(global_blend_col[:3]) + w_res * C_GREY)

                if total_mol_wspan < self.cfg.min_total_mol_wspan:
                    metal_segments_to_plot.append((full_weighted_z_metric, E_full, 0.0, 1.0, global_blend_col, hover_text))
                else:
                    h = 1.0 / len(mol_labels) if mol_labels else 1.0
                    for i, mol_lbl in enumerate(mol_labels):
                        y_bot, y_top = 1.0 - (i + 1) * h, 1.0 - i * h
                        m_rec = rec.get(mol_lbl, {})
                        if m_rec.get('w_span', 0.0) >= self.cfg.min_total_mol_wspan:
                            m_recs = comps.get(mol_lbl, [])
                            if m_recs:
                                # Determine color-specific record for segment
                                top_r = max(m_recs, key=lambda r: r.get("ov", 0.0))
                                seg_col = component_colors[mol_lbl].get(int(top_r['comp_idx']), (0.4, 0.4, 0.4, 1.0))
                                # Independent Z-metric based on absolute shift of matching state
                                z_metric = abs(top_r.get('dE', 0.0))
                            else:
                                seg_col, z_metric = global_blend_col, full_weighted_z_metric
                        else:
                            seg_col, z_metric = global_blend_col, full_weighted_z_metric

                        seg_col = tuple((1.0 - w_res) * np.array(seg_col[:3]) + w_res * C_GREY)
                        molecule_segments_to_plot.append((z_metric, E_full, y_bot, y_top, seg_col, hover_text))

        # Sort independently by segment-specific absolute shift (Highest shifts drawn last/on top)
        metal_segments_to_plot.sort(key=lambda x: x[0])
        molecule_segments_to_plot.sort(key=lambda x: x[0])

        artists_f, hover_map_f = [], {}

        # Draw Tier 1: Metal/Blended states first
        for _, E_full, y_bot, y_top, col, hover_text in metal_segments_to_plot:
            line = ax_f.vlines(E_full + shift, y_bot, y_top, color=col, lw=self.cfg.lw_stick)
            artists_f.append(line); hover_map_f[line] = hover_text

        # Draw Tier 2: Molecule states second
        for _, E_full, y_bot, y_top, col, hover_text in molecule_segments_to_plot:
            line = ax_f.vlines(E_full + shift, y_bot, y_top, color=col, lw=self.cfg.lw_stick)
            artists_f.append(line); hover_map_f[line] = hover_text

        if self.cfg.show_fermi_line: ax_f.axvline(0.0, color=self.cfg.fermi_line_color, linestyle=self.cfg.fermi_line_style, alpha=0.7)
        if self.cfg.show_local_fermi and (lf_full := known_fermis.get("full")) is not None:
             ax_f.axvline(lf_full + shift, color=self.cfg.local_fermi_color, linestyle=self.cfg.local_fermi_style, alpha=0.9, zorder=10)

        if self.cfg.show_vacuum_line and v_aligned is not None:
            ax_f.axvline(v_aligned + shift, color=self.cfg.vacuum_line_color, linewidth=self.cfg.vacuum_line_width,
                         linestyle=self.cfg.vacuum_line_style, zorder=10)

        ax_f.set_title(title if title is not None else self.cfg.title_full); ax_f.set_ylabel(self.cfg.ylabel)
        if xlabel: ax_f.set_xlabel(self.cfg.xlabel)
        if self.cfg.energy_range: ax_f.set_xlim(self.cfg.energy_range)

        self._artists_f, self._hover_map_f = artists_f, hover_map_f
        if self.cfg.annotate_on_hover and HAS_MPLCURSORS and artists_f:
            self._cursor_f = mplcursors.cursor(artists_f, hover=True); self._cursor_f.enabled = False
            self._cursor_by_axes[ax_f] = self._cursor_f
            @self._cursor_f.connect("add")
            def _on_add_f(sel, hmap=hover_map_f):
                if (txt := hmap.get(sel.artist)): sel.annotation.set_text(txt)

    def _wire_global_click(self, fig):
        """
        Attach the figure-level click handler (extracted verbatim from the
        original plot()): left-click enables an axis' hover cursor, right-click
        clears and disables it.
        """
        if HAS_MPLCURSORS:
            def _on_click(event, _map=self._cursor_by_axes):
                if event.inaxes is None: return
                cur = _map.get(event.inaxes)
                if cur is None: return
                if event.button == 1:
                    if not cur.enabled:
                        cur.enabled = True
                elif event.button == 3:
                    for sel in list(cur.selections):
                        cur.remove_selection(sel)
                    cur.enabled = False
            fig.canvas.mpl_connect("button_press_event", _on_click)