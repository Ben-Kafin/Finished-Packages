# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:20:52 2025

@author: Benjamin Kafin
"""

# File: classifier.py
import os
from collections import defaultdict
from typing import List, Dict, Union, Tuple, Any
import numpy as np

class StateBehaviorClassifier:
    """
    Classifies a component band's behavior into shift vs split.
    Handles multiple, dynamically named components from the matcher's output.
    """

    def __init__(self):
        pass

    def classify_state(
        self,
        recs: List[Dict[str, float]]
    ) -> Dict[str, Union[str, float]]:
        """
        Classifies a set of (dE, ov) records into up/zero/down branches,
        computes weighted mean shift & variance, and determines shift vs split mode.
        """
        dE = np.array([r["dE"] for r in recs], dtype=float)
        ov = np.array([r["ov"] for r in recs], dtype=float)
        total_ov = float(ov.sum())

        if total_ov <= 0:
            return { "mode": "shift", "total_ov": 0.0, "mean_shift": 0.0, "variance": 0.0,
                     "E_plus": 0.0, "I_plus":  0.0, "E_zero": 0.0, "I_zero": 0.0,
                     "E_minus": 0.0, "I_minus": 0.0 }

        mean_shift = float((ov * dE).sum() / total_ov)
        variance = float(((ov * (dE - mean_shift)**2).sum()) / total_ov)

        w_plus = float(ov[dE > 0].sum())
        w_zero = float(ov[dE == 0].sum())
        w_minus = float(ov[dE < 0].sum())
        w_total = w_plus + w_zero + w_minus

        I_plus = w_plus / w_total if w_total > 0 else 0.0
        I_zero = w_zero / w_total if w_total > 0 else 0.0
        I_minus = w_minus / w_total if w_total > 0 else 0.0

        E_plus = float((ov[dE > 0] * dE[dE > 0]).sum() / w_plus) if w_plus > 0 else 0.0
        E_zero = 0.0
        E_minus = float((ov[dE < 0] * dE[dE < 0]).sum() / w_minus) if w_minus > 0 else 0.0

        # A state is a "shift" if it has contributions in one or fewer directions (up, down, or zero)
        mode = "shift" if sum(b > 1e-6 for b in (w_plus, w_zero, w_minus)) <= 1 else "split"

        return {
            "mode": mode, "total_ov": total_ov, "mean_shift": mean_shift, "variance": variance,
            "E_plus": E_plus, "I_plus": I_plus, "E_zero": E_zero, "I_zero": I_zero,
            "E_minus": E_minus, "I_minus": I_minus
        }

    def _make_bonding_records(
        self,
        recs: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """
        Clamps any energy shift that crosses the Fermi level (0 eV) for
        bonding/anti-bonding analysis.
        """
        bonded = []
        for r in recs:
            E_comp, dE = r["E"], r["dE"]
            E_full = E_comp + dE
            if E_comp > 0 and E_full < 0:      # Downward crossing
                dE_bond = E_full
            elif E_comp < 0 and E_full > 0:    # Upward crossing
                dE_bond = -E_comp
            else:                              # No crossing
                dE_bond = dE
            bonded.append({"dE": dE_bond, "ov": r["ov"]})
        return bonded

    def classify_state_bonding(
        self,
        recs: List[Dict[str, float]]
    ) -> Dict[str, Union[str, float]]:
        """
        Applies the zero-crossing clamp before running the standard classification.
        """
        bond_recs = self._make_bonding_records(recs)
        return self.classify_state(bond_recs)

    def classify_maps(
        self,
        by_full: Dict[int, Dict[str, List[Dict[str, float]]]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Groups data by component and runs classification (normal + bonding),
        returning the classification maps WITHOUT writing any summary files.

        Extracted from classify_and_write_summaries so callers (e.g. a combined
        multi-run plotter that aligns energies before classifying) can obtain
        classification maps without the file-writing side effect. The grouping
        and classification logic is identical to the original write path.
        """
        comp_groups = defaultdict(lambda: defaultdict(list))
        for comps in by_full.values():
            for comp_label, recs in comps.items():
                for r in recs:
                    comp_groups[comp_label][int(r["comp_idx"])].append(r)

        class_maps_normal = defaultdict(dict)
        class_maps_bonding = defaultdict(dict)
        for comp_label, groups in comp_groups.items():
            for idx, recs in groups.items():
                class_maps_normal[comp_label][idx] = self.classify_state(recs)
                class_maps_bonding[comp_label][idx] = self.classify_state_bonding(recs)

        return dict(class_maps_normal), dict(class_maps_bonding)

    def classify_and_write_summaries(
        self,
        by_full: Dict[int, Dict[str, List[Dict[str, float]]]],
        output_dir: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Groups data by component, runs classification, writes summary files,
        and returns the classification results for direct use in the plotter.
        """
        class_maps_normal, class_maps_bonding = self.classify_maps(by_full)

        comp_groups = defaultdict(lambda: defaultdict(list))
        for comps in by_full.values():
            for comp_label, recs in comps.items():
                for r in recs:
                    comp_groups[comp_label][int(r["comp_idx"])].append(r)

        for comp_label, groups in comp_groups.items():
            safe_label = str(comp_label).strip().replace(os.sep, "_")
            normal_path = os.path.join(output_dir, f"{safe_label}_behavior.txt")
            bonding_path = os.path.join(output_dir, f"bonding_{safe_label}_behavior.txt")

            self._write_summary_file(groups, class_maps_normal[comp_label], normal_path)
            self._write_summary_file(groups, class_maps_bonding[comp_label], bonding_path)
            print(f"[CLASSIFIER] Wrote behavior files for '{comp_label}'")

        return class_maps_normal, class_maps_bonding

    def _write_summary_file(self, groups, class_map, filename):
        """Helper to write a single fixed-width summary file."""
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "w") as f:
            f.write(
                "# comp_idx  band_E    total_ov    "
                "E_plus    I_plus    E_zero    I_zero    E_minus    I_minus    "
                "mean_shift    variance\n"
            )
            for comp_idx in sorted(groups):
                info = class_map.get(comp_idx, {})
                band_E = float(groups[comp_idx][0]["E"]) if groups.get(comp_idx) else 0.0
                f.write(
                    f"{comp_idx:^10d} "
                    f"{band_E:^9.3f} "
                    f"{info.get('total_ov', 0.0):^10.5f}  "
                    f"{info.get('E_plus', 0.0):^9.3f}  {info.get('I_plus', 0.0):^8.3f}  "
                    f"{info.get('E_zero', 0.0):^9.3f}  {info.get('I_zero', 0.0):^8.3f}  "
                    f"{info.get('E_minus', 0.0):^9.3f}  {info.get('I_minus', 0.0):^8.3f}  "
                    f"{info.get('mean_shift', 0.0):^12.5f}  "
                    f"{info.get('variance', 0.0):^10.5f}\n"
                )