import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Locpot
from numpy import array, shape, dot
from os.path import exists, getsize
import sys
import os

# --- Helper Functions (Shared) ---

    
def load_locpot(locpot_path):
    return Locpot.from_file(locpot_path)


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

# --- Main Calculation Function ---

def calculate_work_function(locpot, ef, curvature_tol=5e-8, dipole_threshold=0.15, min_width=10, plot=True, verbose=True):
    """
    Calculates work function using robust wraparound handling.
    plot: Boolean to toggle matplotlib output.
    verbose: Boolean to toggle console printing.
    """
    # Extract and average the potential
    potential_data = locpot.data['total']
    z_potential = np.mean(np.mean(potential_data, axis=0), axis=0)
    nz = z_potential.shape[0]
    z_coords = np.arange(nz)
    
    # --- 1. Periodic Curvature Calculation ---
    z_padded = np.pad(z_potential, (2, 2), mode='wrap')
    slopes_padded = np.gradient(z_padded)
    curvatures_padded = np.abs(np.gradient(slopes_padded))
    curvatures = curvatures_padded[2:-2]
    
    # --- 2. Identify Linear Regions (Raw) ---
    stable_indices = np.where(curvatures < curvature_tol)[0]
    
    if stable_indices.size == 0:
        if verbose: print(f"[ERROR] No stable curvature region found (tol={curvature_tol}).")
        return ef, np.max(z_potential), np.max(z_potential) - ef

    # Group contiguous stable indices
    linear_regions = []
    if stable_indices.size > 0:
        start = stable_indices[0]
        for i in range(1, len(stable_indices)):
            if stable_indices[i] != stable_indices[i - 1] + 1:
                end = stable_indices[i - 1] + 1
                linear_regions.append({
                    'start': start, 'end': end, 
                    'width': end - start, 
                    'avg_v': np.mean(z_potential[start:end]),
                    'is_wrapped': False
                })
                start = stable_indices[i]
        end = stable_indices[-1] + 1
        linear_regions.append({
            'start': start, 'end': end, 
            'width': end - start, 
            'avg_v': np.mean(z_potential[start:end]),
            'is_wrapped': False
        })

    # --- 3. ROBUST WRAPAROUND STITCHING ---
    regions = []
    if len(linear_regions) > 1:
        first_r = linear_regions[0]
        last_r = linear_regions[-1]
        
        # Check if the last region touches the end (N) and first touches start (0)
        touches_start = (first_r['start'] <= 1)
        touches_end = (last_r['end'] >= nz - 1)
        
        if touches_start and touches_end:
            # Merge them into one "Wrapped" region
            total_points = first_r['width'] + last_r['width']
            weighted_v = (first_r['avg_v'] * first_r['width'] + last_r['avg_v'] * last_r['width']) / total_points
            
            wrapped_region = {
                'start': last_r['start'], 
                'end': first_r['end'],    
                'width': total_points, 
                'avg_v': weighted_v,
                'is_wrapped': True,
                'segments': [last_r, first_r] 
            }
            regions = [wrapped_region] + linear_regions[1:-1]
        else:
            regions = linear_regions
    else:
        regions = linear_regions

    # --- 4. Filtering and Selection ---
    
    # A. Filter by Potential Height (must be "vacuum-like")
    v_min = np.min(z_potential)
    v_max = np.max(z_potential)
    midpoint = (v_min + v_max) / 2
    
    candidates = []
    for r in regions:
        if r['avg_v'] > midpoint and r['width'] >= min_width:
            candidates.append(r)
            
    if not candidates:
        if verbose: print("[WARN] No valid vacuum candidates found after filtering. Reverting to widest raw region.")
        candidates = regions

    # B. Sort Spatially
    for r in candidates:
        if r['is_wrapped']:
             r['sort_index'] = nz 
        else:
             r['sort_index'] = r['start']
             
    candidates.sort(key=lambda x: x['sort_index'])

    # C. Dipole Detection
    best_region = None
    selection_mode = "Default"
    dipole_found = False
    
    if len(candidates) > 1:
        max_step_diff = 0
        split_index = -1 
        
        for i in range(len(candidates) - 1):
            diff = abs(candidates[i]['avg_v'] - candidates[i+1]['avg_v'])
            if diff > max_step_diff:
                max_step_diff = diff
                split_index = i
        
        if max_step_diff > dipole_threshold:
            dipole_found = True
            post_step_regions = candidates[split_index+1:]
            if not post_step_regions:
                post_step_regions = [candidates[-1]]
            best_region = max(post_step_regions, key=lambda x: x['width'])
            selection_mode = "Post-Step (Right/Wrapped)"
        else:
            best_region = max(candidates, key=lambda x: x['width'])
            selection_mode = "Widest (No Step)"
    else:
        best_region = candidates[0]
        selection_mode = "Single Candidate"

    # --- 5. Calculate Final Values ---
    if best_region['is_wrapped']:
        seg1 = best_region['segments'][0] 
        seg2 = best_region['segments'][1] 
        
        chunk1 = potential_data[:, :, seg1['start']:seg1['end']]
        chunk2 = potential_data[:, :, seg2['start']:seg2['end']]
        
        vacuum_potential = (np.sum(chunk1) + np.sum(chunk2)) / (chunk1.size + chunk2.size)
        v_start_label = f"Wrapped ({seg1['start']}-{seg1['end']} & {seg2['start']}-{seg2['end']})"
    else:
        v_start = best_region['start']
        v_end = best_region['end']
        vacuum_potential = np.mean(potential_data[:, :, v_start:v_end])
        v_start_label = f"{v_start}-{v_end}"

    work_function = vacuum_potential - ef

    # --- Plotting ---
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 1. Potential Plot
        ax1.plot(z_coords, z_potential, 'b-', label='Planar Averaged Potential')
        ax1.axhline(ef, color='k', linestyle='--', label=f'Fermi Energy ({ef:.2f} eV)')

        # Draw Selection
        if best_region['is_wrapped']:
            s1, s2 = best_region['segments']
            ax1.axvspan(s1['start'], s1['end'], color='green', alpha=0.3, label='Selected Vacuum')
            ax1.axvspan(s2['start'], s2['end'], color='green', alpha=0.3)
        else:
            ax1.axvspan(best_region['start'], best_region['end'], color='green', alpha=0.3, label='Selected Vacuum')

        # Highlight rejected candidates (Grey)
        for r in candidates:
            if r != best_region:
                if r['is_wrapped']:
                    s1, s2 = r['segments']
                    ax1.axvspan(s1['start'], s1['end'], color='gray', alpha=0.1)
                    ax1.axvspan(s2['start'], s2['end'], color='gray', alpha=0.1)
                else:
                    ax1.axvspan(r['start'], r['end'], color='gray', alpha=0.1)

        ax1.set_ylabel("Potential (eV)")
        ax1.set_title(f"Vac: {vacuum_potential:.4f} eV | Ef: {ef:.4f} eV | WF: {work_function:.4f} eV\nMode: {selection_mode}")
        ax1.legend(loc='lower left')
        
        # 2. Curvature Plot
        ax2.plot(z_coords, curvatures, 'r-', label='Curvature |d²V/dz²|')
        ax2.axhline(curvature_tol, color='k', linestyle='--', label='Tolerance')
        ax2.set_ylabel("Change in Slope")
        ax2.set_xlabel("Grid Points (z)")
        ax2.set_yscale('log')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    # --- Console Output ---
    if verbose:
        print(f"Dipole Found: {'Yes' if dipole_found else 'No'}")
        print(f"Selection Mode: {selection_mode}")
        print(f"Selected Region: {v_start_label}")
        print(f"Vacuum Potential: {vacuum_potential:.4f} eV")
        print(f"Fermi Energy: {ef:.4f} eV")
        print(f"Work Function: {work_function:.4f} eV")
    
    return ef, vacuum_potential, work_function

# --- Execution ---
if __name__ == "__main__":
    # 1. Define the directory containing your VASP output files
    # Note: Use the directory path rather than the specific file path for the Fermi search
    target_dir = r'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au/adatom_surface/dpl/kp551'
    locpot_path = os.path.join(target_dir, 'LOCPOT')
    
    if os.path.exists(locpot_path):
        # 2. Load the LOCPOT file
        locpot = load_locpot(locpot_path)
        
        # 3. Retrieve the Fermi Energy (ef) using the robust fallback logic
        # This will look for DOSCAR first, then OUTCAR in the target_dir
        ef = get_fermi_energy(target_dir)
        
        # 4. Execute the work function calculation
        # No changes needed to this function call
        calculate_work_function(locpot, ef, dipole_threshold=0.15, min_width=10, plot=True, verbose=True)
    else:
        print(f"File not found: {locpot_path}")