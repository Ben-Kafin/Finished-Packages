# -*- coding: utf-8 -*-
"""
Created on Fri May  2 14:01:51 2025

@author: Benjamin Kafin
"""

import numpy as np
import matplotlib.pyplot as plt
from spym.io import rhksm4
from scipy.signal import savgol_filter
from scipy.integrate import simpson, cumulative_trapezoid
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, CheckButtons
#from mpl_toolkits.axes_grid1 import make_axes_locatable

class DidVLineAnalyzer:
    def __init__(self, ifile, chunk_index=0, cmap='jet', scalefactor=2.75, scan_mode='single', combine_scans=False):
        """
        Initialize the dI/dV line analyzer.
          ifile         : Path to an SM4 file.
          chunk_index   : Which full line (if more than one exists) for the dI/dV data.
          cmap          : Colormap for plotting.
          scalefactor   : Scale factor for converting positions.
          scan_mode     : 'single' (one spectrum per position) or 'combined' (forward/backward interleaved).
          combine_scans : Only used when scan_mode='combined'. If True, average forward and backward scans.
                          If False, keep forward and backward separate for side-by-side plotting.
        """
        self.ifile = ifile
        self.chunk_index = chunk_index
        self.cmap = cmap
        self.scalefactor = scalefactor
        self._user_scan_mode = scan_mode
        self.combine_scans = combine_scans

        # Load the SM4 file and extract dI/dV data.
        self.f = rhksm4.load(self.ifile)
        self.extract_data()

    def _find_spec_page(self, label):
        """
        Find a spectroscopy page index by its RHK_Label.
        Only considers pages whose RHK_PageTypeName is
        'RHK_PAGE_INTERACTIVE_SPECTRA' (i.e. spectroscopy channels,
        not imaging pages that may share the same label).

        Returns the integer page index, or None if not found.
        """
        for i, page in enumerate(self.f._pages):
            if (page.attrs.get('RHK_PageTypeName') == 'RHK_PAGE_INTERACTIVE_SPECTRA'
                    and page.attrs.get('RHK_Label') == label):
                return i
        return None

    def extract_data(self):
        """Extract the dI/dV (LIAcurrent) and I(V) current data, energy, and position."""
        size = int(np.shape(self.f._pages[0].data)[0])
        pos = np.array(
            [self.f._pages[0].attrs['RHK_Xoffset'] + i * self.f._pages[0].attrs['RHK_Xscale']
             for i in range(size)]
        ) * 1.0e10  # convert position to Å
        pos = (pos / self.scalefactor) - (pos[0] / self.scalefactor)
        # Reverse and shift so that the first value is zero.
        self.pos = pos[::-1] - pos[::-1][0]

        npts = int(np.shape(self.f._pages[6].data)[0])
        self.energy = np.array(
            [self.f._pages[6].attrs['RHK_Xoffset'] + i * self.f._pages[6].attrs['RHK_Xscale']
             for i in range(npts)]
        )
        self.LIAcurrent = np.array(self.f._pages[6].data)  # shape: (npts, total_scans)
        total_scans = self.LIAcurrent.shape[1]

        # --- Locate and load the I(V) current spectroscopy channel ---
        iv_page_idx = self._find_spec_page('Current')
        if iv_page_idx is not None:
            self.iv_raw = np.array(self.f._pages[iv_page_idx].data)  # same shape as LIAcurrent
        else:
            self.iv_raw = None

        # --- Store bias / current setpoint from metadata ---
        self.V_bias = float(self.f._pages[6].attrs['RHK_Bias'])
        self.I_setpoint = float(self.f._pages[6].attrs['RHK_Current'])

        # --- Chunk the LIA data (and I(V) in parallel) ---
        if self._user_scan_mode == 'single':
            if total_scans % size != 0:
                raise ValueError(f"scan_mode='single' but total_scans ({total_scans}) is not a multiple of size ({size}).")
            n_chunks = total_scans // size
            if n_chunks == 1:
                self.chunk = self.LIAcurrent.copy()
                self.iv_chunk = self.iv_raw.copy() if self.iv_raw is not None else None
            else:
                if self.chunk_index < 0 or self.chunk_index >= n_chunks:
                    raise ValueError(f"chunk_index {self.chunk_index} out of range. Only {n_chunks} chunks available.")
                sl = slice(self.chunk_index * size, (self.chunk_index + 1) * size)
                self.chunk = self.LIAcurrent[:, sl]
                self.iv_chunk = self.iv_raw[:, sl] if self.iv_raw is not None else None
            self.scan_mode = 'single'
        elif self._user_scan_mode == 'combined':
            expected_scans = 2 * size
            if total_scans % expected_scans != 0:
                raise ValueError(f"scan_mode='combined' but total_scans ({total_scans}) is not a multiple of 2*size ({expected_scans}).")
            n_chunks = total_scans // expected_scans
            if self.chunk_index < 0 or self.chunk_index >= n_chunks:
                raise ValueError(f"chunk_index {self.chunk_index} out of range. Only {n_chunks} chunks available.")
            sl = slice(self.chunk_index * expected_scans, (self.chunk_index + 1) * expected_scans)
            lia_chunk = self.LIAcurrent[:, sl]
            iv_chunk_raw = self.iv_raw[:, sl] if self.iv_raw is not None else None
            if self.combine_scans:
                self.chunk = (lia_chunk[:, 0::2] + lia_chunk[:, 1::2]) / 2.0
                self.iv_chunk = ((iv_chunk_raw[:, 0::2] + iv_chunk_raw[:, 1::2]) / 2.0
                                 if iv_chunk_raw is not None else None)
                self.scan_mode = 'combined'
            else:
                self.forward = lia_chunk[:, 0::2]
                self.backward = lia_chunk[:, 1::2]
                if iv_chunk_raw is not None:
                    self.iv_forward = iv_chunk_raw[:, 0::2]
                    self.iv_backward = iv_chunk_raw[:, 1::2]
                else:
                    self.iv_forward = None
                    self.iv_backward = None
                self.scan_mode = 'separate'
                self.chunk = None
                self.iv_chunk = None
        else:
            raise ValueError(f"scan_mode must be 'single' or 'combined' (got '{self._user_scan_mode}').")

    def apply_hampel_filter(self, curves, window=3, n_sigma=2.5):
        """Apply a Hampel filter to each column (spectrum) in curves."""
        if window != int(window):
            raise ValueError(
                f"Hampel filter 'window' must be a whole number (got {window}). "
                "Window defines the number of neighboring data points on each side "
                "and cannot be fractional."
            )
        filtered = np.zeros_like(curves)
        for j in range(curves.shape[1]):
            current_curve = curves[:, j]
            filtered_curve = current_curve.copy()
            k = int(window)
            for i in range(len(current_curve)):
                lower = max(0, i - k)
                upper = min(len(current_curve), i + k + 1)
                window_data = current_curve[lower:upper]
                med = np.median(window_data)
                mad = np.median(np.abs(window_data - med))
                threshold = n_sigma * 1.4826 * mad if mad != 0 else 0
                if np.abs(current_curve[i] - med) > threshold:
                    filtered_curve[i] = med
            filtered[:, j] = filtered_curve
        return filtered

    def apply_sgolay_filter(self, curves, window_length=5, polyorder=3):
        """Apply a Savitzky–Golay filter to each column (spectrum) in curves."""
        if window_length % 2 == 0:
            window_length += 1
        return savgol_filter(curves, window_length, polyorder, axis=0)

    def calibrate_didv_curves(self, curves, iv_data=None):
        """
        Calibrate dI/dV (LIA) curves via per-curve linear regression against
        the co-acquired I(V) current channel.

        Physics
        -------
        The true dI/dV relates to the raw LIA signal by:

            true dI/dV(V) = a · LIA(V) + b

        Integrating both sides from 0 V to V:

            I(V) − I(0) = a · X(V) + b · V

        where X(V) ≡ ∫₀ᵛ LIA(V') dV'  is the cumulative integral of the
        experimental LIA signal (computed via cumulative trapezoidal rule).

        For each spectrum j we minimise the least-squares cost

            S(a, b) = Σₙ [ΔI(Vₙ) − a·X(Vₙ) − b·Vₙ]²

        Setting ∂S/∂a = 0 and ∂S/∂b = 0 gives the normal equations:

            ΣΔI·X = a·ΣX² + b·ΣV·X
            ΣΔI·V = a·ΣX·V + b·ΣV²

        In matrix form:

            M · [a; b] = [ΣΔI·X ; ΣΔI·V]

            M = [[ΣX²,  ΣVX],
                 [ΣXV,  ΣV²]]

            [a; b] = M⁻¹ · [ΣΔI·X ; ΣΔI·V]

        The calibrated dI/dV is then  a_j · LIA_j(V) + b_j, which
        simultaneously recovers the scale and offset in a single fit.

        Parameters
        ----------
        curves : ndarray, shape (npts, N)
            The (optionally filtered) LIA data in raw-count column ordering,
            i.e. *before* the spatial-reversal / transpose done for plotting.
        iv_data : ndarray or None
            Matching I(V) data with the same shape and column ordering.
            If None, self.iv_chunk is used (covers single / combined modes).

        Returns
        -------
        calibrated : ndarray, same shape as *curves*
        """
        if iv_data is None:
            iv_data = self.iv_chunk
        if iv_data is None:
            raise ValueError(
                "No I(V) current channel found in this SM4 file. "
                "Cannot perform linear-regression calibration."
            )

        # Sort energy ascending for cumulative integration.
        sort_idx = np.argsort(self.energy)
        e_sorted = self.energy[sort_idx]
        idx_0V = int(np.argmin(np.abs(e_sorted)))

        n_curves = curves.shape[1]
        calibrated = np.empty_like(curves, dtype=np.float64)

        for j in range(n_curves):
            # Sort LIA and I(V) to match ascending energy.
            lia_sorted = curves[sort_idx, j]
            iv_sorted  = iv_data[sort_idx, j]

            # X(V) = cumulative integral of LIA, referenced to 0 V.
            X_raw = cumulative_trapezoid(lia_sorted, e_sorted, initial=0)
            X = X_raw - X_raw[idx_0V]

            # ΔI(V) = I(V) − I(0 V)
            delta_I = iv_sorted - iv_sorted[idx_0V]

            # Voltage array.
            V = e_sorted

            # Assemble normal equations:  M · [a; b] = rhs
            sum_X2 = np.dot(X, X)
            sum_V2 = np.dot(V, V)
            sum_VX = np.dot(V, X)
            sum_IX = np.dot(delta_I, X)
            sum_IV = np.dot(delta_I, V)

            M   = np.array([[sum_X2, sum_VX],
                            [sum_VX, sum_V2]])
            rhs = np.array([sum_IX, sum_IV])

            # Solve for [a, b].
            ab  = np.linalg.solve(M, rhs)
            a_j = ab[0]
            b_j = ab[1]

            # Calibrated dI/dV in original energy ordering.
            calibrated[:, j] = a_j * curves[:, j] + b_j

        return calibrated

    def normalize_curves(self, curves):
        """Normalize each spectrum (column) so its integrated area (via Simpson's rule) is 1."""
        norm_curves = np.zeros_like(curves, dtype=np.float64)
        eps = 1e-8  # tolerance to avoid division by near-zero
        for j in range(curves.shape[1]):
            current_curve = curves[:, j]
            area = np.abs(simpson(current_curve, self.energy))
            if area > eps:
                norm_curves[:, j] = current_curve / area
            else:
                norm_curves[:, j] = current_curve
        return norm_curves

    def plot_line(self, hampel_filter_params=None, sgolay_filter_params=None,
                  apply_calibration=False, apply_normalize=False):
        """
        Process and plot the dI/dV line scan.
        Optional filtering, linear-regression calibration, and area
        normalization are applied.

        Parameters:
          hampel_filter_params : dict for the Hampel filter (or None to skip).
          sgolay_filter_params : dict for the Savitzky–Golay filter (or None to skip).
          apply_calibration    : If True, calibrate each dI/dV curve via per-curve
                                 linear regression against the I(V) current channel.
          apply_normalize      : If True, normalize each spectrum to unit area.
        """
        e_left = min(self.energy[0], self.energy[-1])
        e_right = max(self.energy[0], self.energy[-1])
        if self.scan_mode in ['single', 'combined']:
            curves = self.chunk.copy()   # shape: (npts, N)
            if hampel_filter_params is not None:
                curves = self.apply_hampel_filter(curves, **hampel_filter_params)
            if sgolay_filter_params is not None:
                curves = self.apply_sgolay_filter(curves, **sgolay_filter_params)
            if apply_calibration:
                curves = self.calibrate_didv_curves(curves)
            if apply_normalize:
                curves = self.normalize_curves(curves)
            curves = curves[:, ::-1]
            curves = curves.T
            # Ensure energy increases left to right.
            if self.energy[0] > self.energy[-1]:
                curves = curves[:, ::-1]

            plt.figure(figsize=(8, 6))
            plt.imshow(curves, extent=[e_left, e_right,
                                       self.pos[0], self.pos[-1]],
                       aspect='auto', cmap=self.cmap)
            plt.colorbar(label="dI/dV signal")
            if self.scan_mode == 'combined':
                plt.title("dI/dV (Combined Forward/Backward Scans)")
            else:
                plt.title("dI/dV Line Scan")
            plt.xlabel("Energy (eV)")
            plt.ylabel("Position (Å)")
            plt.show()
        elif self.scan_mode == 'separate':
            fwd = self.forward.copy()
            bck = self.backward.copy()
            if hampel_filter_params is not None:
                fwd = self.apply_hampel_filter(fwd, **hampel_filter_params)
                bck = self.apply_hampel_filter(bck, **hampel_filter_params)
            if sgolay_filter_params is not None:
                fwd = self.apply_sgolay_filter(fwd, **sgolay_filter_params)
                bck = self.apply_sgolay_filter(bck, **sgolay_filter_params)
            if apply_calibration:
                fwd = self.calibrate_didv_curves(fwd, iv_data=self.iv_forward)
                bck = self.calibrate_didv_curves(bck, iv_data=self.iv_backward)
            if apply_normalize:
                fwd = self.normalize_curves(fwd)
                bck = self.normalize_curves(bck)
            fwd = fwd[:, ::-1]
            fwd = fwd.T
            if self.energy[0] > self.energy[-1]:
                fwd = fwd[:, ::-1]
            bck = bck[:, ::-1]
            bck = bck.T
            if self.energy[0] > self.energy[-1]:
                bck = bck[:, ::-1]

            plt.figure(figsize=(16, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(fwd, extent=[e_left, e_right,
                                     self.pos[0], self.pos[-1]],
                       aspect='auto', cmap=self.cmap)
            plt.colorbar(label="dI/dV (Forward)")
            plt.xlabel("Energy (eV)")
            plt.ylabel("Position (Å)")
            plt.title("dI/dV Forward Scans")

            plt.subplot(1, 2, 2)
            plt.imshow(bck, extent=[e_left, e_right,
                                     self.pos[0], self.pos[-1]],
                       aspect='auto', cmap=self.cmap)
            plt.colorbar(label="dI/dV (Backward)")
            plt.xlabel("Energy (eV)")
            plt.ylabel("Position (Å)")
            plt.title("dI/dV Backward Scans")
            plt.show()

    def extract_topography(self, topo_mode='forward', line_index=None):
        """
        Extract topography data from the SM4 file.
        
        Parameters:
          topo_mode: 'forward' -> use page 2,
                     'reverse' -> use page 3,
                     'average' -> average pages 2 and 3.
          line_index: (optional) if provided and the topo data is 2D,
                      this row is selected and returned as a 1D profile.
        
        Returns:
          If line_index is provided:
              (topo_x, topo_profile)
          Otherwise:
              (topo_x, topo_y, topo_data)
              
        The x-coordinates are computed from RHK_Xoffset and RHK_Xscale.
        The z-values (topography data) are scaled using the file attribute 'RHK_Zscale'.
        """
        if topo_mode == 'forward':
            page = self.f._pages[2]
        elif topo_mode == 'reverse':
            page = self.f._pages[3]
        elif topo_mode == 'average':
            if len(self.f._pages) < 4:
                raise ValueError("File does not contain both forward and reverse topo pages.")
            page_forward = self.f._pages[2]
            page_reverse = self.f._pages[3]
            data_forward = np.array(page_forward.data)
            data_reverse = np.array(page_reverse.data)
            topo_data = (data_forward + data_reverse) / 2.0
            page = page_forward
        else:
            raise ValueError("topo_mode must be 'forward', 'reverse', or 'average'.")

        # Determine the shape of the topography data.
        data_shape = np.shape(page.data)
        if len(data_shape) == 1:
            nx = data_shape[0]
            ny = 1
        else:
            ny, nx = data_shape

        # Compute coordinate arrays from the attributes.
        topo_x = np.array([page.attrs['RHK_Xoffset'] + i * page.attrs['RHK_Xscale']
                           for i in range(nx)]) * 1.0e10  # in Å
        topo_y = np.array([page.attrs['RHK_Yoffset'] + i * page.attrs['RHK_Yscale']
                           for i in range(ny)]) * 1.0e10
        topo_x = (topo_x / self.scalefactor) - (topo_x[0] / self.scalefactor)
        topo_y = (topo_y / self.scalefactor) - (topo_y[0] / self.scalefactor)

        # Get topo_data.
        if topo_mode in ['forward', 'reverse']:
            topo_data = np.array(page.data)
        # "average" already computed topo_data.

        # Use the file's Z scale.
        try:
            file_zscale = float(page.attrs['RHK_Zscale'])
        except KeyError:
            file_zscale = 1.0
        topo_data = (topo_data * file_zscale)*1E10

        # If line_index is provided, extract that row.
        if line_index is not None:
            if len(topo_data.shape) == 2:
                if line_index < 0 or line_index >= topo_data.shape[0]:
                    raise ValueError("line_index out of range.")
                topo_profile = topo_data[line_index, :]  # 1D array
                slope_fit = np.polyfit(topo_x, topo_profile, 1)
                topo_profile = topo_profile - np.polyval(slope_fit, topo_x)
                adjusted_topo = topo_profile - np.min(topo_profile)
                return topo_x, adjusted_topo
            else:
                # If data is not 2D, return as is.
                return topo_x, topo_data
        else:
            return topo_x, topo_y, topo_data




    def plot_line_with_topo(self, hampel_filter_params=None, sgolay_filter_params=None,
                            apply_calibration=False, apply_normalize=False,
                            topo_lines=None):
        """
        Plot the dI/dV line scan together with the topography profile.
        
        The dI/dV image appears in the right subplot (with Energy on the x-axis and Position on the y-axis).
        The topography is represented in the left, very narrow subplot as a single vertical stripe whose segment
        colors represent the topography height. The topo colorbar is placed flush to the far left of the entire image.
        The topo panel has no title, axis labels, or tick marks.
        
        Parameters:
          hampel_filter_params : dict for the Hampel filter.
          sgolay_filter_params : dict for the Savitzky–Golay filter.
          apply_calibration    : If True, calibrate dI/dV curves via per-curve
                                 linear regression against the I(V) current channel.
          apply_normalize      : If True, normalize each dI/dV spectrum to unit area.
          topo_lines           : List of (line_index, topo_mode) tuples.  Each profile is
                                 extracted raw, element-wise averaged, then slope-subtracted
                                 and min-shifted.  If None, defaults to the middle row of
                                 the forward topography page.
        """
        # --- Process dI/dV Line Scan Data ---
        e_left = min(self.energy[0], self.energy[-1])
        e_right = max(self.energy[0], self.energy[-1])
        if self.scan_mode in ['single', 'combined']:
            curves = self.chunk.copy()
            if hampel_filter_params is not None:
                curves = self.apply_hampel_filter(curves, **hampel_filter_params)
            if sgolay_filter_params is not None:
                curves = self.apply_sgolay_filter(curves, **sgolay_filter_params)
            if apply_calibration:
                curves = self.calibrate_didv_curves(curves)
            if apply_normalize:
                curves = self.normalize_curves(curves)
            curves = curves[:, ::-1]
            curves = curves.T
            if self.energy[0] > self.energy[-1]:
                curves = curves[:, ::-1]
        elif self.scan_mode == 'separate':
            curves = (self.forward + self.backward) / 2.0
            iv_avg = None
            if self.iv_forward is not None and self.iv_backward is not None:
                iv_avg = (self.iv_forward + self.iv_backward) / 2.0
            if hampel_filter_params is not None:
                curves = self.apply_hampel_filter(curves, **hampel_filter_params)
            if sgolay_filter_params is not None:
                curves = self.apply_sgolay_filter(curves, **sgolay_filter_params)
            if apply_calibration:
                curves = self.calibrate_didv_curves(curves, iv_data=iv_avg)
            if apply_normalize:
                curves = self.normalize_curves(curves)
            curves = curves[:, ::-1]
            curves = curves.T
            if self.energy[0] > self.energy[-1]:
                curves = curves[:, ::-1]
        else:
            raise ValueError("Unknown scan mode.")
        
        # --- Extract Topography Data ---
        try:
            if topo_lines is None:
                topo_lines = [(None, 'forward')]  # default: middle row, forward
            profiles = []
            topo_x = None
            for idx, mode in topo_lines:
                tx, ty, td = self.extract_topography(mode)
                if topo_x is None:
                    topo_x = tx
                row = idx if idx is not None else td.shape[0] // 2
                profiles.append(td[row, :])
            topo_profile = np.mean(profiles, axis=0)
            # Apply slope subtraction + min-shift.
            slope_fit = np.polyfit(topo_x, topo_profile, 1)
            topo_profile = topo_profile - np.polyval(slope_fit, topo_x)
            topo_profile = topo_profile - np.min(topo_profile)
        except Exception as e:
            print("Failed to extract topography:", e)
            return
            
        # Adjust topo_x so that it starts at zero.
        topo_x = topo_x - np.min(topo_x)
        
        # --- Create the Figure ---
        # Use a very narrow left panel for the topo stripe.
        fig, (ax_topo, ax_didv) = plt.subplots(1, 2, #figsize=(12, 6),
                                                gridspec_kw={'width_ratios': [0.175, 3]},
                                                sharey=True)
        # Remove extra space between subplots and at the figure margins.
        #fig.subplots_adjust( left=0.0, right=1.0,wspace=-0.1)
        
        # --- dI/dV Image (Right Subplot) ---
        im = ax_didv.imshow(curves, 
                            extent=[e_left, e_right,
                                    self.pos[0], self.pos[-1]],
                            aspect='auto', cmap=self.cmap)
        ax_didv.set_xlabel("Energy (eV)")
        #ax_didv.set_ylabel("Position (Å)")
        ax_didv.set_title("dI/dV Line Scan")
        # Move the dI/dV y-axis tick labels to the right.
        #ax_didv.yaxis.tick_right()
        plt.colorbar(im, ax=ax_didv, label="dI/dV signal")
        
        # --- Topography as a Vertical Stripe (Left Subplot) ---
        # Build cell edges that span exactly the dI/dV y-extent so the
        # topo stripe fills the panel without half-cells at top/bottom.
        # Reverse topo_profile to match self.pos ordering (both start at 0,
        # but the topo page's row order is spatially flipped).
        y_edges = np.linspace(self.pos[0], self.pos[-1], len(topo_profile) + 1)
        topo_flip = (self.V_bias > 0)
        if topo_flip:
            topo_profile = topo_profile[::-1]
        x_edges = np.array([0.0, 1.0])
        norm = mcolors.TwoSlopeNorm(vmin=np.min(topo_profile),
                                    vcenter=(np.min(topo_profile) + np.max(topo_profile)) / 2,
                                    vmax=np.max(topo_profile))
        custom_topo = mcolors.LinearSegmentedColormap.from_list("custom_topo", ["black","firebrick", "yellow"])
        ax_topo.pcolormesh(x_edges, y_edges, topo_profile[:, None],
                           cmap=custom_topo, norm=norm, shading='flat')
        
        # Tighten the left axis so that only the stripe is visible.
        ax_topo.set_xticks([])
        ax_topo.set_xticklabels([])
        ax_topo.set_ylabel("Position (Å)")
        ax_topo.set_title("Topo")
        #ax_topo.spines['top'].set_visible(False)
        #ax_topo.spines['bottom'].set_visible(False)
        
        # y-limits already matched via sharey + imshow extent.
        
        # --- Place the Topo Colorbar on the Left Side of the Entire Image ---
        #divider = make_axes_locatable(ax_topo)
        #cax = divider.append_axes("left", size="15%", pad=0.0)
        #plt.colorbar(pc, cax=cax, orientation='vertical', label="Height")
        
        plt.tight_layout()
        plt.show()


    # ------------------------------------------------------------------ #
    #  Topo processing helpers (extensible registry)                      #
    # ------------------------------------------------------------------ #

    def _topo_slope_sub(self, topo_x, topo_profile):
        """Linear slope subtraction."""
        slope_fit = np.polyfit(topo_x, topo_profile, 1)
        return topo_profile - np.polyval(slope_fit, topo_x)

    # ------------------------------------------------------------------ #
    #  Interactive redraw helpers                                          #
    # ------------------------------------------------------------------ #

    def _redraw_topo_stripe(self):
        """Recompute processed topo from raw + active processors, redraw stripe."""
        # --- Apply active processors in order ---
        profile = self._int_topo_raw.copy()
        for i, (label, func, _default) in enumerate(self._topo_processors):
            if self._int_topo_active[i]:
                profile = func(self._int_topo_x, profile)
        profile = profile - np.min(profile)
        self._int_topo_processed = profile

        # --- Redraw stripe ---
        ax = self._int_ax_topo
        ax.clear()

        y_edges = np.linspace(self._int_pos[0], self._int_pos[-1],
                              len(profile) + 1)
        x_edges = np.array([0.0, 1.0])
        norm = mcolors.TwoSlopeNorm(
            vmin=np.min(profile),
            vcenter=(np.min(profile) + np.max(profile)) / 2,
            vmax=np.max(profile))
        custom_topo = mcolors.LinearSegmentedColormap.from_list(
            "custom_topo", ["black", "firebrick", "yellow"])
        ax.pcolormesh(x_edges, y_edges, profile[:, None],
                      cmap=custom_topo, norm=norm, shading='flat')

        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylabel("Position (Å)")
        ax.set_title("Topo")

        # Marker overlays are drawn exclusively by _redraw_curves()
        # so they are tracked in _int_marker_artists for clean removal.
        self._redraw_curves()

    def _redraw_curves(self):
        """Redraw marker overlay lines on heatmap/topo and 1D curve panel."""
        # --- Remove old marker artists (Line2D only) ---
        for art in self._int_marker_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._int_marker_artists = []

        # --- Draw marker lines on heatmap ---
        for i, pos_val in enumerate(self._int_marker_pos):
            color = self._int_m_colors[i % len(self._int_m_colors)]
            h_didv = self._int_ax_didv.axhline(
                y=pos_val, color=color, ls='--', lw=2, alpha=0.7,
                picker=5, label=f'marker_{i}')
            self._int_marker_artists.append(h_didv)

        # --- Draw marker lines on topo stripe ---
        for i, pos_val in enumerate(self._int_marker_pos):
            color = self._int_m_colors[i % len(self._int_m_colors)]
            h_topo = self._int_ax_topo.axhline(
                y=pos_val, color=color, ls='--', lw=2, alpha=0.7,
                picker=5, label=f'marker_{i}')
            self._int_marker_artists.append(h_topo)

        # --- Redraw 1D curve panel ---
        ax = self._int_ax_curve
        ax.clear()
        if self._int_avg_on:
            spectra = []
            for pos_val in self._int_marker_pos:
                idx = np.argmin(np.abs(self._int_pos - pos_val))
                idx = len(self._int_pos) - 1 - idx
                spectra.append(self._int_curves_unfiltered[idx, :])
            avg_spectrum = np.mean(spectra, axis=0)
            # Filter the averaged (calibrated + normalized) curve, mirroring the
            # pipeline's filter guards. The filters are reversal-invariant per
            # point, so applying them in plotting (ascending-energy) order is valid.
            avg_col = avg_spectrum[:, None]
            if self._int_hampel_params is not None:
                avg_col = self.apply_hampel_filter(avg_col, **self._int_hampel_params)
            if self._int_sgolay_params is not None:
                avg_col = self.apply_sgolay_filter(avg_col, **self._int_sgolay_params)
            avg_spectrum = avg_col[:, 0]
            ax.plot(self._int_energy, avg_spectrum, color='black', lw=2,
                    label=f'Average ({len(self._int_marker_pos)} markers)')
        else:
            for i, pos_val in enumerate(self._int_marker_pos):
                color = self._int_m_colors[i % len(self._int_m_colors)]
                idx = np.argmin(np.abs(self._int_pos - pos_val))
                idx = len(self._int_pos) - 1 - idx
                spectrum = self._int_curves[idx, :]
                ax.plot(self._int_energy, spectrum, color=color, lw=2,
                        label=f'{pos_val:.1f} Å')
        ax.legend(loc='upper right', frameon=False, fontsize=8)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("dI/dV signal")
        ax.set_title("dI/dV at Markers")

        self._int_fig.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    #  Event handlers                                                     #
    # ------------------------------------------------------------------ #

    def _int_on_pick(self, event):
        """Identify which marker was clicked."""
        artist = event.artist
        label = getattr(artist, 'get_label', lambda: '')()
        if isinstance(label, str) and label.startswith('marker_'):
            idx = int(label.split('_')[1])
            if idx < len(self._int_marker_pos):
                self._int_active_obj = ('marker', idx)

    def _int_on_motion(self, event):
        """Drag the active marker vertically."""
        if self._int_active_obj is None or event.ydata is None:
            return
        if event.inaxes not in [self._int_ax_didv, self._int_ax_topo]:
            return
        _, idx = self._int_active_obj
        pos_min, pos_max = self._int_pos[0], self._int_pos[-1]
        new_pos = np.clip(event.ydata, pos_min, pos_max)
        self._int_marker_pos[idx] = new_pos
        self._redraw_curves()

    def _int_on_release(self, event):
        """End drag."""
        self._int_active_obj = None

    def _int_on_slider_change(self, val):
        """Adjust number of markers."""
        new_count = int(self._int_s_num_marks.val)
        old_count = len(self._int_marker_pos)
        if new_count > old_count:
            pos_min, pos_max = self._int_pos[0], self._int_pos[-1]
            pos_range = pos_max - pos_min
            new_positions = np.linspace(pos_min + 0.1 * pos_range,
                                        pos_max - 0.1 * pos_range,
                                        new_count)
            self._int_marker_pos = list(new_positions)
        elif new_count < old_count:
            self._int_marker_pos = self._int_marker_pos[:new_count]
        # Marker lines on both axes are managed by _redraw_curves()
        self._redraw_curves()

    def _int_on_topo_toggle(self, label):
        """Toggle a topo processing step and redraw the stripe."""
        for i, (proc_label, _func, _default) in enumerate(self._topo_processors):
            if proc_label == label:
                self._int_topo_active[i] = not self._int_topo_active[i]
                break
        self._redraw_topo_stripe()

    def _int_on_avg_toggle(self, label):
        """Toggle averaging of all marker spectra into a single curve."""
        self._int_avg_on = not self._int_avg_on
        self._redraw_curves()

    # ------------------------------------------------------------------ #
    #  Main interactive entry point                                       #
    # ------------------------------------------------------------------ #

    def plot_line_with_topo_interactive(self, hampel_filter_params=None,
                                        sgolay_filter_params=None,
                                        apply_calibration=False,
                                        apply_normalize=False,
                                        marker_positions=None,
                                        topo_lines=None):
        """
        Interactive version of plot_line_with_topo.

        Static panels: topo stripe + dI/dV heatmap (locked after initial draw).
        Interactive: draggable position markers whose 1D dI/dV spectra are shown
        in a third panel.  A slider controls marker count; CheckButtons toggle
        topo processing steps (slope subtraction, etc.).

        Parameters
        ----------
        Same as plot_line_with_topo, plus:
        marker_positions : list of float or None
            Initial marker positions in Å along the line axis.
            If None, two markers are placed at 25% and 75% of the range.
        """
        # ============================================================== #
        #  Phase 1: Pre-compute static data                              #
        # ============================================================== #

        # --- dI/dV curves (identical pipeline to plot_line_with_topo) ---
        e_left = min(self.energy[0], self.energy[-1])
        e_right = max(self.energy[0], self.energy[-1])

        if self.scan_mode in ['single', 'combined']:
            curves = self.chunk.copy()
            if hampel_filter_params is not None:
                curves = self.apply_hampel_filter(curves, **hampel_filter_params)
            if sgolay_filter_params is not None:
                curves = self.apply_sgolay_filter(curves, **sgolay_filter_params)
            if apply_calibration:
                curves = self.calibrate_didv_curves(curves)
            if apply_normalize:
                curves = self.normalize_curves(curves)
            curves = curves[:, ::-1]
            curves = curves.T
            if self.energy[0] > self.energy[-1]:
                curves = curves[:, ::-1]
        elif self.scan_mode == 'separate':
            curves = (self.forward + self.backward) / 2.0
            iv_avg = None
            if self.iv_forward is not None and self.iv_backward is not None:
                iv_avg = (self.iv_forward + self.iv_backward) / 2.0
            if hampel_filter_params is not None:
                curves = self.apply_hampel_filter(curves, **hampel_filter_params)
            if sgolay_filter_params is not None:
                curves = self.apply_sgolay_filter(curves, **sgolay_filter_params)
            if apply_calibration:
                curves = self.calibrate_didv_curves(curves, iv_data=iv_avg)
            if apply_normalize:
                curves = self.normalize_curves(curves)
            curves = curves[:, ::-1]
            curves = curves.T
            if self.energy[0] > self.energy[-1]:
                curves = curves[:, ::-1]
        else:
            raise ValueError("Unknown scan mode.")

        self._int_curves = curves                    # (n_positions, n_energies)
        self._int_energy = np.linspace(e_left, e_right, curves.shape[1])
        self._int_pos = self.pos                     # position axis in Å

        # --- Unfiltered (calibrated + normalized, NO Hampel/SG) curves for the ---
        # --- marker-average path: average these, then filter the averaged mean. ---
        # Same pipeline and orientation as self._int_curves, minus the two filters.
        self._int_hampel_params = hampel_filter_params
        self._int_sgolay_params = sgolay_filter_params
        if self.scan_mode in ['single', 'combined']:
            curves_uf = self.chunk.copy()
            if apply_calibration:
                curves_uf = self.calibrate_didv_curves(curves_uf)
            if apply_normalize:
                curves_uf = self.normalize_curves(curves_uf)
            curves_uf = curves_uf[:, ::-1]
            curves_uf = curves_uf.T
            if self.energy[0] > self.energy[-1]:
                curves_uf = curves_uf[:, ::-1]
        elif self.scan_mode == 'separate':
            curves_uf = (self.forward + self.backward) / 2.0
            iv_avg_uf = None
            if self.iv_forward is not None and self.iv_backward is not None:
                iv_avg_uf = (self.iv_forward + self.iv_backward) / 2.0
            if apply_calibration:
                curves_uf = self.calibrate_didv_curves(curves_uf, iv_data=iv_avg_uf)
            if apply_normalize:
                curves_uf = self.normalize_curves(curves_uf)
            curves_uf = curves_uf[:, ::-1]
            curves_uf = curves_uf.T
            if self.energy[0] > self.energy[-1]:
                curves_uf = curves_uf[:, ::-1]
        else:
            raise ValueError("Unknown scan mode.")

        self._int_curves_unfiltered = curves_uf      # (n_positions, n_energies)

        # --- Topography raw profile ---
        try:
            if topo_lines is None:
                topo_lines = [(None, 'forward')]  # default: middle row, forward
            profiles = []
            topo_x = None
            for idx, mode in topo_lines:
                tx, ty, td = self.extract_topography(mode)
                if topo_x is None:
                    topo_x = tx
                row = idx if idx is not None else td.shape[0] // 2
                profiles.append(td[row, :])
            topo_profile = np.mean(profiles, axis=0)
        except Exception as e:
            print("Failed to extract topography:", e)
            return

        topo_x = topo_x - np.min(topo_x)
        self._int_topo_x = topo_x
        self._int_topo_flip = (self.V_bias > 0)
        self._int_topo_raw = (topo_profile[::-1] if self._int_topo_flip else topo_profile).copy()

        # --- Topo processing registry ---
        self._topo_processors = [
            ("Slope Sub", self._topo_slope_sub, True),
        ]
        self._int_topo_active = [default for (_, _, default) in self._topo_processors]

        # Compute initial processed topo
        profile = self._int_topo_raw.copy()
        for i, (label, func, _default) in enumerate(self._topo_processors):
            if self._int_topo_active[i]:
                profile = func(self._int_topo_x, profile)
        profile = profile - np.min(profile)
        self._int_topo_processed = profile

        # ============================================================== #
        #  Phase 2: Initialize interactive state                          #
        # ============================================================== #

        self._int_m_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#00ced1',
                              '#e377c2', '#17becf', '#bcbd22', '#7f7f7f',
                              '#8c564b', '#d62728']

        pos_min, pos_max = self._int_pos[0], self._int_pos[-1]
        pos_range = pos_max - pos_min
        if marker_positions is not None:
            self._int_marker_pos = [np.clip(p, pos_min, pos_max)
                                    for p in marker_positions]
        else:
            self._int_marker_pos = list(
                np.linspace(pos_min + 0.1 * pos_range,
                            pos_max - 0.1 * pos_range, 2))

        self._int_active_obj = None
        self._int_marker_artists = []
        self._int_avg_on = False

        # ============================================================== #
        #  Phase 3: Build figure layout                                   #
        # ============================================================== #

        fig = plt.figure(figsize=(16, 7))
        self._int_fig = fig

        gs_main = gridspec.GridSpec(2, 1, height_ratios=[1, 0.001], hspace=0.3)
        gs_top = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=gs_main[0],
            width_ratios=[0.175, 3, 0.15, 0.1, 1.5], wspace=0.05)

        self._int_ax_topo = fig.add_subplot(gs_top[0])
        self._int_ax_didv = fig.add_subplot(gs_top[1], sharey=self._int_ax_topo)
        self._int_cax_didv = fig.add_subplot(gs_top[2])
        ax_spacer = fig.add_subplot(gs_top[3])
        ax_spacer.axis('off')
        self._int_ax_curve = fig.add_subplot(gs_top[4])

        # Widget axes (manual placement)
        self._int_s_num_marks = Slider(
            plt.axes([0.15, 0.02, 0.12, 0.03]), 'Markers',
            1, 10, valinit=len(self._int_marker_pos), valstep=1)

        topo_labels = [label for (label, _, _) in self._topo_processors]
        topo_defaults = self._int_topo_active
        self._int_chk_topo = CheckButtons(
            plt.axes([0.35, 0.01, 0.12, 0.05]),
            topo_labels, topo_defaults)

        self._int_chk_avg = CheckButtons(
            plt.axes([0.50, 0.01, 0.12, 0.05]),
            ['Average'], [self._int_avg_on])

        # ============================================================== #
        #  Phase 4: Draw static content                                   #
        # ============================================================== #

        # --- dI/dV heatmap (static) ---
        im = self._int_ax_didv.imshow(
            self._int_curves,
            extent=[e_left, e_right, self._int_pos[0], self._int_pos[-1]],
            aspect='auto', cmap=self.cmap)
        fig.colorbar(im, cax=self._int_cax_didv, label="dI/dV signal")
        self._int_ax_didv.set_xlabel("Energy (eV)")
        self._int_ax_didv.set_title("dI/dV Line Scan")

        # --- Topo stripe + marker overlays + 1D curves ---
        # _redraw_topo_stripe chains to _redraw_curves at the end.
        self._redraw_topo_stripe()

        # ============================================================== #
        #  Phase 8: Wire events and show                                  #
        # ============================================================== #

        fig.canvas.mpl_connect('pick_event', self._int_on_pick)
        fig.canvas.mpl_connect('motion_notify_event', self._int_on_motion)
        fig.canvas.mpl_connect('button_release_event', self._int_on_release)
        self._int_s_num_marks.on_changed(self._int_on_slider_change)
        self._int_chk_topo.on_clicked(self._int_on_topo_toggle)
        self._int_chk_avg.on_clicked(self._int_on_avg_toggle)

        plt.show()


# Sample run code:
if __name__ == '__main__':
    # Replace 'example.sm4' with the correct path.
    #analyzer = DidVLineAnalyzer('C:/Users/Benjamin Kafin/Downloads/NHC-iPr_Au_base8_CT_2024_07_12_17_01_07_799.sm4', chunk_index=0, cmap='jet', scalefactor=2.75, scan_mode='single', combine_scans=False)
    analyzer = DidVLineAnalyzer('C:/Users/Benjamin Kafin/Downloads/NHC-iPr_Au_base4_CT_2023_12_29_10_33_32_074.sm4', chunk_index=1, cmap='jet', scalefactor=2.75, scan_mode='single', combine_scans=False)    
    # Static plot (original).
    '''
    analyzer.plot_line_with_topo(hampel_filter_params={'window': 3, 'n_sigma': 2.5},
                                  sgolay_filter_params={'window_length': 5, 'polyorder': 3},
                                  apply_calibration=True,
                                  apply_normalize=False,
                                  topo_lines=[(0, 'forward')])  # or e.g. [(0, 'forward'), (1, 'reverse')]
    '''
    
    # Interactive plot with draggable markers.
    analyzer.plot_line_with_topo_interactive(
        hampel_filter_params={'window': 3, 'n_sigma': 2},
        sgolay_filter_params={'window_length': 6, 'polyorder': 3},
        apply_calibration=True,
        apply_normalize=True,
        marker_positions=None,              # or e.g. [5.0, 15.0, 25.0]
        #topo_lines=[(0, 'forward'),(0, 'reverse')])
        topo_lines=[(0, 'forward'),(1, 'forward')])        # or e.g. [(0, 'forward'), (1, 'reverse')]