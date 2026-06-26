import re
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from os.path import exists, join
import matplotlib.colors as mc, colorsys
import mplcursors

from doscar_parser import SpinAwareDosParser, SpinMode


class DosPlotter:
    def __init__(self, directory):
        self.directory = directory
        self.doscar = join(directory, 'DOSCAR')
        self.poscar = join(directory, 'CONTCAR') if exists(join(directory, 'CONTCAR')) else join(directory, 'POSCAR')

        if not exists(self.doscar):
            raise FileNotFoundError(f"DOSCAR not found in {directory}")

        self.total_dos = np.array([])
        self.site_dos = np.array([])
        self.mag_dos = None
        self.energies = np.array([])
        self.ef = 0.0
        self.spin_mode = SpinMode.COLLINEAR_UNPOL

        self.orbitals = []
        self.atomtypes = []
        self.atomnums = []
        self.vesta_label_map = {}
        self.element_dos = {}
        self.molecule_dos = None

        # Toggled at runtime in SOC mode (level 4) to overlay |m|
        self.show_mag = False

        self._type_color_map = {
            'Au': 'orange',
            'N': 'blue',
            'C': 'brown',
            'H': 'grey'
        }

        self._parse_all()

    # ------------------------------------------------------------------
    # Orbital label mappings
    # ------------------------------------------------------------------
    # ISPIN=1 / NCL_SOC (after stride extraction): n_orbitals columns
    #   3  → l-decomposed
    #   9  → lm-decomposed spd
    #   16 → lm-decomposed spdf
    # ISPIN=2: interleaved up/dn already split by parser → half columns
    #   6  → 3 orbs  (l-decomposed)
    #   18 → 9 orbs  (lm spd)
    #   32 → 16 orbs (lm spdf)
    _ORBITAL_LABELS = {
        3:  ['s', 'p', 'd'],
        9:  ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2'],
        16: ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2',
             'fy3x2', 'fxyz', 'fyz2', 'fz3', 'fxz2', 'fzx2', 'fx3'],
    }

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _orbit_base(self, orb):
        if orb.endswith('_up'):
            return orb[:-3]
        elif orb.endswith('_down'):
            return orb[:-5]
        return orb

    def _get_element_by_index(self, a):
        curr = 0
        for idx, count in enumerate(self.atomnums):
            if a <= curr + count:
                return self.atomtypes[idx]
            curr += count
        return 'grey'

    def _parse_doscar(self):
        """Use SpinAwareDosParser then reshape into the arrays the plotter expects."""
        parser = SpinAwareDosParser(self.doscar)
        self.energies = parser.energies          # already Fermi-shifted
        self.ef = parser.ef
        self.spin_mode = parser.spin_mode

        # site_dos: (atoms, nedos, n_orbitals) — charge-density DOS
        self.site_dos = parser.spin_up_dos

        # For ISPIN=2 we also need spin-down
        self.spin_down_site_dos = parser.spin_down_dos

        # For NCL/SOC keep |m| DOS
        self.mag_dos = parser.mag_dos

        # --- Reconstruct total_dos from the raw file (first block) ---
        # Re-read just the total-DOS block (parser doesn't expose it)
        with open(self.doscar, 'r') as f:
            int(f.readline().split()[0])
            for _ in range(4):
                f.readline()
            header = f.readline().split()
            nedos = int(header[2])
            total_dos_list = []
            for _ in range(nedos):
                line = [float(x) for x in f.readline().split()]
                total_dos_list.append(line[1:])
        self.total_dos = np.array(total_dos_list)

        # --- Orbital labels ---
        n_orbs = self.site_dos.shape[2]

        if self.spin_mode == SpinMode.COLLINEAR_POL:
            # Parser already split up/dn; n_orbs is per-spin count
            base_labels = self._ORBITAL_LABELS.get(n_orbs, [])
            self.orbitals = []
            for lbl in base_labels:
                self.orbitals.append(f"{lbl}_up")
                self.orbitals.append(f"{lbl}_down")
        else:
            # COLLINEAR_UNPOL or NCL_SOC — same orbital count after extraction
            self.orbitals = self._ORBITAL_LABELS.get(n_orbs, [])

    def _parse_poscar(self):
        with open(self.poscar, 'r') as file:
            lines = file.readlines()
            self.atomtypes = lines[5].split()
            self.atomnums = [int(i) for i in lines[6].split()]

    def _parse_all(self):
        self._parse_doscar()
        self._parse_poscar()
        current_global = 1
        mol_sum = np.zeros_like(self.energies)
        for idx, t in enumerate(self.atomtypes):
            atom_indices = range(current_global - 1, current_global - 1 + self.atomnums[idx])
            e_dos = np.sum(self.site_dos[atom_indices, :, :], axis=(0, 2))
            self.element_dos[t] = e_dos
            if t != 'Au':
                mol_sum += e_dos
            for n_rel in range(1, self.atomnums[idx] + 1):
                self.vesta_label_map[current_global] = f"{t}{n_rel}"
                current_global += 1
        self.molecule_dos = mol_sum

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _lighten_color(self, color, amount=0.3):
        c = mc.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*c)
        return colorsys.hls_to_rgb(h, min(1, l + amount * (1 - l)), s)

    def _site_dos_for_atom(self, a_idx_0based, orbital_col=None):
        """
        Return DOS array for a given atom (0-based index).
        For ISPIN=2 this sums up+dn to give total charge DOS.
        For NCL_SOC this is already total ρ.
        If orbital_col is given, return that single column.
        """
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            combined = self.site_dos[a_idx_0based] + self.spin_down_site_dos[a_idx_0based]
        else:
            combined = self.site_dos[a_idx_0based]

        if orbital_col is not None:
            return combined[:, orbital_col]
        return combined

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_dos_cursors(self, nums=None, types=None):
        fig, ax = plt.subplots()
        self.plot_level = 0
        self.active_element, self.active_atom = None, None

        # For orbital linestyles, use the base (spin-stripped) names
        unique_bases = sorted(set(self._orbit_base(o) for o in self.orbitals))
        styles = ['-', '--', ':', '-.'] + [(0, (3 + i, 2)) for i in range(max(0, len(unique_bases) - 4))]
        linestyle_map = dict(zip(unique_bases, styles))

        is_soc = (self.spin_mode == SpinMode.NCL_SOC)

        def _rescale_y_axis():
            lines = ax.get_lines()
            active_maxes = [np.max(l.get_ydata()) for l in lines if l.get_visible() and l.get_alpha() == 1.0]
            if active_maxes:
                ax.set_ylim(0, max(active_maxes) * 1.1)

        def update_plot_visuals():
            ax.cla()
            S = 0.25

            if self.plot_level == 0:
                total_y = (np.sum(self.total_dos[:, :int(self.total_dos.shape[1] / 2)], axis=1)
                           if self.total_dos.shape[1] > 1 else self.total_dos[:, 0])
                ax.plot(self.energies, total_y, color='black', lw=2.5,
                        label='Total DOS', picker=True, pickradius=5)
                ax.legend(loc='upper right', frameon=False)

            elif self.plot_level == 1:
                total_y = (np.sum(self.total_dos[:, :int(self.total_dos.shape[1] / 2)], axis=1)
                           if self.total_dos.shape[1] > 1 else self.total_dos[:, 0])
                ax.plot(self.energies, total_y, color='black', lw=1.5, alpha=0.1, zorder=1)
                ax.plot(self.energies, self.element_dos['Au'], color='orange', lw=2,
                        label='Au', picker=True, pickradius=5, zorder=2)
                ax.plot(self.energies, self.molecule_dos, color='black', lw=2,
                        label='Molecule', picker=True, pickradius=5, zorder=2)
                proxies = [Line2D([0], [0], color='orange', lw=2),
                           Line2D([0], [0], color='black', lw=2)]
                ax.legend(proxies, ['Au', 'Molecule'], title="Partition",
                          loc='upper right', frameon=False)

            elif self.plot_level == 2:
                for t in self.atomtypes:
                    ax.plot(self.energies, self.element_dos[t],
                            color=self._type_color_map.get(t, 'grey'),
                            lw=2, label=t, picker=True, pickradius=3)
                proxies = [Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2)
                           for t in self.atomtypes]
                ax.legend(proxies, self.atomtypes, title="Atom Types",
                          loc='upper right', frameon=False)

            elif self.plot_level == 3:
                ax.plot(self.energies, self.element_dos[self.active_element],
                        color=self._type_color_map.get(self.active_element, 'grey'),
                        lw=2, alpha=0.15, zorder=1)
                for a_idx, label in self.vesta_label_map.items():
                    if label.startswith(self.active_element):
                        y_sum = np.sum(self._site_dos_for_atom(a_idx - 1), axis=1)
                        ax.plot(self.energies, y_sum,
                                color=self._type_color_map.get(self.active_element, 'grey'),
                                lw=2, label=label, picker=True, pickradius=3, zorder=2)
                proxies = [Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2)
                           for t in self.atomtypes]
                ax.legend(proxies, self.atomtypes, title="Atom Types",
                          loc='upper right', frameon=False)

            elif self.plot_level == 4:
                element_color = self._type_color_map.get(self.active_element, 'grey')

                for a_idx, label in self.vesta_label_map.items():
                    if not label.startswith(self.active_element):
                        continue
                    y_sum = np.sum(self._site_dos_for_atom(a_idx - 1), axis=1)

                    if a_idx == self.active_atom:
                        ax.plot(self.energies, y_sum, color=element_color,
                                lw=2.5, alpha=1.0, zorder=5)
                        orb_artists = []

                        if self.spin_mode == SpinMode.COLLINEAR_POL:
                            # Interleaved up/dn orbital labels
                            for i_orb, orb in enumerate(self.orbitals):
                                spin_tag = orb.split('_')[-1]  # 'up' or 'down'
                                base = self._orbit_base(orb)
                                col_idx = i_orb // 2  # map to per-spin column
                                if spin_tag == 'up':
                                    y_orb = self.site_dos[a_idx - 1, :, col_idx]
                                else:
                                    y_orb = self.spin_down_site_dos[a_idx - 1, :, col_idx]
                                ls = linestyle_map[base]
                                p_color = (self._lighten_color(element_color, 0.3)
                                           if spin_tag == 'up' else element_color)
                                o_line, = ax.plot(self.energies, y_orb, color=p_color,
                                                  linestyle=ls, lw=1.2,
                                                  label=f"{label} – {orb}", zorder=10)
                                orb_artists.append(o_line)

                        else:
                            # COLLINEAR_UNPOL or NCL_SOC
                            for col_idx, orb in enumerate(self.orbitals):
                                y_orb = self.site_dos[a_idx - 1, :, col_idx]
                                ls = linestyle_map[self._orbit_base(orb)]
                                o_line, = ax.plot(self.energies, y_orb,
                                                  color=element_color,
                                                  linestyle=ls, lw=1.2,
                                                  label=f"{label} – {orb}", zorder=10)
                                orb_artists.append(o_line)

                            # SOC: overlay |m| per orbital as dashed-lighter traces
                            if is_soc and self.show_mag and self.mag_dos is not None:
                                mag_color = self._lighten_color(element_color, 0.4)
                                for col_idx, orb in enumerate(self.orbitals):
                                    y_mag = self.mag_dos[a_idx - 1, :, col_idx]
                                    ls = linestyle_map[self._orbit_base(orb)]
                                    m_line, = ax.plot(self.energies, y_mag,
                                                      color=mag_color,
                                                      linestyle=ls, lw=1.0,
                                                      alpha=0.7,
                                                      label=f"{label} – |m| {orb}",
                                                      zorder=9)
                                    orb_artists.append(m_line)

                        cursor = mplcursors.cursor(orb_artists, hover=True)
                        cursor.connect("add",
                                       lambda sel: sel.annotation.set_text(
                                           sel.artist.get_label()))
                    else:
                        orig = mc.to_rgb(element_color)
                        lumi = 0.299 * orig[0] + 0.587 * orig[1] + 0.114 * orig[2]
                        faded_color = (S * np.array(orig)) + ((1 - S) * lumi)
                        ax.plot(self.energies, y_sum, color=faded_color,
                                lw=1.5, alpha=0.05, zorder=2)

                # Legends
                atom_proxies = [Line2D([0], [0],
                                       color=self._type_color_map.get(t, 'grey'), lw=2)
                                for t in self.atomtypes]
                leg1 = ax.legend(atom_proxies, self.atomtypes,
                                 title="Atom Types", loc='upper right', frameon=False)
                ax.add_artist(leg1)

                orb_legend_entries = list(unique_bases)
                orb_legend_proxies = [Line2D([0], [0], color='black',
                                             linestyle=linestyle_map[b], lw=1.5)
                                      for b in unique_bases]
                if is_soc:
                    mag_state = "ON" if self.show_mag else "OFF"
                    orb_legend_entries.append(f"|m| overlay: {mag_state}  [press 'm']")
                    orb_legend_proxies.append(Line2D([0], [0], color='grey',
                                                     linestyle='--', lw=1.0, alpha=0.7))

                ax.legend(orb_legend_proxies, orb_legend_entries,
                          title="Orbitals", loc='upper left', frameon=False)

            mode_tag = {SpinMode.COLLINEAR_UNPOL: "",
                        SpinMode.COLLINEAR_POL: " (spin-polarized)",
                        SpinMode.NCL_SOC: " (SOC)"}
            ax.set_xlabel('energy – $E_f$ / eV')
            ax.set_ylabel('DOS / states eV⁻¹')
            ax.set_title(f"DOS{mode_tag.get(self.spin_mode, '')}")
            _rescale_y_axis()
            fig.canvas.draw_idle()

        def on_pick(event):
            label = event.artist.get_label()
            if self.plot_level == 0:
                self.plot_level = 1
            elif self.plot_level == 1:
                self.plot_level = 2
            elif self.plot_level == 2:
                self.active_element = label
                self.plot_level = 3
            elif self.plot_level == 3:
                self.active_atom = next(k for k, v in self.vesta_label_map.items() if v == label)
                self.active_element = self._get_element_by_index(self.active_atom)
                self.plot_level = 4
            update_plot_visuals()

        def on_click(event):
            if event.inaxes != ax:
                return
            if fig.canvas.manager.toolbar.mode == "" and not event.dblclick:
                hit = any(l.contains(event)[0] for l in ax.get_lines() if l.get_picker())
                if not hit:
                    self.plot_level = max(0, self.plot_level - 1)
                    update_plot_visuals()

        def on_key(event):
            # 'm' toggles |m| overlay in SOC mode at level 4
            if event.key == 'm' and is_soc and self.plot_level == 4:
                self.show_mag = not self.show_mag
                update_plot_visuals()

        fig.canvas.mpl_connect('pick_event', on_pick)
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        update_plot_visuals()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    v_dir = r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/SOC'
    plotter = DosPlotter(v_dir)
    plotter.plot_dos_cursors()
