import numpy as np

class PyramidFilter:
    def __init__(self, lattice, apex_coords, base_coords):
        """
        Initialize the PyramidFilter class with the lattice, apex coordinates,
        and base coordinates.
        """
        self.lattice = lattice
        self.apex_coords = np.array(apex_coords)  # Apex coordinates (Cartesian)
        self.base_coords = np.array(base_coords)  # Base coordinates (Cartesian)

    def calculate_planes(self):
        """
        Define planes for the pyramid using the apex and the given base coordinates.
        """
        planes = []
        num_base_atoms = len(self.base_coords)
        for i in range(num_base_atoms):
            p1 = self.base_coords[i]  # Current base atom
            p2 = self.base_coords[(i + 1) % num_base_atoms]  # Next base atom (wraps around)
            # Calculate the normal vector for the plane
            v1 = p1 - self.apex_coords
            v2 = p2 - self.apex_coords
            normal = np.cross(v1, v2)
            norm_magnitude = np.linalg.norm(normal)
            if norm_magnitude == 0:  # Check for degenerate planes
                print(f"Warning: Degenerate plane detected with points: Apex={self.apex_coords}, P1={p1}, P2={p2}")
                continue  # Skip invalid planes
            normal /= norm_magnitude  # Normalize the vector
            # Calculate the plane's constant (D) from the equation Ax + By + Cz + D = 0
            constant = -np.dot(normal, self.apex_coords)
            planes.append({"normal": normal, "constant": constant})
        return planes

    def is_within_planes(self, atom, planes, tolerance=1e-1):
        """
        Check if an atom lies within the intersection of all planes,
        including atoms that lie exactly on the planes.
        """
        for plane in planes:
            normal = plane["normal"]
            constant = plane["constant"]
            # Calculate the signed distance from the atom to the plane
            distance = np.dot(normal, atom) + constant
            # Include atoms that lie on the plane (distance <= tolerance)
            if distance > tolerance:  # Atom is outside the plane
                return False
        return True

    def filter_atoms(self, atom_coords):
        """
        Filter atoms to retain only those within the pyramidal cross-section,
        and ensure apex and corner atoms are always included.
        """
        planes = self.calculate_planes()
        filtered_atoms = []
        
        # Filter based on pyramid cross-section
        for atom in atom_coords:  # Iterate over given atom coordinates
            if self.is_within_planes(atom, planes):  # Check if atom lies within the pyramid
                filtered_atoms.append(atom)
        
        # Ensure apex and base atoms are included
        apex, base_atoms = self.apex_coords, self.base_coords  # Use provided apex and base coordinates
        for fixed_atom in np.vstack([apex, base_atoms]):  # Combine apex and base atoms into a single array
            if not any(np.allclose(fixed_atom, atom) for atom in filtered_atoms):
                filtered_atoms.append(fixed_atom)  # Append missing apex or base atoms
    
        return np.array(filtered_atoms)

    @staticmethod
    def read_poscar(file_name):
        """
        Read a VASP POSCAR file and extract lattice, positions, and element counts.
        """
        with open(file_name, 'r') as f:
            lines = f.readlines()

        scale = float(lines[1].strip())
        lattice = np.array([list(map(float, line.split())) for line in lines[2:5]]) * scale
        elements = lines[5].split()
        counts = list(map(int, lines[6].split()))

        # Determine the start of atomic positions
        start_index = 7
        if "Selective" in lines[start_index]:  # Check for Selective Dynamics
            start_index += 1
        if "Direct" in lines[start_index] or "Cartesian" in lines[start_index]:
            start_index += 1

        # Extract atomic positions
        positions = [list(map(float, line.split()[:3])) for line in lines[start_index:start_index + sum(counts)]]
        return lattice, np.array(positions), elements, counts

    @staticmethod
    def write_poscar(file_name, lattice, positions, elements, counts):
        """
        Write filtered positions to a new POSCAR file.
        """
        with open(file_name, 'w') as f:
            f.write("Filtered Pyramid POSCAR (Using Given Coordinates)\n")
            f.write("1.0\n")
            for vec in lattice:
                f.write(f"{' '.join(map(str, vec))}\n")
            f.write(f"{' '.join(elements)}\n")
            f.write(f"{' '.join(map(str, counts))}\n")
            f.write("Direct\n")
            for pos in positions:
                f.write(f"{' '.join(map(str, pos))}\n")

# Example usage
poscar_file = 'C:/Users/Benjamin Kafin/Documents/VASP/Silver/POSCAR_tall_silver.vasp'  # Input POSCAR file
output_file = 'C:/Users/Benjamin Kafin/Documents/VASP/Silver/BigTip/POSCAR_FILTERED_skinny'  # Output POSCAR file

# Load data from POSCAR
lattice, positions, elements, counts = PyramidFilter.read_poscar(poscar_file)

# Define apex and base coordinates manually (Cartesian)
apex_coords = [5.87780,  10.18065,  57.59714]  # Example apex coordinates
base_coords = [  # Example base coordinates
    #[14.69451,   5.09033,  50.39711],
    [13.22505,  14.42259,  45.59899],
    #[5.87781,  20.36130,  50.39711],
    [-1.46945,  14.42259,  45.59899],
    #[-2.93890,   5.09033,  50.39711],
    [5.87780,   1.69678,  45.59899]
]

# Initialize PyramidFilter
pyramid_filter = PyramidFilter(lattice, apex_coords, base_coords)

# Filter atoms
filtered_positions = pyramid_filter.filter_atoms(np.dot(positions, lattice))  # Use atom positions in Cartesian
counts[-1] = len(filtered_positions)  # Update atom count for the last element

# Write filtered data to POSCAR
PyramidFilter.write_poscar(output_file, lattice, np.linalg.solve(lattice.T, filtered_positions.T).T, elements, counts)