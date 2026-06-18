import numpy as np
from matplotlib.path import Path

class PyramidFilter:
    def __init__(self, lattice, apex_coords, base_coords):
        """
        Initialize the PyramidFilter class with the lattice, apex coordinates,
        and base coordinates.
        """
        self.lattice = lattice
        self.apex_coords = np.array(apex_coords)  # Apex coordinates (Cartesian)
        self.base_coords = np.array(base_coords)  # Base coordinates (Cartesian)

    def filter_atoms_by_layers(self, atom_coords):
        """
        Filter atoms by dynamically calculated cross-sectional areas.
        Each layer corresponds to a unique z-coordinate from the atom positions.
        :param atom_coords: Array of Cartesian atom coordinates.
        :return: Array of filtered atom coordinates within the pyramid.
        """
        filtered_atoms = []
    
        # Extract unique z-coordinates from the atoms
        unique_z_values = np.unique(atom_coords[:, 2])
    
        for z_level in unique_z_values:
            # Calculate the cross-section for the current z-level
            cross_section = self.calculate_cross_section(z_level)
            print(f"Processing z = {z_level}: Cross-sectional polygon = {cross_section}")
    
            # Filter atoms that belong to this z-level
            for atom in atom_coords:
                if np.isclose(atom[2], z_level):  # Match the current z-level
                    if self.is_within_polygon(atom[:2], cross_section):  # Check if atom lies in the polygon
                        filtered_atoms.append(atom)  # Keep the atom
    
        return np.array(filtered_atoms)

    def calculate_cross_section(self, z_level):
        """
        Calculate the 2D polygon for a specific z-level of the pyramid.
        The polygon is dynamically interpolated based on the base and apex geometry.
        :param z_level: The z-coordinate of the current layer.
        :return: List of 2D vertices defining the polygon.
        """
        section_points = []
    
        for base_point in self.base_coords:
            # Interpolate the point on the plane for the given z-level
            slope = (self.apex_coords - base_point) / (self.apex_coords[2] - base_point[2])
            interpolated_point = base_point + slope * (z_level - base_point[2])
            section_points.append(interpolated_point[:2])  # Take only x, y coordinates
    
        return section_points
    
    def is_within_polygon(self, point, polygon):
        """
        Check if a 2D point lies within or on the boundary of a polygon,
        ignoring unit cell boundaries.
        :param point: (x, y) coordinates of the point.
        :param polygon: List of (x, y) vertices defining the polygon.
        :return: True if the point is inside or on the boundary, False otherwise.
        """
        if len(polygon) == 0:
            return False  # Handle empty polygons gracefully
        path = Path(polygon)  # Create a Path object for the polygon
        return path.contains_point(point) or path.contains_point(point, radius=-1e-0)

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
            f.write("Filtered Pyramid POSCAR (Layer-by-Layer Logic)\n")
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

filtered_positions = pyramid_filter.filter_atoms_by_layers(np.dot(positions, lattice))
counts[-1] = len(filtered_positions)  # Update atom count for the last element

PyramidFilter.write_poscar(output_file, lattice, np.linalg.solve(lattice.T, filtered_positions.T).T, elements, counts)