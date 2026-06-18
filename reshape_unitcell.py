import numpy as np

class POSCARTransformer:
    def __init__(self, input_file, output_file, new_lattice_vectors, mode='centered'):
        self.input_file = input_file
        self.output_file = output_file
        self.new_lattice_vectors = np.array(new_lattice_vectors)
        self.mode = mode
        self.old_lattice_vectors = None
        self.atom_types = None
        self.atom_numbers = None
        self.cartesian_coordinates = None  # Cartesian coordinates remain fixed in space

    def read_poscar(self):
        with open(self.input_file, 'r') as file:
            lines = file.readlines()
        
        scaling_factor = float(lines[1])
        self.old_lattice_vectors = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scaling_factor
        self.atom_types = lines[5].split()
        self.atom_numbers = [int(x) for x in lines[6].split()]
        total_atoms = sum(self.atom_numbers)
        coord_start = 8
        fractional_coordinates = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(coord_start, coord_start + total_atoms)])
        
        if self.mode == 'centered':
            # Zero the center in the xy-plane
            fractional_coordinates[:, :2] -= 0.5  # Shift xy-coordinates to center (subtract 0.5)
        
        self.cartesian_coordinates = self.fractional_to_cartesian(fractional_coordinates, self.old_lattice_vectors)

    def write_poscar(self):
        # Convert Cartesian coordinates back to fractional using the new lattice
        fractional_coordinates = self.cartesian_to_fractional(self.cartesian_coordinates, self.new_lattice_vectors)
        
        if self.mode == 'centered':
            # Restore the center in the xy-plane
            fractional_coordinates[:, :2] += 0.5  # Add 0.5 to shift back to center in xy
        
        # Identify atoms that are within the [0, 1) range in all three dimensions
        mask = np.all((fractional_coordinates >= 0) & (fractional_coordinates < 1), axis=1)
        
        # Filter coordinates
        filtered_coords = fractional_coordinates[mask]
        
        # Update atom counts and types based on survivors
        new_atom_numbers = []
        new_atom_types = []
        current_idx = 0
        
        for i, count in enumerate(self.atom_numbers):
            # Check how many atoms in this species group survived the mask
            species_mask = mask[current_idx : current_idx + count]
            surviving_count = np.sum(species_mask)
            
            if surviving_count > 0:
                new_atom_numbers.append(surviving_count)
                new_atom_types.append(self.atom_types[i])
            
            current_idx += count

        with open(self.output_file, 'w') as file:
            file.write("Generated POSCAR\n")
            file.write("1.0\n")
            for vec in self.new_lattice_vectors:
                file.write("  {:.10f}  {:.10f}  {:.10f}\n".format(*vec))
            file.write("  " + "  ".join(new_atom_types) + "\n")
            file.write("  " + "  ".join(map(str, new_atom_numbers)) + "\n")
            file.write("Direct\n")
            for coord in filtered_coords:
                file.write("  {:.10f}  {:.10f}  {:.10f}\n".format(*coord))

    def fractional_to_cartesian(self, fractional_coords, lattice_vectors):
        return np.dot(fractional_coords, lattice_vectors)

    def cartesian_to_fractional(self, cartesian_coords, lattice_vectors):
        inverse_lattice = np.linalg.inv(lattice_vectors)
        return np.dot(cartesian_coords, inverse_lattice)

    def transform(self):
        self.read_poscar()  # Read old POSCAR and zero-center
        self.write_poscar()  # Transform and filter atoms outside bounds
        print(f"Transformation complete. New POSCAR written to {self.output_file}")


# Example usage
if __name__ == "__main__":

    # Set mode to 'origin' to keep the absolute origin, or 'centered' to use previous behavior
    transformer = POSCARTransformer(
        'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au_smaller/SOC/wvchg/CONTCAR',
        'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au_smaller/SOC/lmaxmix4/POSCAR', 
        [
            [17.9802889999999991,    0.0000000000000000,    0.0000000000000000],
            [-8.9901499999999999,   15.5713910000000002,    0.0000000000000000],
            [0.0000000000000000,    0.0000000000000000,   25.0]
        ],
        mode='origin'
    )
    transformer.transform()