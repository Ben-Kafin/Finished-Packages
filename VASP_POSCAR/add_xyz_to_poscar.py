import numpy as np

class PoscarModifier:
    def __init__(self, xyz_path, poscar_path, output_path,
                 subtract_index, anchor_index, translation_offset,
                 merge_order="poscar-xyz"):
        """
        Reads the XYZ file, adjusts its coordinates, reads the POSCAR file,
        applies a translation to the XYZ coordinates based on the anchor coordinate
        in the POSCAR plus an offset, merges the two sets of data according to the 
        specified merge order, and writes a new POSCAR file.
        
        Parameters:
            xyz_path (str): Path to the input XYZ file.
            poscar_path (str): Path to the input POSCAR/CONTCAR file.
            output_path (str): Path to the output POSCAR file.
            subtract_index (int): Index from the XYZ coordinates for centering.
            anchor_index (int): Index from the POSCAR coordinates used for translation.
            translation_offset (np.array): Additional translation offset, e.g. np.array([0, 0, 2.4]).
            merge_order (str): Either "poscar-xyz" or "xyz-poscar" to set the merging order.
        """
        # Parse the XYZ file.
        coords_xyz, types_xyz, nums_xyz = self.parse_xyz(xyz_path)
        # Copy before modifying.
        coords_xyz = coords_xyz.copy()
        # Center the XYZ coordinates by subtracting the atom at subtract_index.
        coords_xyz -= coords_xyz[subtract_index]
        
        # Parse the POSCAR file.
        poscar_data = self.parse_poscar(poscar_path)
        if len(poscar_data) == 5:
            lv, coords_poscar, types_poscar, nums_poscar, seldyn = poscar_data
        else:
            lv, coords_poscar, types_poscar, nums_poscar = poscar_data
            seldyn = None

        # Compute the translation vector from the POSCAR coordinate at anchor_index + the offset.
        translation = coords_poscar[anchor_index] + translation_offset
        # Shift the XYZ coordinates.
        coords_xyz += translation

        # Merge the coordinates, atom types, and counts according to merge_order.
        if merge_order == "poscar-xyz":
            new_coords = list(coords_poscar) + list(coords_xyz)
            new_types = list(types_poscar) + list(types_xyz)
            new_nums = list(nums_poscar) + list(nums_xyz)
            # Extend selective dynamics flags: assume default ("TTT") for xyz atoms.
            if seldyn is not None:
                seldyn = seldyn + ["TTT"] * len(coords_xyz)
        elif merge_order == "xyz-poscar":
            new_coords = list(coords_xyz) + list(coords_poscar)
            new_types = list(types_xyz) + list(types_poscar)
            new_nums = list(nums_xyz) + list(nums_poscar)
            if seldyn is not None:
                seldyn = ["TTT"] * len(coords_xyz) + seldyn
        else:
            raise ValueError("merge_order must be either 'poscar-xyz' or 'xyz-poscar'")

        # Write out the new POSCAR.
        self.write_poscar(output_path, lv, new_coords, new_types, new_nums, seldyn=seldyn)

    def parse_poscar(self, ifile):
        """
        Parses a POSCAR/CONTCAR file.
        
        Returns:
            latticevectors (np.array): 3x3 array of lattice vectors.
            coord (np.array): Array of atomic coordinates.
            atomtypes (list): List of atom type labels.
            atomnums (list): List with the count of each atom type.
            seldyn (list, optional): List of selective dynamics flags if present.
        """
        with open(ifile, 'r') as file:
            lines = file.readlines()

        sf = float(lines[1])
        # Build the 3x3 lattice vectors.
        latticevectors = [float(lines[i].split()[j]) * sf for i in range(2, 5) for j in range(3)]
        latticevectors = np.array(latticevectors).reshape(3, 3)

        atomtypes = lines[5].split()
        atomnums = [int(i) for i in lines[6].split()]

        # Determine the line where the coordinates start.
        if 'Direct' in lines[7] or 'Cartesian' in lines[7]:
            start = 8
            mode = lines[7].split()[0]
            seldyn = None
        else:
            mode = lines[8].split()[0]
            start = 9
            seldyn = [''.join(lines[i].split()[-3:]) for i in range(start, sum(atomnums) + start)]
        coord = np.array([[float(lines[i].split()[j]) for j in range(3)]
                          for i in range(start, sum(atomnums) + start)])
        
        if mode != 'Cartesian':
            for i in range(sum(atomnums)):
                for j in range(3):
                    # Ensure coordinates are between 0 and 1.
                    while coord[i][j] > 1.0 or coord[i][j] < 0.0:
                        if coord[i][j] > 1.0:
                            coord[i][j] -= 1.0
                        elif coord[i][j] < 0.0:
                            coord[i][j] += 1.0
                coord[i] = np.dot(coord[i], latticevectors)

        try:
            return latticevectors, coord, atomtypes, atomnums, seldyn
        except NameError:
            return latticevectors, coord, atomtypes, atomnums

    def parse_xyz(self, ifile):
        """
        Parses an XYZ file.
        
        Returns:
            coord (np.array): Array of atomic coordinates.
            atomtypes (list): List of atom type labels.
            atomnums (list): List with counts for each atom type.
        """
        with open(ifile, 'r') as f:
            lines = f.readlines()
        # Split each line into tokens.
        for i in range(len(lines)):
            lines[i] = lines[i].split()
        
        coord = []
        atomtypes = []
        atomnums = []
        temp_coord = []

        num_atoms = int(lines[0][0])
        # Process each line that defines an atom (skip the first two header lines).
        for i in range(2, 2 + num_atoms):
            element = lines[i][0]
            if element in atomtypes:
                index = atomtypes.index(element)
                atomnums[index] += 1
            else:
                atomtypes.append(element)
                atomnums.append(1)
                temp_coord.append([])
            index = atomtypes.index(element)
            temp_coord[index].append(np.array([float(lines[i][j + 1]) for j in range(3)]))

        for i in range(len(atomnums)):
            for j in range(atomnums[i]):
                coord.append(temp_coord[i][j])
                
        coord = np.array(coord)
        return coord, atomtypes, atomnums

    def write_poscar(self, ofile, lv, coord, atomtypes, atomnums, **args):
        """
        Writes a POSCAR file using the provided lattice vectors, coordinate list, atom types, and atom counts.
        Optionally includes selective dynamics flags if provided.
        """
        with open(ofile, 'w') as file:
            if 'title' in args:
                file.write(str(args['title']))
            file.write('\n1.0\n')
            # Write lattice vectors.
            for i in range(3):
                for j in range(3):
                    file.write(str('{:<018f}'.format(lv[i][j])))
                    if j < 2:
                        file.write('  ')
                file.write('\n')
            # Write atom types.
            for at in atomtypes:
                file.write('  ' + str(at))
            file.write('\n')
            # Write atom counts.
            for num in atomnums:
                file.write('  ' + str(num))
            file.write('\n')
            if 'seldyn' in args and args['seldyn'] is not None:
                file.write('Selective Dynamics\n')
            file.write('Direct\n')
            
            # Convert Cartesian coordinates to direct coordinates.
            inv_lv = np.linalg.inv(lv)
            direct_coords = [np.dot(c, inv_lv) for c in coord]
            
            for i, d in enumerate(direct_coords):
                for j in range(3):
                    file.write(str('{:<018f}'.format(d[j])))
                    if j < 2:
                        file.write('  ')
                if 'seldyn' in args and args['seldyn'] is not None:
                    for j in range(3):
                        file.write('  ' + args['seldyn'][i][j])
                file.write('\n')
        print('New POSCAR written to: ' + str(ofile))


# Example usage:
if __name__ == "__main__":
    # Set file paths.
    xyz_file = r'D:/xyz/NHC2_iPr_adatom_60deg.xyz'
    poscar_file = r'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC2Au/CONTCAR'
    output_file = r'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC2Au/POSCAR'
    
    # Define the two indices and the translation offset.
    subtract_index = 0           # index used to center the XYZ coordinates
    anchor_index = 63            # index from the POSCAR to be used for translation
    translation_offset = np.array([0, 0, 8.85])
    
    # Choose the merging order: "poscar-xyz" (default) or "xyz-poscar".
    merge_order = "poscar-xyz"
    
    # Create an instance of the class which immediately processes the files and writes a new POSCAR.
    converter = PoscarModifier(xyz_file, poscar_file, output_file,
                               subtract_index, anchor_index, translation_offset,
                               merge_order)