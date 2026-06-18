from numpy import array, dot
from numpy.linalg import inv, norm
import sys
import os

class SeldynAdder:
    def __init__(self, target_dir, zmin, zmax):
        self.target_dir = target_dir
        self.ifile = os.path.join(target_dir, 'POSCAR')
        self.ofile = os.path.join(target_dir, 'POSCAR_seldyn')
        self.zmin = zmin
        self.zmax = zmax
        
        # Hard-coded constraints
        self.ranges = [[-100.0, 100.0], [-100.0, 100.0], [zmin, zmax]]
        self.frozen_axes = [0, 1, 2]
        self.reverse = True  # True means atoms in range are un-frozen ('T')

    def parse_poscar(self):
        with open(self.ifile, 'r') as file:
            lines = file.readlines()
            sf = float(lines[1])
            latticevectors = [float(lines[i].split()[j]) * sf for i in range(2, 5) for j in range(3)]
            latticevectors = array(latticevectors).reshape(3, 3)
            atomtypes = lines[5].split()
            atomnums = [int(i) for i in lines[6].split()]
            
            if 'Direct' in lines[7] or 'Cartesian' in lines[7]:
                start = 8
                mode = lines[7].split()[0]
            else:
                mode = lines[8].split()[0]
                start = 9
                # If seldyn already exists, we extract it (though it will be overwritten)
                self.existing_seldyn = [''.join(lines[i].split()[-3:]) for i in range(start, sum(atomnums) + start)]
            
            coord = array([[float(lines[i].split()[j]) for j in range(3)] for i in range(start, sum(atomnums) + start)])
            
            if mode != 'Cartesian':
                for i in range(sum(atomnums)):
                    for j in range(3):
                        while coord[i][j] > 1.0 or coord[i][j] < 0.0:
                            if coord[i][j] > 1.0:
                                coord[i][j] -= 1.0
                            elif coord[i][j] < 0.0:
                                coord[i][j] += 1.0
                    coord[i] = dot(coord[i], latticevectors)
            
            return latticevectors, coord, atomtypes, atomnums

    def write_poscar(self, lv, coord, atomtypes, atomnums, seldyn):
        with open(self.ofile, 'w') as file:
            file.write("\n")
            file.write('1.0\n')
            for i in range(3):
                for j in range(3):
                    file.write(str('{:<018f}'.format(lv[i][j])))
                    if j < 2:
                        file.write('  ')
                file.write('\n')
            for i in atomtypes:
                file.write('  ' + str(i))
            file.write('\n')
            for i in atomnums:
                file.write('  ' + str(i))
            file.write('\n')
            
            file.write('Selective Dynamics\n')
            file.write('Direct\n')
            
            # Convert Cartesian back to Direct for writing
            inv_lv = inv(lv)
            for i in range(len(coord)):
                dir_coord = dot(coord[i], inv_lv)
                for j in range(3):
                    file.write(str('{:<018f}'.format(dir_coord[j])))
                    if j < 2:
                        file.write('  ')
                
                # Write TTT/FFF strings
                for j in range(3):
                    file.write('  ')
                    file.write(seldyn[i][j])
                file.write('\n')
        print('new POSCAR written to: ' + str(self.ofile))

    def add_seldyn(self):
        lv, coord, atomtypes, atomnums = self.parse_poscar()
        
        frozen_atoms = []
        seldyn = ['' for i in range(sum(atomnums))]
        
        # Identify atoms within the hard-coded X/Y and provided Z range
        for i in range(len(coord)):
            counter = 0
            for j in range(3):
                if coord[i][j] > self.ranges[j][0] and coord[i][j] < self.ranges[j][1]:
                    counter += 1
            if counter == 3:
                frozen_atoms.append(i)
        
        # Assign Selective Dynamics flags
        for i in range(len(coord)):
            if i in frozen_atoms:
                for j in range(3):
                    if j in self.frozen_axes and not self.reverse:
                        seldyn[i] += 'F'
                    elif j in self.frozen_axes and self.reverse:
                        seldyn[i] += 'T'
                    elif self.reverse:
                        seldyn[i] += 'F'
                    else:
                        seldyn[i] += 'T'
            elif self.reverse:
                seldyn[i] += 'FFF'
            else:
                seldyn[i] = 'TTT'
        
        print(str(len(frozen_atoms)) + ' atoms selected')
        if not self.reverse:
            print('atoms frozen:' + str(frozen_atoms))
        else:
            print('atoms un-frozen:' + str(frozen_atoms))
            
        self.write_poscar(lv, coord, atomtypes, atomnums, seldyn)

if __name__ == '__main__':
    # Configuration
    target_directory = 'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au_smallest/'
    z_minimum = 1.0
    z_maximum = 20.0

    # Execute
    adder = SeldynAdder(target_directory, z_minimum, z_maximum)
    adder.add_seldyn()