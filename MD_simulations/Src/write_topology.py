from itertools import product
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix

import read_gro



HEADER = """
; Topology file generated from Go-kit.\n
; https://github.org/gokit1/gokit/\n

[ defaults  ]\n
; nbfunc comb-rule gen-pairs\n
  1      1         no   \n
\n
[ atomtypes ]\n
; name mass  charge ptype c6    c12\n
CA    1.000    0.000 A	0.000000e+00	1.677722e-05 \n
\n
[ moleculetype ]\n
; name            nrexcl\n
  Macromolecule   3\n
\n
"""


def write_gro(f, atype, res, xyz, vel, header='Header'):
    if len(vel):
        xyz = [''.join([f"{round(x,3):8.3f}" for x in list(res) + list(v)]) for res, v in zip(xyz, vel)]
    else:
        xyz = [''.join([f"{round(x,3):8.3f}" for x in res]) for res in xyz]
    idx = [str(i+1) for i in range(len(xyz))]
    with open(f, 'w') as o:
        o.write(f"{header}\n")
        o.write(f"{len(idx)}\n")
        for i, t, r, coords in zip(idx, atype, res, xyz):
            o.write(f"{i:>5s}{r:<4s}{t:>6s}{i:>5s}{coords}\n")
        o.write(" 50.0000"*3 + "\n")

def write_wall_tmp(f, xyz, start, head='Header'):
    t = 'CA'
    idx = [str(i+start) for i in range(len(xyz))]
    r = 'WALL'
    with open(f, 'w') as o:
        o.write(f"{head}\n")
        for i, (x, y, z) in zip(idx, xyz):
            o.write(f"{i+r:>8s}{t:>7s}{i:>5s}{round(x,3):8.3f}{round(y,3):8.3f}{round(z,3):8.3f}\n")
        o.write(" 50.0000"*3 + "\n")

def write_general(f, lines, nl='\n'):
    with open(f, 'w') as o:
        for l in lines:
            o.write(f"{l}{nl}")


def write_topology(ref_file, top_file, rest_p, N):
    topology = [l for l in open(ref_file, 'r')]
    start_top = ''.join(topology[:-7])
    end_top = ''.join(topology[-7:])

    # Add restraints to protein
    rest_top1 = '[ position_restraints ]\n; ai  funct  fcx    fcy    fcz\n' + \
               ''.join([f"{i:>4d}{1:>6d}{1000:>7d}{1000:>7d}{1000:>7d}\n" for i in rest_p])

    # Add wall molecules
    mol_top = '[ moleculetype ]\n; name            nrexcl\n  WALL            3   \n'

    atoms_top = '[ atoms ]\n;nr  type  resnr residue atom  cgnr\n' + \
                ''.join([f"{i:>8d}         CA    1    WALL       CA{i:>8d}\n" for i in [1]])

    rest_top2 = '[ position_restraints ]\n; ai  funct  fcx    fcy    fcz\n' + \
               ''.join([f"{i:>4d}{1:>6d}{1000:>7d}{1000:>7d}{1000:>7d}\n" for i in [1]])

    end_top += f'WALL           {N}\n'

    topology_str = '\n'.join([start_top, rest_top1, mol_top, atoms_top, rest_top2, end_top])
    with open(top_file, 'w') as o:
        o.write(topology_str)



if __name__ == "__main__":
    
    
    f = sys.argv[1]
    N, box, atype, res, xyz, vel = read_gro.load_gro(f)
    if len(vel):
        write_gro('gromacs.gro', atype, res, xyz, vel)
    else:
        write_gro('gromacs.gro', atype, res, xyz)


