from pathlib import Path
import os
import sys

import numpy as np


###################################################
### Read Topology

def load_topol_info(lines, cat):
    Found = False
    for i, l in enumerate(lines):
        if Found:
            if l=='\n':
                break
            if l[0]==';':
                continue
            yield l
        elif cat in l:
            Found = True


def get_atom_info(lines):
    idx, atype, res = [], [], []
    for l in lines:
        splt = l.split()
        idx.append(splt[0])
        atype.append(splt[1])
        res.append(splt[3])
    return idx, atype, res


def load_topology(f):
    lines = [l.replace('\t', ' ') for l in open(f, 'r')]
    info = {k:list(load_topol_info(lines, k)) for k in ['atoms', 'bonds', 'angles', 'dihedrals']}
    
    idx, atype, res = get_atom_info(info['atoms'])
    bonds = [float(l.split()[3]) for l in info['bonds']]
    angles = [float(l.split()[4])*np.pi/180. for l in info['angles']]
    dihedrals = np.array([float(l.split()[5])*np.pi/180. for l in info['dihedrals']]).reshape(len(idx)-3,2)
#   dihedrals = np.radians(np.array([l.split()[5:7])*np.pi/180. for l in info['dihedrals']]).reshape(len(idx)-3,2)
    return idx, atype, res, bonds, angles, dihedrals


###################################################
### Read config (.gro)

def load_gro(f):
    data = [l.strip('\n') for l in open(f, 'r')]
    N = int(data[1])
    box = [float(x) for x in data[-1].split()]
    atype, res, xyz = [], [], np.zeros((N,3), dtype=float)

    no_vel = True if len(data[2].split()) <= 6 else False

    if not no_vel:
        vel = np.copy(xyz)
    for i, d in enumerate(data[2:-1]):
        splt = d.split()
        atype.append(splt[1])
        res.append(''.join([s for s in splt[0] if not s.isdigit() and s!=' ']))
        xyz[i] = [float(x) for x in splt[3:6]]
        if not no_vel:
            vel[i] = [float(x) for x in splt[6:9]]
    if not no_vel:
        return N, box, atype, res, xyz, vel
    else:
        return N, box, atype, res, xyz, []





