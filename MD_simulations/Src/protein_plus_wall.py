from itertools import product
import os

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix

import write_topology as WT


###################################################
### Parse arguments


def parse_arguments():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("-k", type=int, default=10)
    return parser.parse_args()


###################################################
### Read files

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
    return idx, atype, res, bonds, angles, dihedrals


def update_position_restraints(f, k):
    lines = [l for l in open(f, 'r')]
    for i, (j, l) in enumerate(load_position_restraints(lines)):
        if i:
            splt = l.split()
            splt[4] = str(k)
            new_l = ''.join([f"{s:8s}" for s in splt])
            lines[j] = f"{new_l}\n"

    out_path = "/home/jmcbride/CotransFold/Test/Init/gromacs.top"
    with open(out_path, 'w') as o:
        for l in lines:
            o.write(l)
    

###################################################
### Create protein input configuration


def solve_theta(vec1, theta, sign=1):
    cos_t = np.cos(theta)
    if vec1[0]==0:
        y2 = cos_t / vec1[1]
        x2 = np.sqrt(1 - y2**2) * sign
    elif vec1[1]==0:
        x2 = cos_t / vec1[0]
        y2 = np.sqrt(1 - x2**2) * sign
    else:
        print('Some sort of error')
    return np.array([x2, y2, 0])


def calculate_dihedral(vec, vecA, vecB):
    norm1 = np.cross(vecA, vecB)
    norm2 = np.cross(vecB, vec)
    return np.arctan2(np.dot(np.cross(norm1, vecB), norm2), np.dot(norm1, norm2))


def optimise_vector(vec, vecA, vecB, r0, theta0, phi0):
    vec /= np.linalg.norm(vec)        
    phi0 -= np.pi*2*int(phi0/2/np.pi)
    theta = np.arccos(np.dot(-vecB, vec))
    d_theta =  min(abs(theta - theta0), abs(360 - theta - theta0))
    d_phi = calculate_dihedral(vec, vecA, vecB) - phi0
    return np.abs([d_theta, d_phi]).sum()
        

def optimise_vector_2(dir_vec, vecA, vecB, theta0, phi0):
    vec = []
    err = []
    direction = []
    for p, t in product(np.linspace(0, 2*np.pi, 100), np.linspace(0,np.pi, 50)):
        vec.append(np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)]))
        err.append(optimise_vector(vec[-1], vecA, vecB, 1, theta0, phi0))
        direction.append(np.dot(vec[-1], dir_vec))
    idx = np.argsort(err)[:10]
    idx = np.where(err < min(err)*1.5)[0]
    print(min(np.array(direction)[idx]), max(np.array(direction)[idx]))
    print(min(err))
    return vec[idx[np.argsort(np.array(direction)[idx])[-1]]]


def generate_xyz(N, bonds, angles, dihedrals):
    xyz = [np.array([0,0,0])]
    start_vec = np.array([1,0,0])
    for i in range(N-1):
        if not i:
            new_pos = xyz[-1] + bonds[i] * start_vec
        elif i==1:
            new_vec = solve_theta(-start_vec, angles[i-1])
            new_pos = xyz[-1] + bonds[i] * new_vec
        else:
            vec1 = xyz[-1] - xyz[-2]
            vec2 = xyz[-2] - xyz[-3]
            vec = optimise_vector_2(xyz[-1]/np.linalg.norm(xyz[-1]), vec2/np.linalg.norm(vec2), vec1/np.linalg.norm(vec1), angles[i-1], dihedrals[i-2,0])
            new_pos = xyz[-1] + bonds[i] * vec
#           args = (vec2/np.linalg.norm(vec2), vec1/np.linalg.norm(vec1), 1, angles[i-1], dihedrals[i-2,0])
#           res = minimize(optimise_vector, np.array([1,0,0]), args=args)
#           new_pos = xyz[-1] + bonds[i] * res.x / np.linalg.norm(res.x)
#       print(new_pos)
        xyz.append(new_pos)
        if i>0:
            vec1 = xyz[-1] - xyz[-2]
            vec2 = xyz[-2] - xyz[-3]
    return np.array(xyz)


def choose_normal_vec(xyz, cent, i):
    vec = []
    for p, t in product(np.linspace(0, 2*np.pi, 200), np.linspace(0,np.pi, 100)):
        vec.append(np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)]))
    min_dist = []
    for v in vec:
        dist = np.array([np.dot(v, x-cent) for x in xyz])
        if np.sign(dist[0] * dist[-1]) != -1:
            min_dist.append(0)
            continue
        left = [True if np.sign(dist[j] * dist[j+1])==1 else False for j in range(i-1)]
        right = [True if np.sign(dist[j] * dist[j+1])==1 else False for j in range(i,dist.size-1)]
        if np.alltrue(left) and np.alltrue(right):
            min_dist.append(np.abs(dist).min())
        else:
            min_dist.append(0)
    i_max = np.argmax(min_dist)
    return vec[i_max]


def get_orthogonal(norm):
    test = np.roll(norm, 1)
    if np.dot(test, norm) == 0:
        v1 = test
    else:
        v1 = np.cross(test, norm)
    v2 = np.cross(norm, v1)
    return v1, v2


def add_wall(N, xyz, i_cent, sigma=1.2):
    cent = xyz[i_cent]
    norm = choose_normal_vec(xyz[[i for i in range(len(xyz)) if i!=i_cent]], cent, i_cent)
    v1, v2 = get_orthogonal(norm)
    wall = []
    n = int(np.ceil(N**.5)/2)
    idx = np.array(list(product(*[range(-n, n+1)]*2)))
    idx = idx[np.argsort(np.abs(idx).sum(axis=1))]
    for i, j in idx[1:]:
        pos = cent + sigma * (i * v1 + j * v2)
        wall.append(pos)
    dist = distance_matrix(wall, xyz)
    idx_overlap = np.unique(np.where(dist<sigma*1.0)[0])
    idx_nooverlap = [i for i in range(len(wall)) if i not in idx_overlap]
    return np.array(wall)[idx_nooverlap[:N]]


def input_elongate_wall(f, N=100, i_cent=1):
    idx, atype, res, bonds, angles, dihedrals = load_topology()
    prot = generate_xyz(len(idx), bonds, angles, dihedrals)
    wall = add_wall(N, prot, i_cent)
    atype += ['CA']*len(wall)
    res += ['WALL']*len(wall)
    write_gro('tmp.gro', atype, res, np.append(prot, wall, axis=0))

    topology = [l for l in open(f, 'r')]
    start_top = ''.join(topology[:-7])
    end_top = ''.join(topology[-7:])

    rest = [1]
    rest_top1 = '[ position_restraints ]\n; ai  funct  fcx    fcy    fcz\n' + \
               ''.join([f"{i:>4d}{1:>6d}{1000:>7d}{1000:>7d}{1000:>7d}\n" for i in rest])
    mol_top = '[ moleculetype ]\n; name            nrexcl\n  WALL            3   \n'
    atoms_top = '[ atoms ]\n;nr  type  resnr residue atom  cgnr\n' + \
                ''.join([f"{i:>8d}         CA    1    WALL       CA{i:>8d}\n" for i in range(1,len(wall)+1)])
    dist = distance_matrix(wall, wall)
    contacts = [(i,j) for i, j in zip(*np.where(dist<=0.4)) if i < j]
    bonds_top = '[ bonds ]\n;ai     aj      func    r0(nm)  Kb   \n' + \
                ''.join([f"{i+1:>6d}{j+1:>6d}      1      0.4     2000\n" for i, j in contacts])
    rest_top2 = '[ position_restraints ]\n; ai  funct  fcx    fcy    fcz\n' + \
               ''.join([f"{i:>4d}{1:>6d}{1000:>7d}{1000:>7d}{1000:>7d}\n" for i in range(1, len(wall)+1)])

    topology_str = '\n'.join([start_top, rest_top1, mol_top, atoms_top, bonds_top, rest_top2, end_top, 'Wall         1\n'])
    write_topology('tmp.top', topology_str)

    return xyz



###################################################
### Write files

def write_gro(f, atype, res, xyz):
    idx = [str(i+1) for i in range(len(xyz))]
    with open(f, 'w') as o:
        o.write(f"{len(idx)}\n")
        for i, t, r, (x, y, z) in zip(idx, atype, res, xyz):
            o.write(f"{i+r:>8s}{t:>7s}{i:>5s}{round(x,3):8.3f}{round(y,3):8.3f}{round(z,3):8.3f}\n")
        o.write(" 50.0000"*3 + "\n")

def write_wall_tmp(f, xyz, start):
    t = 'CA'
    idx = [str(i+start) for i in range(len(xyz))]
    r = 'WALL'
    with open(f, 'w') as o:
        for i, (x, y, z) in zip(idx, xyz):
            o.write(f"{i+r:>8s}{t:>7s}{i:>5s}{round(x,3):8.3f}{round(y,3):8.3f}{round(z,3):8.3f}\n")
        o.write(" 50.0000"*3 + "\n")

def write_general(f, lines, nl='\n'):
    with open(f, 'w') as o:
        for l in lines:
            o.write(f"{l}{nl}")


def write_top(f, head):
    with open(f, 'w') as o:
        o.write(head)

    

if __name__ == "__main__":

    path = "/home/jmcbride/CotransFold/Test/Init/template/gromacs.top"
#   update_position_restraints(path, args.k)



