import argparse
from itertools import product
from pathlib import Path, PosixPath
import os
import sys

import mdtraj as md
from multiprocessing import Pool
import numpy as np


N_PROC = 28

PATH_BASE = Path("/home/jmcbride/CotransFold/3BID")
PATH_BASE = Path().joinpath(*Path().cwd().parts[:5])
#ATH_BASE = Path("/home/jmcbride/CotransFold/2OT2")
PATH_DATA = Path("/home/jmcbride/CotransFold/Test/Init")


###################################################
### Parse arguments


def parse_arguments():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("-k", type=int, default=10)
#   parser.add_argument("-t", "--top", type=str, default='gromacs.top', dest='top_file')
    parser.add_argument("-g", "--gro", type=str, default='gromacs.gro', dest='gro_file')
    parser.add_argument("-q", "--calc_q", action='store_true', dest='calc_q')
    parser.add_argument("-t", "--traj", type=str, default='traj.xtc', dest='traj_file')
    parser.add_argument("-d", "--dir", type=str, default='', dest='dir')
    parser.add_argument("-a", "--amalg", action='store_true', dest='amalg')
    parser.add_argument("--base_time", type=float, default=365.5, dest='base_time')
    parser.add_argument("--seqlen", type=int, default=58, dest='seqlen')
    return parser.parse_args()



def get_filenames():
    for root, dirs, files in os.walk(PATH_DATA):
        if 'template' in root:
            continue
        for f in files:
            if f=='confout.gro':
                yield os.path.join(root, f)


def load_and_calc_end2end(f):
    lines = [l for l in open(f, 'r')]
    xyz = np.array([[float(x) for x in l.split()[3:6]] for l in lines[2:-1]])
    return np.linalg.norm(xyz[0] - xyz[-1])


def unravel(dat):
    out = []
    for i, d in enumerate(dat):
        if len(d.shape):
            for x in d:
                out.append(x)
        else:
            out.append(d)
    return out


def load_and_unravel(path, i, j):
    data = [load_qfile(path.joinpath(f"RES{i:03d}", "traj.map")) for i in range(i, j)]
    return np.array(unravel(data))


def get_native_contacts(traj, native, scaling, Qfile, pairs):
    print (">>in get_native_contacts:", traj, '\n', native, scaling, Qfile)
    traj_pdb = md.load_pdb(str(native))
    # compute distances in native-CA file.
    dist_pdb = md.compute_distances(traj_pdb, pairs, periodic=False, opt=True)[0]
    # get distances for all pairs in all frames from trajfile
    contacts_xtc = md.compute_distances(traj, pairs, opt=True)
    Q_traj = np.array([len(np.where(con <= dist_pdb*scaling)[0])/len(pairs) for con in contacts_xtc])
    np.savetxt(Qfile, Q_traj)
    return Q_traj


def get_pairs_ext(filename):
    print ('>>in get_pairs_ext',filename)
    pairs = np.loadtxt(filename, dtype=int)[:,[1,3]]
    print('No. of contacts read from file', filename, '=',len(pairs))
    pairs = pairs - 1 #0 is first residue.
    return pairs


def native_contacts_traj(traj_file, gro_file):
    scaling=1.2
    Qfile = os.path.splitext(traj_file)[0] + '.map'
    if isinstance(traj_file, PosixPath):
        native = Path().joinpath(*traj_file.parts[:5], "PDB", "native_ca.pdb")
        pairfile = Path().joinpath(*traj_file.parts[:5], "PDB", "external_pair")
    else:
        native = PATH_BASE.joinpath("PDB", "native_ca.pdb")
        pairfile = PATH_BASE.joinpath("PDB", "external_pair")

    traj = md.load_xtc(str(traj_file), str(gro_file), stride=1)
    pairs = get_pairs_ext(pairfile)

    return get_native_contacts(traj, native, scaling, Qfile, pairs)


def load_qfile(q_file):
    if q_file.exists() and 1:
        Q = np.loadtxt(q_file)
#       print(q_file, Q)
    else:
        traj_file = q_file.with_name("traj.xtc")
        gro_file = q_file.with_name("confout.gro")
        Q = native_contacts_traj(traj_file, gro_file)
    return Q


def get_folding_rates_temp(repeat=999, cut=0.90, base=''):
    if base!='':
        path_base = Path().joinpath(*PATH_BASE.parts[:-1], base)
    else:
        path_base = PATH_BASE
    temp = np.arange(80, 112, 2)
    temp = np.arange(100,102,2)
#   temp = np.arange(90,92,2)
    tau_f = np.zeros((temp.size, repeat), dtype=float)
    for i, T in enumerate(temp):
        for j in range(repeat):
            q_file = path_base.joinpath("FoldingTemp", f"T{T:03d}", f"{j+1:03d}", f"traj.map")
            Q = load_qfile(q_file)
            i_fold = np.where(Q>cut)[0]
            if len(i_fold):
                tau_f[i,j] = i_fold[0]
            else:
                tau_f[i,j] = np.nan
    return tau_f

    
def get_tau_fold(path, st, k, seqlen, cut=0.90):
    path_ext = path.joinpath(f"trans_time_{st:05d}", f"{k+1:04d}")
#   Q = load_and_unravel(path_ext, 1, 59)
    Q = load_qfile(path_ext.joinpath(f"RES{seqlen:03d}", "traj.map"))
#   return Q[0]
#   return Q
    i_fold = np.where(Q>cut)[0]
    if len(i_fold):
        return i_fold[0]
    else:
        return np.nan


def get_speedup(path_idx, repeat=9999, cut=0.90, mp=False, base='', base_time=365.5, seqlen=-1, dt=100):
    if base!='':
        path_base = Path().joinpath(*PATH_BASE.parts[:4], base)
    else:
        path_base = PATH_BASE
    paths = [path_base.joinpath(f"{a}_first_{path_idx:02d}") for a in 'NC']
    steps = [int(x.strip('\n')) for x in open(paths[0].joinpath('steps.txt'))][:16]
    tau_f = np.zeros((2, len(steps), repeat), dtype=float)
    for i, s in enumerate(steps):
        tau_f[:,i,:] = s * seqlen
    if mp:
        with Pool(N_PROC) as pool:
            results = pool.starmap(get_tau_fold, product(paths, steps, range(repeat), [seqlen]), 100)
            tau_f = tau_f + np.array(results).reshape(tau_f.shape) * dt
    else:
        for i, path in enumerate(paths):
            for j, st in enumerate(steps):
                for k in range(repeat):
                    tau_f[i,j,k] = get_tau_fold(path, st, k, seqlen)
    return tau_f
    

if __name__ == "__main__":
    if 0:
        dist = [load_and_calc_end2end(f) for f in get_filenames()]
        for f in sorted(get_filenames()):
            print(f, round(load_and_calc_end2end(f),3))

    args = parse_arguments()
    
    if args.calc_q:
        if len(args.dir):
            for i in range(1,args.seqlen+1):
                traj_file = os.path.join(args.dir, f"RES{i:03d}", args.traj_file)
                gro_file  = os.path.join(args.dir, f"RES{i:03d}", args.gro_file)
                Q = native_contacts_traj(traj_file, gro_file)
        else:
            Q = native_contacts_traj(args.traj_file, args.gro_file)

    if args.amalg:
        i = 3
        tau_f = get_speedup(i, mp=True, base_time=args.base_time, seqlen=args.seqlen)
        np.save(PATH_BASE.joinpath(f'rates_{i:02d}.npy'), tau_f)


