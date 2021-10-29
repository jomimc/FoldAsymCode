from collections import Counter, defaultdict
from itertools import product
import os
from pathlib import Path
import pickle
import titlecase

from Bio.SeqUtils import IUPACData 
import freesasa
import Levenshtein
import mdtraj as md
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.spatial import distance_matrix

from asym_io import PATH_BASE, PATH_ASYM_DATA
import asym_utils as utils

N_PROC = 50
MAX_SASA = {a:float(b) for l in open(PATH_BASE.joinpath('max_sasa.txt'), 'r') for a, b in [l.strip('\n').split()]}
letters3to1 = {k.upper():v for k, v in IUPACData.protein_letters_3to1.items()}


### Secondary structure length distributions

def ss_length_dist(ss_list, c='S'):
    size = []
    for ss in ss_list:
        count = 0
        for i, s in enumerate(ss):
            if s==c:
                count += 1
            else:
                if count:
                    size.append(count)
                    count = 0
        if count:
            size.append(count)
    return size


def get_ss_length_dist(df):
    length_dist = {ss: ss_length_dist(df['SS_PDB2'], c=ss) for ss in 'SH.D'}
    pickle.dump(length_dist, open(PATH_ASYM_DATA.joinpath("ss_length_dist.pickle"), 'wb'))
    return length_dist


### Secondary structure asymmetry
def ss_asym(ss, c='S'):
    i = int(len(ss)/2)
    if not i:
        return 0
    N = sum([(j+1)/i for j, s in enumerate(ss[:i][::-1]) if s==c])
    C = sum([(j+1)/i for j, s in enumerate(ss[-i:]) if s==c])
    return (N - C) / max(len(ss), 1)


def ss_frac_and_asym(ss):
    out = []
    N = max(len(ss), 1)
    count = Counter(ss)
    # Calcualte fractions
    for s in 'SH.D':
        out.append(float(count[s] / N))
    # Calculate asymmetry
    for s in 'SH.D':
        out.append(ss_asym(ss, c=s))
    return out


### Contact Order
def calc_contact_order(xyz, idx='', cut=10, k=0, abs_co=True):
    contacts = get_contacts(xyz, cut=cut, k=k)
    if abs_co:
        denom = len(contacts)
    else:
        denom = float(len(xyz)*len(contacts))
    if isinstance(idx, str):
        numer = np.sum(np.abs(np.diff(contacts)))
    # Preserve actual residue indices
    else:
        numer = 0.
        for i, j in contacts:
            numer += abs(idx[i] - idx[j])
    return numer / max(1, denom)


def get_contacts(xyz, cut=6, k=0):
    dist = distance_matrix(xyz, xyz)
    return [(i,j) for i, j in zip(*np.where(dist<cut)) if j > (i + k)]



### Solvent Accessibile Surface Area (Relative Surface Accessibility: RSA)

### Sometimes SASA values are given for partial residues.
### We ignore any residues that are missing alpha carbons.
### Any non-standard or modified amino acids are renamed 'X'
def get_rsa(pdb_id, chain, imin, imax):
    imin, imax = utils.parse_pdb_indices(imin, imax)

    try:
        path = f"/home/protein/Protein/PDB/data/nonred/{pdb_id}_{chain}.pdb"
        struct = freesasa.Structure(path, options={'hetatm':True})
        result = freesasa.calc(struct)
        sasa = defaultdict(float)
        res = {}
        res_has_CA = defaultdict(bool)
        for i in range(result.nAtoms()):
            j = int(''.join([x for x in struct.residueNumber(i).strip() if x.isdigit()]))
            if imin <= j <= imax:
                res_num = struct.residueNumber(i).strip()
                sasa[res_num] += result.atomArea(i)

                res_name = titlecase.titlecase(struct.residueName(i).strip(' '))
                res[res_num] = IUPACData.protein_letters_3to1.get(res_name, 'X')

                res_has_CA[res_num] += struct.atomName(i).strip() == 'CA'

        out_res = ''
        out_sasa = []
        for aa, (i, v) in zip(res.values(), sasa.items()):
            if res_has_CA[i]:
                out_sasa.append( v / MAX_SASA.get(aa, 1) )
                out_res += aa

        return np.array(out_sasa), out_res

    except Exception as e:
        print(f"{e}\n{pdb_id}_{chain}")
        return []


def get_rsa_all(df):
    with Pool(N_PROC) as pool:
        return np.array(pool.starmap(get_rsa, zip(*df.loc[:, ['PDB', 'CHAIN', 'PDB_BEG', 'PDB_END']].values.T), 10))    


### Tolerance allows for the fact that some amino acids are
### modified in the PDB. This results in an "X" being allocated
### by freesasa when calculating the SASA.
def rsa_fill_in_disorder(ss, seq, rsa_seq, rsa, tol=0.90):
    new_rsa = np.zeros(len(ss)) + 100
    idx_order = np.where(np.array(list(ss))!='D')[0]
    seq_order = ''.join(np.array(list(seq))[idx_order])
    l_seq = len(seq_order)
    l_rsa = len(rsa_seq)

    if l_seq == l_rsa:
        new_rsa[idx_order] = rsa
    elif l_seq > l_rsa:
        for j in range(l_seq - l_rsa + 1):
            if sum(np.array(list(seq_order[j:j+l_rsa])) == np.array(list(rsa_seq)))/max(l_rsa, 1) > tol:
                new_rsa[idx_order[j:j+l_rsa]] = rsa
                break
            else:
                continue
    elif l_seq < l_rsa:
        for j in range(l_rsa - l_seq + 1):
            if sum(np.array(list(seq_order)) == np.array(list(rsa_seq[j:j+l_seq])))/max(l_seq, 1) > tol:
                new_rsa[idx_order] = rsa[j:j+l_seq]
                break
            else:
                continue

    # Some sequences cannot be matched well due to inconsistencies...
    # In particular, sometimes there are two sets of coordinates for a single residue;
    # when this happens we take the one denoted 'A' or '1' in the PDB;
    # sometimes 'A'/'1' is actually missing an alpha carbon, while 'B'/'2' is not;
    # in these cases, the residue is annoted as ordered in the PDB, but ignored
    # by the above "get_rsa" algorihtm
    # In total, these appear to be only 40 out of 15,000 proteins. So we just ignore them.
    if np.all(new_rsa == 100.):
        print('Error: no matching sequence found')
        return None
    return new_rsa


def rsa_fill_in_disorder_all(df, rsa_list):
    out_rsa = []
    for i, seq, ss, rsa_dat in zip(df.index, df.SEQ_PDB2, df.SS_PDB2, rsa_list):
        if len(rsa_dat) < 2:
            out_rsa.append([])
            continue
        rsa, rsa_seq = rsa_dat
        if len(rsa_seq) == 0:
            out_rsa.append([])
            continue
        out_rsa.append(rsa_fill_in_disorder(ss, seq, rsa_seq, rsa))
    return np.array(out_rsa)


def rsa_asym(rsa, c='B'):
    try:
        q = np.quantile(rsa, np.linspace(0, 1, 4))
        rsa_string = ''
        for r in rsa:
            if r <= q[1]:
                rsa_string += 'B'
            elif r <= q[2]:
                rsa_string += 'M'
            elif r <= q[3]:
                rsa_string += 'E'

        N = Counter(rsa_string[:int(len(rsa)/2)])
        C = Counter(rsa_string[-int(len(rsa)/2)-1:])
        return (N.get(c, 0) - C.get(c, 0)) / len(rsa)
    except Exception as e:
        print(e)
        return 0
    


### Hierarchical clustering functions
def create_linkage(dist):
    idx = np.triu_indices(len(dist), k=1)
    return linkage(dist[idx], 'ward')


def cluster_proteins_by_ss(df, SS='SS_PDB2', N=50, whole_seq=True, reverse=False):
    # Cluster proteins based on the secondary structure of the first and last N residues
    if whole_seq:
        ss_list = [s[:N] + s[-N:] for s in df.loc[df.AA_PDB>=N*2, SS]]
    # Cluster proteins based on the secondary structure of the last N residues (C terminal)
    elif reverse:
        ss_list = [s[-N:] for s in df.loc[df.AA_PDB>=N*2, SS]]
    # Cluster proteins based on the secondary structure of the first N residues (N terminal)
    else:
        ss_list = [s[:N] for s in df.loc[df.AA_PDB>=N*2, SS]]

    with Pool(N_PROC) as pool:
        results = pool.starmap(Levenshtein.distance, product(ss_list, ss_list))
    ss_dist = np.array(results).reshape(*[len(ss_list)]*2)
    li = create_linkage(ss_dist)
    labels = np.array([fcluster(li, li[-i,2], criterion='distance') for i in range(2, 151)])
    np.save(PATH_ASYM_DATA.joinpath("nonred_clustering.npy"), labels)
    return labels


### Hydrogen-bond analysis


def hbond_get_seq_idx(top, hbonds, key):
    seqid = []
    for hbond in hbonds:
        i, j = [top.atom(hbond[k]).residue.index for k in [0,2]]
        if i in key and j in key:
            l, m = key[i], key[j]
            seqid.append([l,m])
    return np.array(seqid)


def match_index(pdb_id, top, seq, ss, imin):
    key = {}
    non_disorder = np.sum(np.array(list(ss))!='D')
    if non_disorder == 0:
        print("Protein is completely disordered")
        return key

    start = False
    i = 0
    while ss[i] == 'D':
        i += 1

    if imin != 'None':
        imin = int(imin)
    else:
#       imin = i + 1
        imin = int(str(list(top.residues)[0])[3:])
        

    for r in top.residues:
        a = letters3to1.get(str(r)[:3])
        j = r.index        
        iseq = int(str(r)[3:])
#       print(i, j, a, seq[i])

        if not start:
#           print(imin, iseq, i)
            if iseq == imin and a == seq[i]:
                start = True
                key[j] = i
                i += 1
        else:
            if a != seq[i]:
                print("Sequence mismatch!", pdb_id, i, a, seq[i])
                break
            if ss[i] == 'D':
                print("Disorder!!!", pdb_id, i, a, seq[i])
                break
                
            key[j] = i
            i += 1

        if i >= len(ss):
            break

        while ss[i] == 'D':
            i += 1
            if i >= len(ss):
                break

        if i >= len(ss):
            break

    if non_disorder != len(key):
        print("Wrong number of entries!", pdb_id, non_disorder, len(key))

    return key
    


# Before running this, one must first add hydrogens to PDB files.
# We used the "reduce" software to do this.
def get_hbonds(pdb_id, chain, seq, ss, pdb_beg):
    non_disorder = np.sum(np.array(list(ss))!='D')
    try:
        path = f"/data/Protein/Reduce/FoldAsym/Data/{pdb_id}_{chain}.pdb"
        t = md.load_pdb(path)
        if t.n_residues < non_disorder:
            print(f"Not enough residues found for {pdb_id}_{chain}")
            return None
        hbonds = md.wernet_nilsson(t, periodic=False)[0]
        key = match_index(pdb_id, t.topology, seq, ss, pdb_beg)
        return hbond_get_seq_idx(t.topology, hbonds, key)
    except Exception as e:
        print(f"Error at {pdb_id}_{chain}:\n{e}")
        return None


def get_mean(count, k):
    vals = count[k]
    if len(vals):
        return np.mean(vals)
    else:
        return np.nan


def get_mean_bond_length(ss, hb):
    count = defaultdict(list)
    key = ['SS', 'HH', '..', 'HS', '.H', '.S']
    try:
        for i, j in zip(*hb.T):
            s1, s2 = ss[i], ss[j]
            count[''.join(sorted([s1,s2]))].append(abs(i - j))
        return np.array([get_mean(count, k) for k in key])
    except:
        return np.ones(len(key)) * np.nan


def get_bond_length_asym(ss, hb):
    count = defaultdict(list)
    key = ['NN', 'CC', 'NC']
    n = int(len(ss)/2)
    try:
        for i, j in zip(*hb.T):
            if i < n and j < n:
                count['NN'].append(abs(i - j))
            elif i >= n and j >= n:
                count['CC'].append(abs(i - j))
            else:
                count['NC'].append(abs(i-j))
        return np.array([get_mean(count, k) for k in key])
    except:
        return np.ones(len(key)) * np.nan


def get_hbond_vs_ss(pdb, hbonds):
    ss_key = []
    hb_list = []
    for ss, hb in zip(pdb.SS_PDB2, hbonds):
        try:
            for i, j in zip(*hb.T):
                s1, s2 = ss[i], ss[j]
                k = ''.join(sorted([s1, s2]))
                ss_key.append(k)
                hb_list.append(abs(i-j))
        except:
            continue
    return pd.DataFrame(data={'ss':ss_key, 'hbdist':hb_list})


def run_hbond_analyses(pdb):
    with Pool(N_PROC) as pool:
        hbonds = list(pool.starmap(get_hbonds, zip(pdb.PDB, pdb.CHAIN), 10))
        mean_len = np.array(list(pool.starmap(get_mean_bond_length, zip(pdb.SS_PDB2, hbonds), 10)))
        asym = np.array(list(pool.starmap(get_bond_length_asym, zip(pdb.SS_PDB2, hbonds), 10)))

    return hbonds, mean_len, asym






