"""
Code for creating a non-redundant set of protein domains.

    At the time of editing this, Pfam is version 33.1

    Note that Pfam releases a file called "pdbmap", that one
    could use to map to PDB domains. However, I could not find
    any documentation for / information about this file.
    Furthermore, upon inspection, there were odd things about the
    sequence indices that were provided.
    The fix for this is to write a parser from scratch.
    

"""
from collections import defaultdict
from pathlib import Path
import pickle

from multiprocessing import Pool
import numpy as np
import pandas as pd

import asym_io
from asym_io import PATH_BASE, PATH_ASYM_DATA, PATH_PDB_BASE
import folding_rate
import mmcif_parser
import structure

N_PROC = 50

PATH_PFAM = PATH_BASE.joinpath("Pfam")
PATH_PFAM_PDBMAP = PATH_PFAM.joinpath("Downloads", "pdbmap")


def is_pfam_in_pdb(ac, ac_list):
    return ac in ac_list


# For Pfam entries that match Uniprot Accession IDs in the PDB,
# extract the Pfam Accession ID and start/end index of the domain
def get_pdb2pfam_key(df, pfam=''):
    if PATH_ASYM_DATA.joinpath("pdb_pfam_domains.pickle").exists():
        dom_key = pickle.load(open(PATH_ASYM_DATA.joinpath("pdb_pfam_domains.pickle"), 'rb'))
    else:
        if isinstance(pfam, str):
            pfam = asym_io.load_all_pfam_seq()

        ac_list = df.SP_PRIMARY.unique()
        with Pool(N_PROC) as pool:
            idx = pool.map(is_pfam_in_pdb, product(pfam.AC_U, [ac_list]), 200)
        
        dom_key = defaultdict(list)
        for i, ac_u, ac_p, ID in zip(idx, *pfam_seq.loc[idx, ["AC_U", "AC_P", "ID"]].values.T):
            try:
                idx = ID.split("/")[1]
                beg, start = [int(x) for x in idx.split('-')]
                dom_key[ac_u].append((ac_p.split('.')[0].replace(' ',''), beg, start))
            except Exception as e:
                print(i, ac_u, ac_p, ID, e)

    return dom_key


# Match Pfam entries to the PDB
def match_pdb_to_pfam(df, save=False):
    dom_key = get_pdb2pfam_key(df)
    df['DOM_AC'] = df.SP_PRIMARY.apply(lambda x: [y[0] for y in dom_key[x]] if x in dom_key else [])
    df['DOM'] = df.SP_PRIMARY.apply(lambda x: [y[1:] for y in dom_key[x]] if x in dom_key else [])
    df['NDOM'] = df['DOM'].apply(len)
    return df


# Calculate contact order from information in the mmCIF file
def domain_co_fn(row, cut=10):
    try:
        pdb_id, chain, dom, res_beg, res_end = row
        if not len(dom):
            return [], []

        # The data was previously extracted, and reformatted as a numpy array.
        path = PATH_PDB_BASE.joinpath("data", "nonred", f"{pdb_id}_{chain}.npy")
        if path.exists():
            config = np.load(path)
        else:
            _, config = mmcif_parser.extract_config_and_secondary_structure(mmcif_parser.load_mmcif(pdb_id), chain, save=True)

        # Extract the part of 'config' that matches
        # the Uniprot sequence
        config = config[res_beg-1:res_end]

        # Extract domain using Uniprot indices (beg:end)
        beg, end = dom
        idx = [i for i, c in enumerate(config) if (beg <= i + 1 <= end) and not (np.any(np.isnan(c)))]
        return config, idx, structure.calc_contact_order(config[idx], idx)

    except Exception as e:
        print(f"{pdb_id} {chain}\n{e}")
        return -1



# Get Contact Order for each domain
def get_domain_contact_order(df):
    with Pool(N_PROC) as pool:
        results = list(pool.map(domain_co_fn, zip(*df.loc[:,['PDB', 'CHAIN', 'DOM', "RES_BEG", "RES_END"]].values.T), 25))

    df.loc[:, 'CO'] = results
    return df



# Extract a set of non-redundant domains
def get_nonredundant_pdb_domain(df):
    used = set()
    domain_dat = defaultdict(list)
    cols = ['SEQ', 'SP_BEG', 'SP_END', 'SEQ_PDB2', 'SS_PDB2', 'DOM', 'NDOM', 'k_trans', ]
    for i, seq, sp_beg, sp_end, seq_pdb, ss, d_idx, ndom, k_trans in zip(df.index, *df.loc[:, cols].values.T):
        for j, (dom_beg, dom_end) in enumerate(d_idx):
            # Domain sequence as defined by uniprot
            uni_seq = seq[dom_beg-1:dom_end]
            # Domain sequence as given in the pdb
            pdb_beg = dom_beg - sp_beg
            pdb_end = dom_end - sp_beg + 1
            dom_seq = seq_pdb[pdb_beg:pdb_end]

            # If not an exact match, do not include
            if uni_seq != dom_seq:
                continue
            # If matching sequence already accepted, do not include
            if dom_seq in used:
                continue
            used.add(dom_seq)
            domain_dat['IDX'].append(i)
            domain_dat['SEQ'].append(dom_seq)
            domain_dat['SS'].append(ss[pdb_beg:pdb_end])
            domain_dat['DOM'].append((dom_beg, dom_end))
            domain_dat['NDOM'].append(ndom)
            domain_dat['POS'].append(j + 1)
            domain_dat['AA'].append(len(dom_seq))
            domain_dat['k_trans'].append(k_trans)

    # Add other features needed for analyses
    ASYM = [[structure.ss_asym(s, c=c) for s in domain_dat['SS']] for c in 'SH.D']
    for c, a in zip(asym_cols, ASYM):
        domain_dat.update({c:a})
    dom = pd.DataFrame(data=domain_dat)
    dom = folding_rate.get_folding_translation_rates(dom)

    copy_cols = ['PDB', 'CHAIN', 'SP_PRIMARY', 'RES_BEG', 'RES_END']
    for c in copy_cols:
        dom[c] = [df.loc[i, c] for i in dom.IDX]
    
    return dom








