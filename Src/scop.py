from collections import Counter
from pathlib import Path
import os

import numpy as np
import pandas as pd

from asym_io import PATH_BASE


def extract_classes(s):
    out_dict = {}
    for item in s.split(','):
        c_type, c_id = item.split('=')
        out_dict[c_type] = int(c_id)
    return out_dict


def match_scop_key_with_pdb(pdb, cut=0):
    scop = pd.read_csv(os.path.join(PATH_BASE, "SCOP", "scop-cla-latest.txt"), delimiter=' ')
    for j, pdb_id, chain in zip(pdb.index, *pdb.loc[:,['PDB', 'CHAIN']].values.T):
        idx = scop.loc[scop['FA-PDBID']==pdb_id.upper()].index
        for i in idx:
            region, clas = scop.loc[i, ['FA-PDBREG', 'SCOPCLA']]
            if region[0] != chain:
                continue
            start, end = [int(''.join([y for y in x if y.isdigit()])) for x in region.split(':')[1].split('-')[-2:]]
            if cut > 0:
                if start > cut:
                    continue
            for k, v in extract_classes(clas).items():
                pdb.at[j,k] = v
    for key in ['TP', 'CL', 'CF', 'SF', 'FA']:
        pdb.loc[pdb[key].isnull(), key] = ''
        pdb[key] = pdb[key].astype(str)
        pdb.loc[pdb[key].str.len()>0, key] = pdb.loc[pdb[key].str.len()>0, key].apply(lambda x: str(int(float(x))))
    return pdb


def evaluate_scop_prob(df, idx, cut=10, cat='CF'):
    scop = pd.read_pickle(PATH_BASE.joinpath("SCOP", "scop-cla-latest.pickle"))
    scop_desc = {row[1]:row[2] for row in pd.read_csv(PATH_BASE.joinpath('SCOP/scop-des-latest.txt')).itertuples()}
    idx = [i in idx for i in df.index]

    euk_tot = len(df.loc[df.k_trans==5])
    pro_tot = len(df.loc[df.k_trans==10])

    euk_scop_tot = len(df.loc[(df.CL.str.len()>0)&(df.OC!='Viruses')&(df.k_trans==5)])
    pro_scop_tot = len(df.loc[(df.CL.str.len()>0)&(df.OC!='Viruses')&(df.k_trans==10)])


    for k, v in sorted(Counter(df.loc[(df.CL.str.len()>0)&(idx), cat]).items(), key=lambda x:x[1], reverse=True):
        if v > cut:
            euk_cl_tot = len(df.loc[(df[cat]==k)&(df.k_trans==5)])
            pro_cl_tot = len(df.loc[(df[cat]==k)&(df.k_trans==10)])
            ratio1 = (pro_cl_tot / pro_scop_tot)
            ratio2 = (euk_cl_tot / euk_scop_tot)
            try:
                ratio3 = ratio1 / ratio2
            except:
                ratio3 = np.nan
            print(k, f"  {v:<5d}  {scop_desc[int(k)][:40]:40s}  {ratio1:10.2f}  {ratio2:10.2f}  {ratio3:10.2f}")




