"""
Parser for the Protein Folding kinetics DataBase (PFDB)



Some proteins were excluded:
    1ARR: seems to be two monomers joined together via a linker;
          can't assume that the pdb structure is the same as the paper
    1AU7: Specified residue indices (103-160) do not match either
          PDB (1-146) or UNIPROT (128-273)


For some proteins, the PDB data does not match the length
of the {Lpdb} column:
1L8W
3O4B
3049
1UCH


It seems that some indices given in the {PDB} column refer to
PDB indices, while others refer to UNIPROT indices

"""


from pathlib import Path

import numpy as np
import pandas as pd

import asym_utils as utils
import mmcif_parser
import structure


PATH_BASE = Path("/data/Jmcbride/ProteinDB")
PATH_KDB = PATH_BASE.joinpath("KineticDB")



def load_pfdb(process=False):
    if process:
        try:
            return pd.read_pickle(PATH_KDB.joinpath("pfdb.pickle"))
        except:
            pass
    df1 = pd.read_csv(PATH_KDB.joinpath("Final_2Sm.csv"))
    df2 = pd.read_csv(PATH_KDB.joinpath("Final_N2Sm.csv"))
    df2 = df2.rename(columns={"Protein\nshort name":'Protein short name'})
    cols = ['Protein short name', 'PDB', 'Class', 'Fold', 'Lpdb', 'L', 'pH',
           'Temp\n(°C)', 'Folding\ntype', 'ln(kf)', 'ln(kf)\n(25°C)', 'βT']
    df = pd.concat([df1.loc[:88, cols],  df2.loc[:51, cols]]).reset_index(drop=True)
    if process:
        df = process_pfdb(df)
        df.to_pickle(PATH_KDB.joinpath("pfdb.pickle"))
        return df
    else:
        return df


def load_old_processed():
    return pd.read_pickle(PATH_KDB.joinpath("kowajima.pickle"))


def process_pfdb(df):
    df = df.rename(columns={"ln(kf)":"ln_kf", "ln(kf)\n(25°C)":"ln_kf_25",
                   'Temp\n(°C)':"Temp_C"})
#   df.loc[df['ln_kf_25'].isnull(), 'ln_kf_25'] = df.loc[df['ln_kf_25'].isnull(), 'ln_kf']
    df['log_kf'] = np.log10(np.exp(df['ln_kf']))

    # Load some data from a previous version:
    # Extra data includes
    df_old = load_old_processed()
    cols = ['PDBid', 'CHAIN', 'idx', 'SEQ', 'SS', 'AA', 'idx3']
    for c in cols:
        df[c] = df_old[c]

    # Some of the chains were labelled wrong
    # These are some corrections
    df['CHAIN'] = 'A'
    alt_chain = {8:'B', 9:'C', 49:'C', 71:'C', 90:'C', 91:'B'}
    for i, c in alt_chain.items():
        df.loc[i, 'CHAIN'] = c

    # "idx3" is used when only a portion of a PDB CHAIN is relevant,
    # i.e., when kinetics are only available for a single domain in a
    # multi-domain protein.
    # Some indices were labelled wrong -- these are the apppropriate indices
    # See comments at the top, and code at the bottom for more info
    alt_idx3 = {4:(4, 104), 14:(1, 111), 15:(0, 36), 23:(0, 81), 38:(2, 77), 63:(12, 105), 71:(0, 57),
                74:(0, 89), 77:(152, 243), 78:(0, 94), 83:(0, 112), 88:(2, 99), 91:(7, 101), 101:(0, 130), 127:(5, 261)}

    for i, idx3 in alt_idx3.items():
        df.at[i, 'idx3'] = idx3

    
    df['use'] = True
    df.loc[(df.Temp_C<20)|(df.Temp_C>40)|(df.pH<5)|(df.pH>8), 'use'] = False
    df.loc[[0, 124], 'use'] = False

    df, configs = match_configs(df)

    return df


def load_pdb_config_all_chain(pdb_id):
    df = mmcif_parser.extract_coords(mmcif_parser.load_mmcif(pdb_id))
    chains = df.loc[df['label_atom_id']=='CA', "label_asym_id"].unique()
    return {c: df.loc[(df['label_atom_id']=='CA')&(df["label_asym_id"]==c)] for c in chains}


def get_config_segment(df, idx):
    idx = np.arange(*idx) + 1
    return unpack_unique_segment(df.loc[df.label_seq_id.apply(lambda x: int(x) in idx)])
    

def unpack_unique_segment(df):
    used = set()
    idx = []
    for i, j in zip(df.index, df.label_seq_id):
        if j not in used:
            idx.append(i)
            used.add(j)
    return utils.coordsdf2arr(df.loc[idx]), np.array(sorted(list(used)), int)


def match_configs(df):
    raw_configs = {i: load_pdb_config_all_chain(p.lower())  for i, p in zip(df.index, df.PDBid)}
    configs = {}
    for i in df.index:
        idx, chain = df.loc[i, ["idx3", "CHAIN"]]
        try:
            if isinstance(idx, tuple):
                c = get_config_segment(raw_configs[i][chain], idx)
            else:
                c = unpack_unique_segment(raw_configs[i][chain])
            configs.update({i:c})
        except:
            print(i)

    df['CO'] = [structure.calc_contact_order(con, idx) for con, idx in configs.values()]

    return df, configs


### This function was used to help determine the correct indices.
### Each problematic entry in the PFDB had to be manually checked.
def get_pdb_vs_uniprot_start(df, df_pdb):
    cols = ['CHAIN', 'RES_BEG', 'PDB_BEG', 'SP_BEG']
    return {i: {x[0]: x[1:] for x in df_pdb.loc[df_pdb.PDB==p.lower(), cols].values}  for i, p in zip(df.index, df.PDBid)}



