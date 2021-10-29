from collections import defaultdict
from itertools import product
import os
from pathlib import Path
import pickle
import sqlite3
import sys

from Bio import SeqIO, pairwise2
from Bio.PDB import MMCIF2Dict, PDBParser, Selection 
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import IUPACData 
from goatools.obo_parser import GODag
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

PATH_BASE = Path("/data/Jmcbride/ProteinDB")


N_PROC = 50


PATH_GO  = PATH_BASE.joinpath("GeneOntology", "Database")
PATH_SIFTS = PATH_BASE.joinpath("Sifts")
PATH_SCOP = PATH_BASE.joinpath("SCOP")
PATH_KDB = PATH_BASE.joinpath("KineticDB")

PATH_PDB_BASE = Path("/home/protein/Protein/PDB/")
PATH_PDB_PDB = PATH_PDB_BASE.joinpath("data", "structures", "divided", "pdb")
PATH_PDB_MMCIF = PATH_PDB_BASE.joinpath("data", "structures", "divided", "mmCIF")

PATH_UNI  = PATH_BASE.joinpath("Uniprot")
SPROT_DB  = str(PATH_UNI.joinpath('Data', 'sprot_db.sqlite'))
TREMBL_DB = str(PATH_UNI.joinpath('Data', 'trembl_db.sqlite'))
TBL_SPROT = "Sprot"
TBL_TREMBL = 'trembl'


PATH_PFAM = Path("/home/jmcbride/Jmcbride/ProteinDB/Pfam")
PFAM_DB = PATH_PFAM.joinpath('Data', 'pfam_db.sqlite')
TBL_I = "Pfam_info"
TBL_S = "Pfam_seq"

PATH_ASYM = PATH_BASE.joinpath("Asymmetry")
PATH_ASYM_DATA = PATH_ASYM.joinpath("Data")

import asym_utils as utils
import pfdb_parser

sys.path.insert(0, '/data/Jmcbride/ProteinDB/ProteinSurface/masif/source/input_output/')
from extractPDB import extractPDB



#---------------------------------------------------------#
#   Data for Figures

# Load data for figures
def load_data_for_figures():
    pdb_all = pd.read_pickle(PATH_ASYM_DATA.joinpath("pdb_whole.pickle"))
    pdb = pd.read_pickle(PATH_ASYM_DATA.joinpath("pdb_nonredundant.pickle"))
    dom = pd.read_pickle(PATH_ASYM_DATA.joinpath("pdb_nonredundant_domain.pickle"))
    pfdb = load_pfdb()
    return pdb_all, pdb, dom, pfdb


#---------------------------------------------------------#
#   Gene Ontology

# Load Gene Ontology graph
def load_godag():
    return GODag(PATH_GO.joinpath('go-basic.obo'))


def load_go_pdb():
    path = PATH_SIFTS.joinpath("pdb_chain_go.csv")
    path_pickle = PATH_SIFTS.joinpath("pdb_chain_go.pickle")
    if path_pickle.exists():
        go = pickle.load(open(path_pickle, 'rb'))
    else:
        go = defaultdict(list)
        for l in open(path, 'r'):
            try:
                s = l.strip('\n').split(',')
                pdb_chain = '_'.join([s[0].upper(), s[1]])
                go[pdb_chain].append(s[-1])
            except Exception as e:
                print(l, e)
        pickle.dump(go, open(path_pickle, 'wb'))
    return go


#---------------------------------------------------------#
#   UniProt data

# Load individual sequences from UniProt
def load_sequence_from_ac(ac):
    conn_s =  sqlite3.connect(SPROT_DB)
    crsr_s = conn_s.cursor()
    sel = crsr_s.execute(f"SELECT SEQ FROM {TBL_SPROT} WHERE AC='{ac}'").fetchall()
    if len(sel):
        return sel[0][0]

    conn_t =  sqlite3.connect(TREMBL_DB)
    crsr_t = conn_t.cursor()
    sel = crsr_t.execute(f"SELECT SEQ FROM {TBL_TREMBL} WHERE AC='{ac}'").fetchall()
    if len(sel):
        return sel[0][0]
    return ''


# Load all sequences/data from UniProt
def load_info_from_uniprot(col='SEQ', sprot_only=False, df=False):
    conn_s =  sqlite3.connect(SPROT_DB)
    conn_t =  sqlite3.connect(TREMBL_DB)

    if df:
        if sprot_only:
            return pd.read_sql(f"SELECT {col} FROM {TBL_SPROT}", conn_s)
        else:
            return pd.concat([pd.read_sql(f"SELECT {col} FROM {t}", c) for t, c in zip([TBL_SPROT,  TBL_TREMBL], [conn_s, conn_t])], ignore_index=True)

    crsr_s = conn_s.cursor()
    crsr_t = conn_t.cursor()

    sprot = crsr_s.execute(f"SELECT {col} FROM {TBL_SPROT}").fetchall()
    if sprot_only:
        return sprot

    trembl = crsr_t.execute(f"SELECT {col} FROM {TBL_TREMBL}").fetchall()
    return sprot + trembl


# Load sequences/data from UniProt given some condition
def load_info_from_uniprot_condition(cond='', col='SEQ', df=False):
    seq = []
    conn_s =  sqlite3.connect(SPROT_DB)
    conn_t =  sqlite3.connect(TREMBL_DB)
    if df:
        return pd.concat([pd.read_sql(f"SELECT {col} FROM {t} {cond}", c) for t, c in zip([TBL_SPROT,  TBL_TREMBL], [conn_s, conn_t])], ignore_index=True)
    else:
        crsr_s = conn_s.cursor()
        crsr_t = conn_t.cursor()

        sprot = crsr_s.execute(f"SELECT {col} FROM {TBL_SPROT} {cond}").fetchall()
        if 'AC' in cond:
            if len(sprot):
                return sprot

        trembl = crsr_t.execute(f"SELECT {col} FROM {TBL_TREMBL} {cond}").fetchall()
        if 'AC' in cond:
            if len(trembl):
                return trembl
            else:
                return ''

        return sprot + trembl


#---------------------------------------------------------#
#   Sifts

# Load PDB 2 UNIPROT key
def load_pdb2uniprot():
    return pd.read_csv(PATH_SIFTS.joinpath("pdb_chain_uniprot.csv"))


# Load PDB 2 PFAM key
def load_pdb2pfam():
    return pd.read_csv(PATH_SIFTS.joinpath("pdb_chain_pfam.csv"))


# Load PDB 2 SCOP key
def load_pdb2scop2b():
    return pd.read_csv(PATH_SIFTS.joinpath("pdb_chain_scop2b_sf_uniprot.csv"))



#---------------------------------------------------------#
#   SCOP

def load_scop_all():
    scop_clas = pd.read_pickle(PATH_SCOP.joinpath("scop-cla-latest.pickle"))
    scop_desc = pd.read_csv(PATH_SCOP.joinpath("scop-des-latest.txt"))
    return scop_clas, scop_desc


#---------------------------------------------------------#
#   PDB

def load_pdb_seqres(pdb, chain):
    try:
        for record in SeqIO.parse(PATH_PDB_PDB.joinpath(pdb[1:3], f"pdb{pdb}.ent"), "pdb-seqres"):
            if record.annotations['chain'] == chain:
                return str(record.seq)
    except Exception as e:
        print(pdb, chain, e)
        return ''

def load_pdb_seqres_all(df):
    with Pool(N_PROC) as pool:
        seqres = pool.starmap(load_pdb_seqres, zip(df.PDB, df.CHAIN), 20)
    df['SEQ_PDB'] = seqres
    return df
        

def load_mmcif(pdb_id='', path='', reformat=True):
    if not len(path):
        path = PATH_PDB_MMCIF.joinpath(f"{pdb_id[1:3]}/{pdb_id}.cif")
    if reformat:
        return utils.reformat_mmcif(MMCIF2Dict.MMCIF2Dict(path))
    else:
        return MMCIF2Dict.MMCIF2Dict(path)


def check_pdb_file_exists(pdb, ptype='pdb'):
    if ptype == 'pdb':
        path = PATH_PDB_PDB.joinpath(f"{pdb[1:3]}", f"pdb{pdb}.ent")
    elif ptype == 'cif':
        path = PATH_PDB_MMCIF.joinpath(f"{pdb[1:3]}", f"{pdb}.cif")

    if not path.exists():
        path_zip = path.with_suffix(path.suffix + '.gz')
        if path_zip.exists():
            return 1, path_zip
        else:
            return 2, path
    return 0, ''


def check_all_files_exist(df):
    with Pool(N_PROC) as pool:
        pdb_files = list(pool.imap_unordered(check_pdb_file_exists, df.PDB, 20))
        cif_files = pool.starmap(check_pdb_file_exists, product(df.PDB, ['cif']), 20)

    sort_paths = defaultdict(list)
    for p, c in zip(pdb_files, cif_files):
        for item in [p, c]:
            sort_paths[item[0]].append(item[1])
    to_unzip, to_download = sort_paths[1], sort_paths[2]
    return to_unzip, to_download
        


#---------------------------------------------------------#
#   PDB configurations

# Extract and save invididual PDB chains from PDB files
def extract_chain(inputs, fresh=True):
    pdb_id, chain = inputs
    try:
        inp_f = PATH_PDB_PDB.joinpath(f"{pdb_id[1:3]}/pdb{pdb_id}.ent")
        out_f = PATH_PDB_BASE.joinpath("data", "nonred", f"{pdb_id}_{chain}.pdb")
        if not fresh:
            if os.path.exists(out_f):
                return
        extractPDB(inp_f, out_f, chain)
    except Exception as e:
        print(f"{e}\n{pdb_id}_{chain}")
        return pdb_id, chain


def extract_pdb_chains(df):
    with Pool(N_PROC) as pool:
        return list(pool.imap_unordered(extract_chain, zip(*df.loc[:, ['PDB', 'CHAIN']].values.T), 10))


def load_pdb_all_config(pdb_id, cut=10):
    pdb_id = pdb_id.lower()
    path = PATH_PDB_PDB.joinpath(f"{pdb_id[1:3]}/pdb{pdb_id}.ent")
    parser = PDBParser()
    struct = parser.get_structure("_", path)
    header = parser.get_header()

    data = defaultdict(dict)
    coords = defaultdict(list)
    for i, d in header["compound"].items():
        chain = d['chain'].upper()
        for c in chain.replace(' ','').split(','):
            data[c]['name'] = d['molecule']
            data[c]['contacts'] = []


    for chain in struct[0]:
        chain_id = chain.id.upper()
        ligands = []
        for res in chain:
            het = res.get_id()[0]
            if het[0] not in ['W', ' ']:
                ligands.append((het, [a.element for a in res]))
            else:
                for a in res:
                    coords[chain_id].append(a.get_vector().get_array())

        coords[chain_id] = np.array(coords[chain_id])
        data[chain_id]['ligands'] = ligands
        print(chain_id, len(coords[chain_id]))

    chain_list = list(data.keys())
    for i in range(len(chain_list)-1):
        c1 = chain_list[i]
        for j in range(i+1, len(chain_list)):
            c2 = chain_list[j]
            if np.any(distance_matrix(coords[c1], coords[c2]) < cut):
                data[c1]['contacts'].append(c2)
                data[c2]['contacts'].append(c1)

    return data


def load_pdb_chain_config(pdb_id, chain, imin, imax, save_idx=False):
    imin, imax = utils.parse_pdb_indices(imin, imax)
    coords = []
    seq_idx = []
    try:
        path = PATH_PDB_BASE.joinpath("data", "nonred", f"{pdb_id}_{chain}.pdb")
        parser = PDBParser()
        struct = parser.get_structure("_", path)
        model = struct[0]

        for chain in model:
            for res in chain:
                idx = res.get_id()[1]
                if (imin <= idx <= imax) and ('CA' in res):
                    coords.append(res['CA'].get_vector().get_array())
                    seq_idx.append(idx)
        if save_idx:
            return np.array(seq_idx), np.array(coords)
        else:
            return np.array(coords)

    except Exception as e:
        print(f"{e}\n{pdb_id}_{chain}")
        if save_idx:
            return np.array([]), np.array(coords).reshape(0,0)
        else:
            return np.array(coords).reshape(0,0)


def load_all_pdb_chain_config(df, save_idx=False):
    if save_idx:
        with Pool(N_PROC) as pool:
            inputs = zip(*df.loc[:, ['PDB', 'CHAIN', 'PDB_BEG', 'PDB_END']].values.T, [True]*len(df))
            results = pool.starmap(load_pdb_chain_config, inputs, 10)
            return np.array([r[0] for r in results]), np.array([r[1] for r in results])
    else:
        with Pool(N_PROC) as pool:
            inputs = zip(*df.loc[:, ['PDB', 'CHAIN', 'PDB_BEG', 'PDB_END']].values.T)
            return np.array(list(pool.starmap(load_pdb_chain_config, inputs, 10)))


    




#---------------------------------------------------------#
#   Protein folding databases


def load_pfdb():
    return pfdb_parser.load_pfdb(process=True)


def load_acpro():
    df = pd.read_csv(PATH_KDB.joinpath("acpro_db.csv"))
    df['ln kf'] = np.log10(np.exp(df['ln kf']))
    df = df.rename(columns={'ln kf': 'log_kf', 'Protein Length':'L'})
    return df


#---------------------------------------------------------#
#   PFAM

def load_pfam_seq_condition(cond, col='SEQ'):
    conn = sqlite3.connect(str(PFAM_DB))
    return pd.read_sql(f"SELECT {col} FROM {TBL_S} {cond}", conn)


def load_all_pfam_seq():
    conn = sqlite3.connect(str(PFAM_DB))
    return pd.read_sql(f"SELECT * FROM {TBL_S}", conn)


def load_all_pfam_info():
    conn = sqlite3.connect(str(PFAM_DB))
    return pd.read_sql(f"SELECT * FROM {TBL_I}", conn)


def load_entire_pfam_database():
    conn = sqlite3.connect(str(PFAM_DB))
    df_seq = load_all_pfam_seq()
    df_pfam = load_all_pfam_info()
    return df_seq, df_pfam





