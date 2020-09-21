from collections import defaultdict
from datetime import datetime

from ete3 import NCBITaxa
from multiprocessing import Pool
import numpy as np
import pandas as pd

from asym_io import PATH_ASYM_DATA
import asym_io
import asym_utils as utils
import folding_rate
import mmcif_parser
import pfam
import run_signalp
import scop
import structure


N_PROC = 50


# Get lineage from NCBI reference number
def get_parent_OC(OH):
    i = int(OH.split(';')[0].split('=')[1])
    NCBI = NCBITaxa()
    lineage = NCBI.get_lineage(i)
    if 2759 in lineage:
        return "Eukaryota"
    elif 2 in lineage:
        return "Bacteria"
    elif 2157 in lineage:
        return "Archaea"



# In reality, there is little data on k_trans, but the available data
# suggests that eukaryotes produce proteins at a rate of about 2-8 AA/s.
# and prokaryotes at 5-15 AA/s. Here we make a simplifying assumpion:
# If proteins are produced by eukaryotes, k_trans = 5
# If proteins are produced by prokaryotes, k_trans = 10
def get_k_trans(OC, OH):
    if 'Eukaryota' in OC:
        return 5
    elif 'Bacteria' in OC:
        return 10
    elif 'Archaea' in OC:
        return 10
    elif 'Viruses' in OC:
        if 'Eukaryota' in OH:
            return 5
        elif 'Bacteria' in OH:
            return 10
        elif 'Archaea' in OH:
            return 10
    return np.nan


# Load sequence and other info from Uniprot
def load_uniprot_info(df):
    cols = 'SEQ,AA,OC,OH,Fun,KW'
    col_list = cols.split(',')
    col_type = ['', 0] + [''] * 4
    for c, t in zip(col_list, col_type):
        df[c] = t
    
    for i, ac in zip(df.index, df.SP_PRIMARY):
        # Load data from Uniprot database
        d = asym_io.load_info_from_uniprot_condition(cond=f"WHERE AC='{ac}'", col=cols)
        if len(d) == 0:
            print(f"No data found for: {i}, {df.at[i, 'PDB']}, {df.at[i, 'SP_PRIMARY']}")
            continue

        # Reformat entries and save
        d = list(d[0])
        for j in range(len(d)):
            if isinstance(d[j], type(None)):
                d[j] = ''

        df.at[i, 'SEQ'] = d[0].replace('\n', '')
        df.at[i, 'AA'] = d[1]
        df.at[i, 'OC'] = d[2].replace('\n', '').replace(' ', '').split(';')[0]
        df.at[i, 'OH'] = d[3].replace('\n', '').replace(' ', '')
        df.at[i, 'Fun'] = d[4].replace('\n', ' ').rstrip()
        df.at[i, 'KW'] = d[5].replace('\n', ' ').rstrip()

        # Convert OH entry into one of Eukaryota, Archaea, Bacteria
        if df.at[i, 'OH'] != '':
            df.at[i, 'OH'] = get_parent_OC(df.at[i, 'OH'])

    return df


# Load config and secondary structure for each chain
def _load_structure(pdb, chain):
    ### Some chains are not labelled correctly
    ### (not sure if SIFTS or the PDB entry is incorrect)
    ### In this case, we try to swap the chain labels from "label" to "auth" to see if that works
    try:
        return mmcif_parser.extract_config_and_secondary_structure(mmcif_parser.load_mmcif(pdb), chain, save=True)
    except Exception as e:
        print(f"{pdb} {chain}:\n{e}")
        return '', np.empty((0,3))
        # Some exceptions did arise...
        # SIFTS chain labels were sometimes "AAA", "LLL", etc.;
        # while this is how they are displayed on the RCSB website,
        # they are not labelled as such in the MMCIF files, so I manually amended
        # the SIFTS pdb2uniprot DataFrame (about 5 entries in total):
        # "4abz", "4ue3"


def load_structure(df):
    with Pool(N_PROC) as pool:
        res = pool.starmap(_load_structure, zip(df.PDB, df.CHAIN), 20)
    df['SS_PDB'] = [r[0] for r in res]
    config_list = np.array([r[1] for r in res])
    return df, config_list


# Some structures have no secondary structure annotations, yet the
# structure is well-defined;
# Upon viewing many of these on the RSCB website, they appear to have
# helices and sheets;
# From a survey of such PDB structures, it appears that this is not true
# only for short proteins
# Thus, we exclude proteins from the analysis if they have more than
# 50 residues, yet no sheets or helices
def exclude_unrealistic_structures(df):
    df['Exclude'] = False
    df.loc[(df.Sfrac+df.Hfrac)==0, 'Exclude'] = True
    return df
 

# Get PDB structure deposition date
def get_date(pdb):
    try:
        return mmcif_parser.extract_date(mmcif_parser.load_mmcif(pdb, reformat=False))
    except Exception as e:
        # If exception, just put in an old date
        print(f"{pdb}\n{e}")
        return datetime.strptime("1900-01-01", "%Y-%m-%d")


# Compare Uniprot and PDB sequences so that we can
# identify unique, non-modified/non-mutated/non-spliced proteins
def compare_sequences(df, check_date=False):
    df['AA_PDB'] = df.SEQ_PDB.str.len()
    df['AA_PDB2'] = df.SEQ_PDB2.str.len()
    df['AA_FRAC'] = df.AA_PDB2 / df.AA
    with Pool(N_PROC) as pool:
        df['SEQ_DIST'] = pool.starmap(utils.get_dist, zip(df.SEQ, df.SEQ_PDB2), 40)
        if check_date:
            df['DATE'] = pool.map(get_date, df.PDB, 40)

    # Note the latest version of each unique protein
    if check_date or 1:
        df['LATEST'] = False
        ac_list = df.SP_PRIMARY.unique()
        for ac in ac_list:
            try:
                df.loc[df.loc[(df.SP_PRIMARY==ac)&(df.Exclude==False), 'DATE'].idxmax(), 'LATEST'] = True
            except ValueError:
                pass
        
    return df



# Get SS fractions and asymmetry
def get_ss_asym(df, SS='SS_PDB2'):
    cols = [f"{s}{ext}" for ext in ['frac', '_ASYM'] for s in 'SH.D']
    with Pool(N_PROC) as pool:
        res = np.array(pool.map(structure.ss_frac_and_asym, df[SS], 40))
    for c, r in zip(cols, res.T):
        df[c] = r
    return df


def _trim_pdb_seq_to_uniprot(beg, end, seq, ss, config):
    beg, end = int(beg), int(end)
    try:
        return seq[beg-1:end], ss[beg-1:end], config[beg-1:end]
    except Exception as e:
        print(e)
        return seq, ss, config


# Trim any sequences that have extra residues (e.g. Histidine tags)
def trim_pdb_seq_to_uniprot(df, config_list):
    with Pool(N_PROC) as pool:
        inputs = zip(*df.loc[:, ['RES_BEG', 'RES_END', 'SEQ_PDB', 'SS_PDB']].values.T, config_list)
        res = np.array(pool.starmap(_trim_pdb_seq_to_uniprot, inputs, 20))
    for c, r in zip(['SEQ_PDB2', 'SS_PDB2'], res.T):
        df[c] = r
    return df, np.array(res[:,2])


# Calculate contact order
def generate_config_no_disorder(config_list):
    for c in config_list:
        idx = np.where(np.isnan(c)==False)[0]
        yield c[idx], idx


def get_contact_order(config_list):
    with Pool(N_PROC) as pool:
        CO = pool.starmap(structure.calc_contact_order, generate_config_no_disorder(config_list), 20)
    return CO


def get_nonredunant_proteins(df):
    nonred = df.loc[(df.LATEST)&(df.SIG!=True)&(df.SEQ_DIST==0)]
    seq_idx = defaultdict(dict)
    # Even if sequences have different Uniprot IDs, only include
    # each sequence once;
    # Always choose the latest structure
    for i, seq, date in zip(nonred.index, nonred.SEQ, nonred.DATE):
        if seq not in seq_idx:
            seq_idx[seq] = {'idx':i, 'date':date}
        else:
            if date > seq_idx[seq]['date']:
                seq_idx[seq] = {'idx':i, 'date':date}
    idx_list = [v['idx'] for v in seq_idx.values()]
    return nonred.loc[idx_list]



# Annotate the entire PDB database from scratch,
# and create a non-redundant set for further analysis
def construct_pdb_database():

    #-----------------------------------------------------------------#
    ### Load Sequence Information 

    # Load mappings between PDB and Uniprot
    df = asym_io.load_pdb2uniprot()

    # Load info from Uniprot
    df = load_uniprot_info(df)

    # Load Sequences from PDB
    df = asym_io.load_pdb_seqres_all(df)


    #-----------------------------------------------------------------#
    ### Load Structural Information

    # Load Secondary Structures and configs from PDB
    # and match sequences to Uniprot
    df, config_list = load_structure(df)
    df, config_list_trimmed = trim_pdb_seq_to_uniprot(df, config_list)

    # Get secondary structure fractions and asymmetry
    df = get_ss_asym(df)

    # Get folding / translation rates
    df['k_trans'] = [get_k_trans(OC, OH) for OC, OH in zip(df.OC, df.OH)]
    df = folding_rate.get_folding_translation_rates(df)


    #-----------------------------------------------------------------#
    ### Create a non-redundant set of proteins

    # Exclude ordered structures that do not report having
    # any secondary structural information
    df = exclude_unrealistic_structures(df)

    # Load SignalP results
    df = run_signalp.match_pdb_with_signalp_results(df, 'pdb_whole')

    # Get information which will help reduce the data to only useful proteins
    df = compare_sequences(df, check_date=True)

    # Create non-redundant set of proteins
    nonred = get_nonredundant_proteins(df)


    #-----------------------------------------------------------------#
    ### More Structure Information

    # Add some more structural features (contact order, relative surface accessibility)
    nonred['CO'] = get_contact_order(config_list_trimmed[nonred.index])
    nonred['RSA'] = structure.rsa_fill_in_disorder_all(nonred, structure.get_rsa_all(nonred))
    # A few chains could not be automatically aligned for various reasons.
    # I just ignore these chains (about 40)
    nonred['USE_RSA'] = nonred.RSA.apply(lambda x: len(x) > 0)
    clusters = structure.cluster_proteins_by_ss(nonred)
    nonred['cl_12'] = clusters[10]
    nonred = match_scop_key_with_pdb(nonred)


    #-----------------------------------------------------------------#
    ### Create a non-redundant set of domains

    # Match PFAM domains
    df = pfam.match_pdb_to_pfam(df)
    dom = get_nonredundant_pdb_domain(df)
    dom = pfam.get_domain_contact_order(dom)


    #-----------------------------------------------------------------#
    ### Save

    df.to_pickle(PATH_ASYM_DATA.joinpath("pdb_whole.pickle"))
    nonred.to_pickle(PATH_ASYM_DATA.joinpath("pdb_nonredundant.pickle"))
    dom.to_pickle(PATH_ASYM_DATA.joinpath("pdb_nonredundant_domain.pickle"))



    return df, nonred






