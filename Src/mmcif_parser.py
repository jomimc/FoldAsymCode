"""
Some exceptions are found, and ignored:
    "1aw8"      in "ref", "PYR" is listed as part of chain B,
                but it is not part of the protein, or listed on
                the RSCB site;
                it seems like its due to a modification of the Serine at
                position 1 where a pyruvic acid is added

    22 entries have NaN values for chain (typically large assemblies)
                e.g. 5d8b, 5j4d, 6b4v, 6tz5

"""
from datetime import datetime
import string

import numpy as np
import pandas as pd

import asym_io
import asym_utils as utils



def load_mmcif(pdb_id, reformat=True):
    return asym_io.load_mmcif(pdb_id, reformat=reformat)


def extract_date(mmcif):
    fmt = "%Y-%m-%d"
    try:
        return datetime.strptime(mmcif['_pdbx_database_status']['recvd_initial_deposition_date'], fmt)
    except:
        return datetime.strptime(mmcif['_pdbx_database_status.recvd_initial_deposition_date'], fmt)


def extract_coords(mmcif, chain='', CA=False, all_col=False, use_auth=False):
    coords = pd.DataFrame(mmcif['_atom_site'])
    # Only consider data from one NMR model
    nmr_models = sorted(coords.pdbx_PDB_model_num.unique())
    coords = coords.loc[coords.pdbx_PDB_model_num==nmr_models[0]]
    # If not needed, remove author-contributed labels
    if not all_col:
        coords = coords.loc[:, coords.columns[:16]]
    # Choose a specific chain
    if chain:
        # Choose the chain according to author- label, or
        # other label (I've no idea why this happens...)
        if use_auth:
            coords = coords.loc[(coords.auth_asym_id==chain)]
        else:
            coords = coords.loc[(coords.label_asym_id==chain)]
    # Return only alpha carbons
    if CA:
        coords = coords.loc[(coords.label_atom_id=='CA')]
    return coords


def extract_helices(mmcif, chain='', all_col=False):
    # Check for structural information
    if not len(mmcif['_struct_conf']):
        return []
    # Check if there is only one row, and convert
    # each dict value into a list
    elif not isinstance(mmcif['_struct_conf']['id'], list):
        for k, v in mmcif['_struct_conf'].items():
            mmcif['_struct_conf'][k] = [v]
    helix = pd.DataFrame(mmcif['_struct_conf'])
    # If not needed, remove author-contributed labels
    if not all_col:
        helix = helix.loc[:, helix.columns[:11]]
    # Choose a specific chain
    if chain:
        helix = helix.loc[(helix.beg_label_asym_id==chain)]
    return helix


def extract_sheets(mmcif, chain='', all_col=False):
    # Check for structural information
    if not len(mmcif['_struct_sheet_range']):
        return []
    # Check if there is only one row, and convert
    # each dict value into a list
    elif not isinstance(mmcif['_struct_sheet_range']['id'], list):
        for k, v in mmcif['_struct_conf'].items():
            mmcif['_struct_sheet_range'][k] = [v]
    sheet = pd.DataFrame(mmcif['_struct_sheet_range'])
    # If not needed, remove author-contributed labels
    if not all_col:
        sheet = sheet.loc[:, sheet.columns[:10]]
    # Choose a specific chain
    if chain:
        sheet = sheet.loc[(sheet.beg_label_asym_id==chain)]
    return sheet


def match_entity_id_chain(mmcif):
    d = mmcif['_entity_poly']
    return {chain: entity for chain, entity in zip(d['pdbx_strand_id'], d['entity_id'])}


def match_asym_id_chain(ref):
    key = {}
    for asym_id, strand_id in zip(ref.asym_id, ref.pdb_strand_id):
        key[strand_id] = asym_id
    return key


def extract_ref_indices(mmcif, chain='', use_auth=False):
    ref = pd.DataFrame(mmcif['_pdbx_poly_seq_scheme'])
    # Choose a specific chain
    if chain:
        # Choose the chain according to author- label, or
        # other label (I've no idea why this happens...)
        if use_auth:
            ref = ref.loc[ref.pdb_strand_id==chain]
        else:
            ref = ref.loc[ref.asym_id==chain]
    return ref


def extract_config_and_secondary_structure(mmcif, chain='', use_auth=True, save=False):
    # If "use_auth", use "auth_asym_id" values for the chain ID
    # As far as I can see, this is appropriate for matching with the
    # SIFTS database
    if use_auth:
        coords = extract_coords(mmcif, chain=chain, CA=True, all_col=True, use_auth=True)
        ref = extract_ref_indices(mmcif, chain=chain, use_auth=True)
    else:
        coords = extract_coords(mmcif, chain=chain, CA=True)
        ref = extract_ref_indices(mmcif, chain=chain)

    # Secondary structure is not labelled with "auth_asym_id"
    # so this needs to be converted to "label_asym_id"
    if use_auth: 
        chain = match_asym_id_chain(ref)[chain]
    helix = extract_helices(mmcif, chain=chain)
    sheet = extract_sheets(mmcif, chain=chain)


    # Note every residue which has coords for CA as coil (.)
    ss_key = {i: '.' for i in coords.label_seq_id}

    # Replace (.) with helix (H)
    if len(helix):
        for beg, end in zip(helix.beg_label_seq_id, helix.end_label_seq_id):
            for i in range(int(beg), int(end) + 1):
                # Sometimes regions are annotated as structured
                # despite part of a region missing coordinates.
                # This avoids such disordered regions being labelled
                # structured.
                if str(i) in ss_key:
                    ss_key[str(i)] = 'H'

    # Replace (.) with sheet (S)
    if len(sheet):
        for beg, end in zip(sheet.beg_label_seq_id, sheet.end_label_seq_id):
            for i in range(int(beg), int(end) + 1):
                # Sometimes regions are annotated as structured
                # despite part of a region missing coordinates.
                # This avoids such disordered regions being labelled
                # structured.
                if str(i) in ss_key:
                    ss_key[str(i)] = 'S'

    # Initialise configuration
    config = []
    count = 0

    # Label the rest of the residues as disordered (D)
    # and amalgamate into a string
    ss = ''
    for i in ref.seq_id:
        if i not in ss_key:
            ss += 'D'
            # Preserve position of disordered residues
            config.append([np.nan]*3)
        else:
            ss += ss_key[i]
            config.append([coords.iloc[count][f"Cartn_{x}"] for x in 'xyz'])
            count += 1

    config = np.array(config, float)
    if save:
        pdb_id = mmcif['data_']
        path = asym_io.PATH_PDB_BASE.joinpath("data", "nonred", f"{pdb_id}_{chain}.npy")
        np.save(path, config)
        

    return ss, config







