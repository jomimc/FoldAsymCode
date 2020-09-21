import argparse
from itertools import product
import json
import logging
import os
import pickle
import re
import sqlite3
import sys
import time

import multiprocessing as mp
import numpy as np
import pandas as pd

BASE_DIR = "/home/jmcbride/Jmcbride/ProteinDB/Uniprot"
SRC_DIR = "/home/jmcbride/Jmcbride/ProteinDB/Uniprot/Src"
DOWN_DIR = "/home/jmcbride/Jmcbride/ProteinDB/Uniprot/Downloads/"
PATH_TREMBL = "/home/jmcbride/Jmcbride/ProteinDB/Uniprot/Downloads/TrEMBL/"
SPROT_F = os.path.join(DOWN_DIR, "uniprot_sprot.dat")
TREMBL_F = os.path.join(DOWN_DIR, "uniprot_trembl.dat")

SQL_DB = os.path.join(BASE_DIR, 'Data', 'sprot_db.sqlite')
TREMBL_DB = os.path.join(BASE_DIR, 'Data', 'trembl_db.sqlite')

TBL = "Sprot"
TBL_TREMBL = 'trembl'



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('database')
    return parser.parse_args()


### Parse keys - (e.g. journal) reference keys
def parse_references(entry):
    refs = [[x for x in s.split('\n') if x] for s in '\n'.join([e for e in entry if len(re.findall('R[A-Z]   ', e))]).split('RN   ') if s]
    N = len(refs)
    info = {'RN':[], 'RP':{}, 'RX':{}, 'RA':{}, 'RT':{}, 'RL':{}}
    for i, ref in enumerate(refs):
        rID = str(i+1)
        info['RN'].append(rID)
        for key in ['RX', 'RP', 'RT', 'RA', 'RL']:
            info[key].update({rID:' '.join([l.replace(f'{key}   ', '') for l in ref if l[:2] == key])})
    return info

    
def parse_CC(entry):
    info = {}
    keys = ['ACTIVITY REGULATION', 'BIOPHYSICAL PROPERTIES', 'CATALYTIC ACTIVITY',
            'COFACTOR', 'DISEASE', 'DOMAIN', 'FUNCTION', 'INDUCTION', 'INTERACTION',
            'SUBCELLULAR LOCATION', 'PTM', 'SUBUNIT', 'SIMILARITY', 'TISSUE SPECIFICITY']
    k_short = ['AR', 'BpP', 'CA', 'Co', 'Di', 'Do', 'Fun', 'Ind', 'Int', 'ScL', 'PTM', 'Su', 'Sim', 'TS']
    key_convert = {keys[i]: k_short[i] for i in range(len(keys))}
    cc = ''.join([e.replace('CC   ', '') for e in entry if e[:5] == 'CC   ']).split('-!-')
    for c in cc:
        cs = c.split(':')
        k = ' '.join(cs[0].split())
        de = ':'.join(cs[1:])
        if k in keys:
            info.update({key_convert[k]:de})
    return info


def parse_SQ(entry):
    seq_info = [l.replace('SQ   ', '') for l in entry if l[:5] == f"SQ   "][0].split()
    AA = int(seq_info[1])
    MW = int(seq_info[3])
    CRC = seq_info[5]
    seq = ''.join([l.replace(f'     ', '') for l in entry if l[:5] == f"     "]).replace(' ', '')
    return {'AA':AA, 'MW':MW, 'CRC':CRC, 'SEQ':seq}


def parse_DR(entry):
    DR = []
    PFAM = []
    PDB = []
    for dr in [l.replace('DR   ', '') for l in entry if l[:5] == f"DR   "]:
        ss = dr.split(';')
        if ss[0] == 'Pfam':
            PFAM.append(ss[1])
        elif ss[0] == 'PDB':
            PDB.append(ss[1])
        else:
            DR.append(dr)
    return {k:v for k, v in zip(['DR', 'PDB', 'PFAM'],[DR, PDB, PFAM])}


### Parse keys - keys written in the same simple format
def parse_key(lines, key):
    info = ''.join([l.replace(f'{key}   ', '') for l in lines if l[:5] == f"{key}   "])
    return {key:info}


def parse_OH(entry):
    return {'OH':';;'.join([l.replace(f'OH   ', '') for l in entry if l[:5] == f"OH   "])}
    

def parse_AC(entry):
    AC = ''.join([l.replace('AC   ', '') for l in entry if l[:5] == f"AC   "]).replace(' ', '').split(';')
    return {'AC':AC[0], 'AC2':';'.join(AC[1:])}
    

def process_entry(entry):
    info = {}
    info.update(parse_AC(entry))
    for key in ['ID', 'DE', 'GN', 'OS', 'OG', 'OC', 'OX', 'PE', 'KW', 'FT']:
        info.update(parse_key(entry, key))
    info.update(parse_OH(entry))
    info.update(parse_CC(entry))
    info.update(parse_references(entry))
    info.update(parse_DR(entry))
    info.update(parse_SQ(entry))
    return info
    
    
### Load full Pfam-A database from scratch
def load_sprot():
    data = [[s for s in e.split('\n') if s] for e in ''.join(open(SPROT_F, encoding="ISO-8859-1")).split('//\n') if len(e)]
    return data


def generate_entry(path):
    entry = []
    for line in open(path):
        if line == "//\n":
            yield entry
            entry = []
        else:
            entry.append(line)



### Inserts values from dict into table
### Converts any nested dictionaries into json string
def dict_to_insert_command(crs, d, tbl):
    keys = d.keys()
    vals = list(d.values())
    for i, k in enumerate(keys):
        if k[0] == 'R':
            vals[i] = json.dumps(d[k])
        if isinstance(vals[i], list):
            vals[i] = json.dumps(d[k])
    try:
        crs.execute(f"INSERT INTO {tbl}(" + " ,".join(keys) + ") VALUES (" + " ,".join(['?']*len(vals)) + ")", vals)
    except Exception as e:
        print(f"Error at entry AC = {d['AC']}\n\t{e}")
        logging.warning(f"{d['AC']}")
    

### Creates PFAM tables
def create_sql_sprot_table():
    ts = time.time()
    conn = sqlite3.connect(SQL_DB)
    crs = conn.cursor()


    print(f"Time to load entries: {(time.time()-ts)/60.} min")

    col = [ ['AC VARCHAR PRIMARY KEY'] ] + \
            [f"{s} VARCHAR" for s in ['ID', 'AC2', 'DE', 'GN', 'OS', 'OG', 'OC', 'OX', 'OH',
                                      'AR', 'BpP', 'CA', 'Co', 'Di', 'Do', 'Fun', 'Ind',
                                      'ScL', 'PTM', 'Su', 'Sim', 'TS', 'RN', 'RX', 'RP',
                                      'RT', 'RA', 'RL', 'DR', 'PDB', 'PFAM', 'PE', 'KW', 'FT', 'SEQ', 'CRC']] + \
            ['AA int', 'MW int']
            
    tbl = f'CREATE TABLE IF NOT EXISTS {TBL}' +  \
           '(AC PRIMARY KEY, ID, AC2, DE, GN, OS, OG, OC, OX, OH, ' + \
           'AR, BpP, CA, Co, Di, Do, Fun, Ind, Int, ScL, PTM, Su, Sim, TS,' + \
           'RN, RX, RP, RT, RA, RL, DR, PDB, PFAM, PE, KW, FT, AA, MW, CRC, SEQ)'

    crs.execute(tbl)

    for i, e in enumerate(generate_entry(SPROT_F)):
        print(i)
        info = process_entry(e)
        dict_to_insert_command(crs, info, TBL)

    conn.commit()
    conn.close()

    print(f"Time to save database: {(time.time()-ts)/60.} min")
    

def load_trembl(f):
    return [[s for s in e.split('\n') if s] for e in ''.join(open(f, encoding="ISO-8859-1")).split('//\n') if len(e)][0]


def create_sql_trembl_table():
    ts = time.time()
    conn = sqlite3.connect(TREMBL_DB)
    crs = conn.cursor()

    tbl = f'CREATE TABLE IF NOT EXISTS {TBL_TREMBL}' +  \
           '(AC PRIMARY KEY, ID, AC2, DE, GN, OS, OG, OC, OX, OH, ' + \
           'AR, BpP, CA, Co, Di, Do, Fun, Ind, Int, ScL, PTM, Su, Sim, TS,' + \
           'RN, RX, RP, RT, RA, RL, DR, PDB, PFAM, PE, KW, FT, AA, MW, CRC, SEQ)'

    crs.execute(tbl)


### Check file "separate_trembl.py" to find how the following '.dat' files were
### created. Don't judge me.
### You can nab the parser from that and turn it into a generator instead of writing to file.
#   for i in range(147413762):
#   for i in range(100):
    for i, entry in enumerate(generate_entry(TREMBL_F)):
        print(f"{i:09d}")
        info = process_entry(entry)
        dict_to_insert_command(crs, info, TBL_TREMBL)

    print(f"Time to save database: {(time.time()-ts)/60.} min")

    conn.commit()
    conn.close()

def load_sql_sprot_table():
    with sqlite3.connect(SQL_DB) as conn:
        return pd.read_sql("SELECT * FROM Sprot", conn)


if __name__ == "__main__":

    args = parse_arguments()
    if args.database == 'sprot':
        logging.basicConfig(filename='errors_sprot.log', filemode='w', format="%(message)s")
        create_sql_sprot_table()

    elif args.database == 'trembl':
        logging.basicConfig(filename='errors_trembl.log', filemode='w', format="%(message)s")
        create_sql_trembl_table()


