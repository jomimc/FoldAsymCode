import json
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd

BASE_DIR = "/home/jmcbride/Jmcbride/ProteinDB/Pfam"
SRC_DIR = "/home/jmcbride/Jmcbride/ProteinDB/Pfam/Src"
DOWN_DIR = "/home/jmcbride/Jmcbride/ProteinDB/Pfam/Downloads"

FAS_F = os.path.join(DOWN_DIR, 'Pfam-A.fasta')
FULL_F = os.path.join(DOWN_DIR, 'Pfam-A.full')

SQL_DB = os.path.join(BASE_DIR, 'Data', 'pfam_db.sqlite')

TBL_I = "Pfam_info"
TBL_S = "Pfam_seq"

### Parse keys - (e.g. journal) reference keys
def parse_references(ref):
    refs = [s for s in ''.join(ref).split('#=GF RN') if s]
    N = len(refs)
    info = {'RN':[], 'RM':{}, 'RT':{}, 'RA':{}, 'RL':{}}
    for i, ref in enumerate(refs):
        rID = str(i+1)
        info['RN'].append(rID)
        lines = ref.split('#=GF ')

        RM = [l.split()[1] for l in lines if l[:2] == 'RM']
        if len(RM) == 1:
            info['RM'].update({rID:RM[0]})
        else:
            info['RM'].update({rID:''})

        for key in ['RT', 'RA', 'RL']:
            info[key].update({rID:' '.join([l[:5] for l in lines if l[:2] == key])})

    return info


### Parse keys - database references
def parse_database_references(ref):
    seps = [r[:7].split(';') for r in ref]
    DR = {s[0]:';'.join(s[1:]) for s in seps}
    return {'DR':DR}

    
### Parse keys - simple keys
def parse_key(lines, key):
    info = ''.join([l[7:] for l in lines if l[:7] == f"#=GF {key}"])
#   if key == 'AC':
#       info = info.split('.')[0]
    return {key:info}

### Parse information on aligned sequences
def parse_SEQ(entry, P_AC):
    GS  = [d for d in entry if '#=GS ' in d if d]
    GC  = [d for d in entry if '#=GC ' in d if d]
    GR  = [d for d in entry if '#=GR ' in d if d]
    NOG = [d for d in entry if '#=' != d[:2] and d]

    ids = [g.split()[1] for g in GS]
    out = {i:{'ID':i,'AC_P': P_AC, 'SEQ':'', 'SS':'', 'PDB':'', 'DR':''} for i in ids}

    for s in NOG:
        ss = s.split()
        if ss[0] in out.keys():
            out[ss[0]]['SEQ'] = ss[1]

    for g in GS:
        ss = g.split()
        if ss[2] == 'AC':
            out[ss[1]]['AC_U'] = ss[3].split('.')[0]
        elif ss[2] == 'DR':
            if 'DR PDB' in g:
                pdb = ';'.join(g.split(';')[1:])
                out[ss[1]]['PDB'] = out[ss[1]]['PDB'] + pdb
            else:
                dr = g.split(' DR ')[1]
                out[ss[1]]['DR'] = out[ss[1]]['PDB'] + dr

    for g in GR:
        ss = g.split()
        if ss[2] == 'AS':
            out[ss[1]]['ASi'] = ss[3]
        elif ss[2] == 'IN':
            out[ss[1]]['INt'] = ss[3]
        else:
            out[ss[1]][ss[2]] = ss[3]

    cons = P_AC + '_cons'
    out.update({cons:{'ID':cons, 'SEQ':'', 'SS':'', 'PDB':'', 'DR':''}})
    for g in GC:
        ss = g.split()
        feat = ss[1].split('_')[0].upper()
        out[cons][feat] = ss[2]

    return out


### Parse alignment information
def parse_GF(entry):
    info = {}
    for key in ['AC', 'ID', 'DE', 'BM', 'SM', 'SE', 'GA', 'NC', 'TC', 'TP', 'SQ', 'NE', 'NL', 'CC']:
        info.update(parse_key(entry, key))
    info.update(parse_references([e for e in entry if '#=GF R' in e]))
    info.update(parse_database_references([e for e in entry if '#=GF DR' in e]))
    return info


### Load full Pfam-A database from scratch
def load_pfam():
    data = [[s for s in e.split('\n') if s] for e in ''.join(open(FULL_F, encoding="ISO-8859-1")).split('//\n') if len(e)]
    return data


def generate_entry():
    entry = []
    for line in open(FULL_F, encoding="ISO-8859-1"):
        if line == "//\n":
            yield entry
            entry = []
        else:
            entry.append(line)


### Inserts values from dict into table
### Converts any nested dictionaries into json string
def dict_to_insert_command(crs, d, tbl):
    vals = [v if isinstance(v, str) else json.dumps(v) for v in d.values()]
    try:
        crs.execute(f"INSERT INTO {tbl}(" + " ,".join(d.keys()) + ") VALUES (" + " ,".join(['?']*len(vals)) + ")", vals)
    except Exception as e:
        name = [('AC',da['AC']) if 'AC' in da.keys() else ('ID',da['ID']) for da in [d]][0]
        print(f"Error at entry {name[0]} = {name[1]},\n\t{e}")
    

### Creates PFAM tables
def create_sql_pfam_tables(crs):
#   col_i = [['AC', 'varchar', 'PRIMARY KEY'],
    col_i = [['AC', 'varchar', 'PRIMARY KEY'],
             ['ID', 'varchar'],
             ['DE', 'varchar'],
             ['BM', 'varchar'],
             ['SM', 'varchar'],
             ['SE', 'varchar'],
             ['GA', 'varchar'],
             ['NC', 'varchar'],
             ['TC', 'varchar'],
             ['TP', 'varchar'],
             ['SQ', 'int'],
             ['RN', 'varchar'],
             ['RM', 'varchar'],
             ['RT', 'varchar'],
             ['RA', 'varchar'],
             ['RL', 'varchar'],
             ['DR', 'varchar'],
             ['CC', 'varchar'],
             ['NE', 'varchar'],
             ['NL', 'varchar']]

    tbl_i = 'CREATE TABLE IF NOT EXISTS {tb} ({col})'.format(tb=TBL_I, col=', '.join([' '.join(s) for s in col_i]))
    crs.execute(tbl_i)
#            '(AC varchar PRIMARY KEY, ID, DE, BM, SM, SE, GA, NC, TC, ' + \
#            'TP, SQ, RN, RM, RT, RA, RL, DR, CC, NE, NL)'

    col_s = [['IDX', 'int', 'PRIMARY KEY'],
             ['ID', 'varchar'],
             ['AC_P', 'varchar'],
             ['AC_U', 'varchar'],
             ['SEQ', 'varchar'],
             ['DR', 'varchar'],
             ['PDB', 'varchar'],
             ['SS', 'varchar'],
             ['SA', 'varchar'],
             ['TM', 'varchar'],
             ['PP', 'varchar'],
             ['LI', 'int'],
             ['ASi', 'varchar'],
             ['pAS', 'varchar'],
             ['sAS', 'varchar'],
             ['INt', 'varchar']]


    tbl_s = 'CREATE TABLE IF NOT EXISTS {tb} ({col})'.format(tb=TBL_S, col=', '.join([' '.join(s) for s in col_s]))
    crs.execute(tbl_s)
#   tbl_s = f'CREATE TABLE IF NOT EXISTS {TBL_S}' + \
#            '(AC_P, ID PRIMARY KEY, AC_U, SEQ, DR, PDB, SS, SA, TM, PP, LI, ASi, pAS, sAS, INt)'

#   return

    AC = set()
    ID = set()
    count = 0

    for i, e in enumerate(generate_entry()):
        print(i)
        info = parse_GF(e)
        if info['AC'] in AC:
            print(i)
            print([s for s in e if s[:4] == '#=GF'])
            print(info)
        else:
            AC.add(info['AC'])
        dict_to_insert_command(crs, info, TBL_I)

        seq_info = parse_SEQ(e, info['AC'])
        for s in seq_info:
            seq_info[s]['IDX'] = count
            count += 1
        [dict_to_insert_command(crs, seq_info[si], TBL_S) for si in seq_info]


if __name__ == "__main__":

    conn = sqlite3.connect(SQL_DB)
    crs = conn.cursor()

    create_sql_pfam_tables(crs)

    conn.commit()
    conn.close()

