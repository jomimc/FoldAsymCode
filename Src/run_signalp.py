from pathlib import Path

from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import pandas as pd



def write_fasta_for_signalp(df, name, AC='AC', SEQ='SEQ'):
    idx = df.loc[df.AA>=50].index
    chunk = 200000
    steps = int(len(idx)/chunk)+1
    for i in range(steps):
        idx2 = idx[i*chunk:(i+1)*chunk]
        folder = Path(f'/home/jmcbride/Jmcbride/ProteinDB/Uniprot/Sequences/')
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder.joinpath(f'{name}_{i%chunk:02d}.fasta'), 'w') as o:
            SeqIO.write([SeqRecord(Seq(df.loc[j, SEQ]), id=df.loc[j, AC], name='..', description='..') for j in idx2], o, "fasta")


def load_signalp_results(name, org):
    path = Path(f'/home/jmcbride/Jmcbride/ProteinDB/Uniprot/Sequences/SignalP')
    path = path.joinpath(f"{name}_00_{org}_summary.signalp5")
    if org == 'euk':
        cols = ["ID", "Prediction", "SP(Sec/SPI)", "OTHER", "CS Position"]
    else:
        cols = ["ID", "Prediction", "SP(Sec/SPI)", "TAT(Tat/SPI)", "LIPO(Sec/SPII)", "OTHER", "CS Position"]

    

    data = [l.strip('\n').replace('\t',' ').split() for l in open(path, 'r')][2:]
    for i in range(len(data)):
        if len(data[i]) == len(cols)-1:
            data[i].append('')
        else:
            new_d = data[i][:len(cols)-1]
            new_d.append(' '.join(data[i][len(cols)-1:]))
            data[i] = new_d

    data = np.array(data).T
    data_dict = {c: d for c, d in zip(cols, data)}
    return pd.DataFrame(data_dict)
    

def match_pdb_with_signalp_results(df, name):
    df = df.loc[df.AA>=50]
    org_list = ['euk', 'arch', 'gram-', 'gram+']
    org_key = {o1:o2 for o1, o2 in zip(org_list, ['Eukaryota', 'Archaea', 'Bacteria', 'Bacteria'])}
    sigp_res = {o: load_signalp_results(name, o)['Prediction'].values for o in org_list}

    for i, j in enumerate(df.index):
        if 'Eukaryota' in df.loc[j, 'OC']:
            df.loc[j, 'SIG'] = sigp_res['euk'][i] != 'OTHER'
        elif 'Archaea' in df.loc[j, 'OC']:
            df.loc[j, 'SIG'] = sigp_res['arch'][i] != 'OTHER'
        elif 'Bacteria' in df.loc[j, 'OC']:
            df.loc[j, 'SIG'] = (sigp_res['gram-'][i] != 'OTHER') | (sigp_res['gram+'][i] != 'OTHER')
        elif 'Viruses' in df.loc[j, 'OC']:
            if 'Eukaryota' in df.loc[j, 'OH']:
                df.loc[j, 'SIG'] = sigp_res['euk'][i] != 'OTHER'
            elif 'Archaea' in df.loc[j, 'OH']:
                df.loc[j, 'SIG'] = sigp_res['arch'][i] != 'OTHER'
            elif 'Bacteria' in df.loc[j, 'OH']:
                df.loc[j, 'SIG'] = (sigp_res['gram-'][i] != 'OTHER') | (sigp_res['gram+'][i] != 'OTHER')

    return df








