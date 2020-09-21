from collections import Counter, defaultdict

import Levenshtein
import numpy as np
import statsmodels.nonparametric.api as smnp

AA_ALPHA = np.array(list("ACDEFGHILKMNPQRSTVYW"))

def smooth_dist_kde(Xgrid, Xobs):
    kde = smnp.KDEUnivariate(np.array(Xobs))
    kde.fit(kernel='gau', bw='scott', fft=True, gridsize=10000, cut=20)
    y = np.array(kde.evaluate(Xgrid))
    return y.reshape(y.size)

        

def reformat_mmcif(mmcif):
    new_dict = defaultdict(dict)
    for k, v in mmcif.items():
        if '.' not in k:
            new_dict[k] = v
            continue
        k1 = k.split('.')[0]
        k2 = '.'.join(k.split('.')[1:])
        new_dict[k1][k2] = v
    return new_dict


def coordsdf2arr(df):
    return np.array(df.loc[:, [f'Cartn_{a}' for a in 'xyz']].values, dtype=float)


def add_to_dict(d, k):
    try:
        d[k] += 1
    except KeyError:
        pass
    return d



# Edit distance
def get_dist(s1, s2):
    try:
        return Levenshtein.distance(s1, s2)
    except:
        return -1


# Identify continuous segments of secondary structure / disorder
def ss_segment(ss, c='S'):
    idx = []
    segments = []
    rest = []
    for i, s in enumerate(ss):
        if s==c:
            idx.append(i)
        else:
            if len(idx):
                segments.append(idx)
                idx = []
            rest.append(i)
    if len(idx):
        segments.append(idx)
    return segments, rest


# Shuffle a SS sequence, while keeping strands of a particular
# structure / disorder intact
def shuffle_seq_strand_intact(ss, s='S', bias_ends=False):
    segments, rest = ss_segment(ss, c=s)

    if bias_ends:
        if len(segments) == 1:
            if np.random.rand() > 0.5:
                idx = segments[0] + rest
            else:
                idx = rest + segments[0]
        else:
            n_seg = len(segments)
            idx_beg = []
            idx_end = []
            for seg in segments:
                if np.random.rand() > 0.5:
                    idx_beg.extend(seg)
                else:
                    idx_end.extend(seg)
            idx = idx_beg + rest + idx_end
    else:
        combined = segments + rest
        # If there is only one segment...
        if len(combined) == 1: combined = combined[0]

        idx = []
        for i in np.random.choice(combined, size=len(combined), replace=False):
            if isinstance(i, int):
                idx.append(i)
            elif isinstance(i, list):
                idx.extend(i)
    return ''.join(np.array(list(ss))[idx])




### Values of 'None' are given when no coordinates are available
### for the first/last residues (i.e. disordered residues)
### In this case, we simply take all residues;
### care must be taken to ensure that this does not affect the results,
### either by upstream or downstream processing
def parse_pdb_indices(imin, imax):
    if imin == 'None':
        imin = -50
    else:
        imin = int(''.join([x for x in imin if x.isdigit()]))

    if imax == 'None':
        imax = 500000
    else:
        imax = int(''.join([x for x in imax if x.isdigit()]))
    return imin, imax


def rel_rate_conf_int(df0):
    ci_lo = [9.0, -4.0]
    ci_hi = [12.1, -5.7]
    df1 = df0.copy()
    df2 = df0.copy()
    df1['ln_kf'] = df1.SS_PDB.apply(pred_fold, coefs=ci_lo)
    df2['ln_kf'] = df2.SS_PDB.apply(pred_fold, coefs=ci_hi)
    df1['REL_RATE'] = - df1['ln_kf'] - df1['k_trans'] / df1['AA']
    df2['REL_RATE'] = - df2['ln_kf'] - df2['k_trans'] / df2['AA']

    Xnew = np.linspace(-7, 7, 1000)
    dist = np.array([smooth_dist_kde(Xnew, df.loc[df.REL_RATE.notnull(), 'REL_RATE']) for df in [df0, df1, df2]])
    return dist


def circ_perm(s, i):
    return s[i:] + s[:i]


def assign_quantile(x, quantiles):
    for i, (ql, qu) in enumerate(zip(quantiles[:-1], quantiles[1:])):
        if ql<= x < qu:
            return i


# Count number of each type of secondary structure / disorder:
#   S - Sheet
#   H - Helix
#   . - Coil
#   D - Disorder
# Count as a function of distance from, respectively, the N terminal
# and the C terminal
def pdb_end_stats_disorder_N_C(df, cut=50, N=50, s1='SEQ_PDB', s2='SS_PDB', cat='.DSH'):
    used = set()
    count_N = {x:[{aa:0 for aa in AA_ALPHA} for i in range(N)] for x in list(cat)}
    count_C = {x:[{aa:0 for aa in AA_ALPHA} for i in range(N)] for x in list(cat)}
    for seq, ss in df.loc[:,[s1, s2]].values:
        if len(seq) < cut or len(seq) != len(ss):
            continue
        if seq not in used:
            for j in range(min(int((len(seq)-1)/2), N)):
                count_N[ss[j+1]][j] = add_to_dict(count_N[ss[j+1]][j], seq[j+1])
                count_C[ss[-j-1]][j] = add_to_dict(count_C[ss[-j-1]][j], seq[-j-1])
            used.add(seq)
    print(f"{len(used)} unique sequences counted")
    return count_N, count_C


# N Terminal Enrichment
#
# The probability that the N terminal will contain more
# sheets / helices compared to the C terminal.
# First divide proteins deciles according to REL_RATE;
# then calculate the N terminal enrichment as a function of REL_RATE
def calculate_enrichment(inputs):
    pdb, q = inputs
    # For Fig3C, we want to make sure that the quantiles are the same,
    # so they must be specified to be the same as in Fig3A-B
    # This is only used when getting seperate Eukaryote/Prokaryote data
    if q:
        # Use this one if ln_kf is predicted based on PFDB
#       quantiles = np.array([-8. , -2.6, -2. , -1.5, -1. , -0.6, -0.3,  0. ,  0.4,  0.8,  5.6])
        # Use this one if ln_kf is predicted based on ACPro
        quantiles = np.array([-6.3, -3. , -2.7, -2.4, -2.1, -1.8, -1.7, -1.5, -1.2, -1. ,  1.7])
    else:
        quantiles = pdb['REL_RATE'].quantile(np.arange(0,1.1,.1)).values
    pdb['quant'] = pdb['REL_RATE'].apply(lambda x: assign_quantile(x, quantiles))
    ratio = np.zeros((2, len(quantiles)-1), dtype=float)
    for i, Y in enumerate(['H_ASYM', 'S_ASYM']):
        for j in range(len(quantiles)-1):
            left  = len(pdb.loc[(pdb.quant==j)&(pdb[Y]<0)]) / max(len(pdb.loc[(pdb.quant==j)]), 1)
            right = len(pdb.loc[(pdb.quant==j)&(pdb[Y]>0)]) / max(len(pdb.loc[(pdb.quant==j)]), 1)
            ratio[i,j] = right - left
    return quantiles, ratio


def sheets_rsa_seq_dist(rsa_list, ss_list, quantiles, ss_key='S', cat='BME', N=50):
    count_N = {x:np.zeros(N) for x in cat}
    count_C = {x:np.zeros(N) for x in cat}
    for rsa, ss in zip(rsa_list, ss_list):
        for i in range(min(int(len(rsa)/2), N)):
            if ss[i] in ss_key:
                for j, q in enumerate(quantiles):
                    if rsa[i] <= q:
                        count_N['BME'[j]][i] += 1
                        break
            if ss[-1-i] in ss_key:
                for j, q in enumerate(quantiles):
                    if rsa[-1-i] <= q:
                        count_C['BME'[j]][i] += 1
                        break
    return count_N, count_C
    

def get_rel_rate(df):
    df['REL_RATE'] = - df.ln_kf - np.log10(df.AA / df.k_trans)
    return df


def R_frac_1(df, k=0, cut=0):
    if k:
        return len(df.loc[(df.REL_RATE<=cut)&(df.k_trans==k)]) / max(1, len(df.loc[df.k_trans==k]))
    else:
        return len(df.loc[(df.REL_RATE<=cut)]) / max(1, len(df))


def R_frac_2(df, k=0, cut1=-1, cut2=1):
    if k:
        return len(df.loc[(df.REL_RATE>=cut1)&(df.REL_RATE<=cut2)&(df.k_trans==k)]) / max(1, len(df.loc[df.k_trans==k]))
    else:
        return len(df.loc[(df.REL_RATE>=cut1)&(df.REL_RATE<=cut2)]) / max(1, len(df))







