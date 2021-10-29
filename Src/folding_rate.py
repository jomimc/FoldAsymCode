
import numpy as np
import pandas as pd
import statsmodels.api as sm

import asym_io



def pred_fold(X, coefs):
    return coefs[0] + coefs[1] * X


def linear_fit(X, Y):
    X = sm.add_constant(X)
    mod = sm.OLS(Y, X)
    return mod.fit()


def linear_fit_with_ci(X, Y, ci=0.05):
    res = linear_fit(X, Y)
    fit_ci = np.array(res.conf_int(ci))
    lo = fit_ci[[0,1],[0,1]]
    hi = fit_ci[[0,1],[1,0]]
    return res.params, lo, hi


def get_folding_kinetics(pfdb, ci=0, X='L', Y='log_kf'):
    Xf = np.log10(pfdb[X])
    Yf = pfdb[Y]
    if ci != 0:
        return linear_fit_with_ci(Xf, Yf)
    else:
        return linear_fit(Xf, Yf).params


def get_folding_translation_rates(df, which='best', acpro=False, reduce_pfdb=True, only2s=False):
    if acpro:
        pfdb = asym_io.load_acpro()
        reduce_pfdb = False
    else:
        pfdb = asym_io.load_pfdb()
        if only2s:
            pfdb = pfdb.loc[:88]
    if reduce_pfdb:
        pfdb = pfdb.loc[pfdb.use]
    if which == 'best':
        coef = get_folding_kinetics(pfdb)
    else:
        idx = {'lo':1, 'hi':2}[which]
        coef = get_folding_kinetics(pfdb, ci=0.05)[idx]

    df['ln_kf'] = df.AA_PDB.apply(lambda x: pred_fold(np.log10(x), coef))
    df['T_TRANS'] = np.log10(df.AA_PDB / df.k_trans)
    df['REL_RATE'] = - df['ln_kf'] - df['T_TRANS']
    return df


def kdb_results(pdb, dom, kdb):
    pdb = pdb.copy()
    dom = dom.copy()
    coef = linear_fit(np.log10(kdb['Protein Length']), kdb['ln kf']).params
    pdb['ln_kf'] = pred_fold(np.log10(pdb.AA_PDB), coef)
    pdb = get_rel_rate(pdb)

    dom['ln_kf'] = pred_fold(np.log10(dom.AA_PDB), coef)
    dom = get_rel_rate(dom)

    boot_R_and_save(pdb, kdb='_kdb')
    boot_enrich_and_save(pdb, dom, kdb='_kdb')



