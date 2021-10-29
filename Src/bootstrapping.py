from collections import defaultdict, Counter
from itertools import product, permutations
import os
from pathlib import Path
import pickle
import time

from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import linregress

import asym_io
from asym_io import PATH_ASYM
import asym_utils as utils
import folding_rate
import scop
import structure

N_PROC = 50


PATH_FIG = PATH_ASYM.joinpath("Figures")
PATH_FIG_DATA = PATH_FIG.joinpath("Data")



#---------------------------------------------------------#
### Run all bootstrapping and sensitivity analyses
### Files are automatically saved, and will each be loaded
### by the necessary figure


def run_all_bootstrapping(pdb, dom):
    boot_SS_asym_and_save(pdb)
    boot_R_and_save(pdb, acpro='')
    boot_enrich_and_save(pdb, dom)
#   pdb_CO_different_params(configs)

    dom_pos_boot(dom)
#   dom_ndom_boot(dom)
    dom_smco_dist_boot(dom, aa=False)
    dom_smco_dist_boot(dom, aa=True)

    boot_scop(pdb)
    boot_ss_max_asym_region(pdb)

    run_bootstrap_fitting_params(pdb)

    pdb_acpro = folding_rate.get_folding_translation_rates(pdb, acpro=True)
    dom_acpro = folding_rate.get_folding_translation_rates(dom, acpro=True)

    boot_R_and_save(pdb_acpro, acpro='_acpro')
    boot_enrich_and_save(pdb_acpro, dom_acpro, acpro='_acpro')



def print_important_statistics(pdb):
    # SCOP domains
    for l, k in zip(['Euk', 'Prok'], [5, 10]):
        idx1 = (pdb.CF.str.len()>0)&(pdb.k_trans==k)
        N_alphabeta  = len(pdb.loc[(idx1)&((pdb.CL=='1000002')|(pdb.CL=='1000003'))])
        N_tot = len(pdb.loc[idx1])
        print(f"Fraction of AB proteins in {l}: {N_alphabeta / N_tot}")

    pass


#---------------------------------------------------------#
### Bootstrapping SS Asym (Fig 1)


def pdb_ss_new_counter(inputs, N=50, cut=0, s1='SEQ_PDB2', s2='SS_PDB2', cat='.DSH'):
    df, N = inputs
    used = set()
    count_N = {x:np.zeros(N) for x in cat}
    count_C = {x:np.zeros(N) for x in cat}
    for seq, ss in df.loc[:,[s1, s2]].values:
        if len(seq) < cut or len(seq) != len(ss):
            continue
        if seq not in used:
            for j in range(min(int((len(seq)-1)/2), N)):
                count_N[ss[j+1]][j] += 1
                count_C[ss[-j-1]][j] += 1
            used.add(seq)
    asym = {x: np.array([count_N[x][i]/max(count_C[x][i],1) for i in range(N)]) for x in cat}
    return count_N, count_C, asym


def generate_sample(df, N=50, replace=True, n_samp=1000):
    for i in range(n_samp):
        yield (df.sample(len(df), replace=replace), N)


def pdb_end_stats_bootstrap(df, n_boot=1000, N=50, cat='.DSH', ci=.95):
    df = df.copy().loc[:, ['SEQ_PDB2', 'SS_PDB2']]
    Nvals = {x:np.zeros((n_boot, N)) for x in cat}
    Cvals = {x:np.zeros((n_boot, N)) for x in cat}
    asym  = {x:np.zeros((n_boot, N)) for x in cat}
    with Pool(N_PROC) as pool:
        for i, (n, c, a) in enumerate(pool.imap_unordered(pdb_ss_new_counter, generate_sample(df, N=N), 5)):
            tot = np.mean([n[x] for x in cat], axis=0)
            for x in cat:
                Nvals[x][i] = n[x] / (tot*2)
                Cvals[x][i] = c[x] / (tot*2)
                asym[x][i] = a[x]

    Nboot = {x:{} for x in cat}
    Cboot = {x:{} for x in cat}
    asymB = {x:{} for x in cat}
    for x in cat:
        Nboot[x]['lo'] = np.array([np.quantile(Nvals[x][:,i], (1-ci)/2.) for i in range(N)])
        Nboot[x]['hi'] = np.array([np.quantile(Nvals[x][:,i], 1 - (1-ci)/2.) for i in range(N)])
        Nboot[x]['mean'] = np.array([np.mean(Nvals[x][:,i]) for i in range(N)])
        Cboot[x]['lo'] = np.array([np.quantile(Cvals[x][:,i], (1-ci)/2.) for i in range(N)])
        Cboot[x]['hi'] = np.array([np.quantile(Cvals[x][:,i], 1 - (1-ci)/2.) for i in range(N)])
        Cboot[x]['mean'] = np.array([np.mean(Cvals[x][:,i]) for i in range(N)])
        asymB[x]['lo'] = np.array([np.quantile(asym[x][:,i], (1-ci)/2.) for i in range(N)])
        asymB[x]['hi'] = np.array([np.quantile(asym[x][:,i], 1 - (1-ci)/2.) for i in range(N)])
        asymB[x]['mean'] = np.array([np.mean(asym[x][:,i]) for i in range(N)])

    return Nboot, Cboot, asymB


def boot_SS_asym_and_save(df):
    df_list = [df,
               df.loc[(df.OC!='Viruses')&(df.k_trans==5)],
               df.loc[(df.OC!='Viruses')&(df.k_trans==10)]]
    for df0, s in zip(df_list, ['', '_euk', '_prok']):
        SS_asym = pdb_end_stats_bootstrap(df0)
        pickle.dump(SS_asym, open(PATH_FIG_DATA.joinpath(f"pdb_ss_dist_boot{s}.pickle"), 'wb'))



#---------------------------------------------------------#
# REL_RATE distribution (Fig 2)


def generate_R_samples(Xgrid, X, n_samp):
    for i in range(n_samp):
        yield Xgrid, np.random.choice(X, size=X.size, replace=True)


def bootstrap_R_distribution(X, n_samp=1000):
    Xgrid = np.linspace(-10, 10, 1000)
    with Pool(N_PROC) as pool:
        Xfit = np.array(pool.starmap(utils.smooth_dist_kde, generate_R_samples(Xgrid, X, n_samp), 5))
    mean = np.mean(Xfit, axis=0)
    lo = np.quantile(Xfit, 0.025, axis=0)
    hi = np.quantile(Xfit, 0.975, axis=0)
    return mean, lo, hi


def boot_R_and_save(pdb, acpro=''):
    ts = time.time()
    df_list = [pdb,
               pdb.loc[(pdb.OC!='Viruses')&(pdb.k_trans==5)],
               pdb.loc[(pdb.OC!='Viruses')&(pdb.k_trans==10)]]
    out_dict = {'grid':np.linspace(-10, 10, 1000)}
    lbls = ['All', 'Euk', 'Prok']
    for l, df in zip(lbls, df_list):
        out_dict[l] = bootstrap_R_distribution(df.loc[df['REL_RATE'].notnull(), 'REL_RATE'])
        print(f"{l} finished: {(time.time()-ts)/60} minutes")
    pickle.dump(out_dict, open(PATH_FIG_DATA.joinpath(f"R_dist{acpro}.pickle"), 'wb'))


#---------------------------------------------------------#
# N terminal enrichment (Fig 3 & 4)


def generate_sample_enrich(df, replace=True, n_samp=1000, q=False):
    df = df.copy().loc[:, ['REL_RATE', 'H_ASYM', 'S_ASYM']]
    for i in range(n_samp):
        yield df.sample(len(df), replace=replace), q


def bootstrap_enrichment(pdb, n_samp=1000, q=False):
    with Pool(N_PROC) as pool:
        enrich = list(pool.imap_unordered(utils.calculate_enrichment, generate_sample_enrich(pdb, n_samp=n_samp, q=q), 20))
    enrich_edges = np.array([e[0] for e in enrich])
    enrich_vals = np.array([e[1] for e in enrich])
    return enrich_edges, enrich_vals


def boot_enrich_and_save(pdb, dom, acpro=''):
    ts = time.time()
    df_list = [pdb,
               pdb.loc[(pdb.OC!='Viruses')&(pdb.k_trans==5)],
               pdb.loc[(pdb.OC!='Viruses')&(pdb.k_trans==10)],
               dom.loc[(dom.NDOM==1)&(np.isfinite(dom.REL_RATE))],
               dom.loc[(dom.NDOM>1)&(np.isfinite(dom.REL_RATE))]]
    file_names = [f"fig3_enrich{acpro}.pickle"] + \
                 [f"fig3_enrich_{a}{acpro}.pickle" for a in ['eukaryote', 'prokaryote']] + \
                 [f"fig4_enrich_{a}{acpro}.pickle" for a in ['single', 'multi']]
    if len(acpro):
        qu = np.array([-6.3, -3.0, -2.7, -2.4, -2.1, -1.8, -1.7, -1.5, -1.2, -1.0,  1.7])
    else:
        qu = np.array([-8.0, -2.6, -1.9, -1.4, -1.0, -0.6, -0.3,  0.0,  0.4,  0.9,  5.6])

    q_arr = [0, qu, qu, 0, 0]

    for df, f, q in zip(df_list, file_names, q_arr):
        edges, vals = bootstrap_enrichment(df, q=q)
        out_dict = {'edges':edges, 'H':vals[:,0,:], 'S':vals[:,1,:]}
        pickle.dump(out_dict, open(PATH_FIG_DATA.joinpath(f), 'wb'))
        print(f"{f} finished: {(time.time()-ts)/60} minutes")



#---------------------------------------------------------#
# Domain Bootstrapping

def dom_smco_fn(df, aa=False, N=30):
    if aa:
        bins = np.arange(0,620,20)
        hist = np.histogram(df['AA'], bins=bins, density=True)[0]
    else:
        bins = np.arange(0,61,2.0)
        hist = np.histogram(df['CO'], bins=bins, density=True)[0]
    return (hist / hist.sum())[:N]


def generate_dom_smco_sample(df, replace=True, n_samp=1000, aa=False):
    for i in range(n_samp):
        yield df.sample(len(df), replace=replace), aa

    
def dom_smco_dist_boot(df, n_samp=1000, N=30, ci=0.95, aa=False):
    out_dict = {}
    for pos in [1,2]:
        with Pool(N_PROC) as pool:
            CO = np.array(list(pool.starmap(dom_smco_fn, generate_dom_smco_sample(df.loc[(df.NDOM==2)&(df.POS==pos)], n_samp=n_samp, aa=aa), 20)))
        LO = np.array([np.quantile(CO[:,i], (1-ci)/2.) for i in range(N)])
        HI = np.array([np.quantile(CO[:,i], 1 - (1-ci)/2.) for i in range(N)])
        MEAN = np.array([np.mean(CO[:,i]) for i in range(N)])
        out_dict[f"pos{pos}"] = [MEAN, LO, HI]
    if aa:
        pickle.dump(out_dict, open(PATH_FIG_DATA.joinpath("dom_aa_dist_boot.pickle"), 'wb'))
    else:
        pickle.dump(out_dict, open(PATH_FIG_DATA.joinpath("dom_smco_dist_boot.pickle"), 'wb'))
    return out_dict



def domain_asym_correlations(df, Y='S_ASYM', q=10, n_samp=1000):
    output = []
    quantiles = df['REL_RATE'].quantile(np.arange(0,1.1,.1)).values
    df['quant'] = df['REL_RATE'].apply(lambda x: utils.assign_quantile(x, quantiles))
    for k, Y in enumerate(['H_ASYM', 'S_ASYM']):
        ratio = np.zeros((n_samp, q), dtype=float)
        for i in range(n_samp):
            df2 = df.sample(n=len(df), replace=True)
            for j in range(q):
                total = max(len(df2.loc[df2.quant==j]), 1)
                left  = len(df2.loc[(df2.quant==j)&(df2[Y]<0)]) / total
                right = len(df2.loc[(df2.quant==j)&(df2[Y]>0)]) / total
                ratio[i,j] = right - left
        MEAN = np.array([ratio[:,i].mean() for i in range(ratio.shape[1])])
        LO   = np.array([np.quantile(ratio[:,i], 0.025) for i in range(ratio.shape[1])])
        HI   = np.array([np.quantile(ratio[:,i], 0.975) for i in range(ratio.shape[1])])
        output.append((quantiles, MEAN, LO, HI))
    return MEAN, LO, HI


def dom_pos_boot(dom):
    dom_pos_boot = defaultdict(dict)
    for i in range(2,5):
        for j in range(1,i+1):
#           dom_pos_boot[i][j] = domain_asym_correlations(dom.loc[(dom.NDOM==i)&(dom.POS==j)&(np.isfinite(dom.REL_RATE))])
            dom_pos_boot[i][j] = bootstrap_enrichment(dom.loc[(dom.NDOM==i)&(dom.POS==j)&(np.isfinite(dom.REL_RATE))], n_samp=1000, q=False)
    pickle.dump(dom_pos_boot, open(PATH_FIG_DATA.joinpath("dom_pos_boot.pickle"), 'wb'))


#ef dom_ndom_boot(dom):
#   dom_ndom_boot = {'single':domain_asym_correlations(dom.loc[(dom.NDOM==1)&(np.isfinite(dom.REL_RATE))]),
#                    'multi':domain_asym_correlations(dom.loc[(dom.NDOM>1)&(np.isfinite(dom.REL_RATE))])}
#   return dom_ndom_boot




#---------------------------------------------------------#
# Folding rates


def evaluate_fit(pdb, l, a, b):
    X = pdb.loc[:, l] if l=='CO' else np.log10(pdb.loc[:, l]) 
    pdb['ln_kf'] = folding_rate.pred_fold(X, [a, b])
    pdb = utils.get_rel_rate(pdb)
    quantiles, ratio = utils.calculate_enrichment((pdb, 0))
    i = np.argmin(ratio[0])
    j = np.argmax(ratio[1])
    Hmax = np.mean(quantiles[i:i+2])
    Smax = np.mean(quantiles[j:j+2])
    df_list = [pdb, pdb.loc[pdb.k_trans==5], pdb.loc[pdb.k_trans==10]]
    frac = [fn(df, k) for df in df_list for fn in [utils.R_frac_1, utils.R_frac_2] for k in [5,10]]
    return [Hmax, Smax] + frac


def generate_fit_inputs(pdb, l, coef_a, coef_b):
    for a, b in product(coef_a, coef_b):
        yield pdb, l, a, b


def map_regression_space(pdb, pfdb, ci=0.01, ngrid=3):
    pdb = pdb.copy().loc[:, ["AA_PDB", "CO", "REL_RATE", "ln_kf", "k_trans", "H_ASYM", "S_ASYM"]]
    Xf_in = [np.log10(pfdb.loc[pfdb.use, s]) for s in ['L']]
    Y = pfdb.loc[pfdb.use, 'log_kf'].values
    lbls = ["AA_PDB"]
    if len(lbls) > 1:
        x_arr = pfdb.loc[pfdb.use, lbls]
    else:
        x_arr = [np.array(pfdb.loc[pfdb.use, lbls].values).reshape(sum(pfdb.use))]
    metrics = {}
    for l, Xf in zip(lbls, Xf_in):
        res = folding_rate.linear_fit(Xf[Xf>0], Y[Xf>0])
        fit_ci = np.array(res.conf_int(ci))
        coef_a = np.linspace(*fit_ci[0], ngrid)
        coef_b = np.linspace(*fit_ci[1], ngrid)
        coef_a = np.linspace( 9.5, 18.0, ngrid)
        coef_b = np.linspace(-8.2, -3.8, ngrid)
        with Pool(N_PROC) as pool:
            m = np.array(list(pool.starmap(evaluate_fit, generate_fit_inputs(pdb, l, coef_a, coef_b))))
        resize = int(m.size / ngrid**2)
        metrics[l] = {'c1':coef_a, 'c2':coef_b, 'met':m.reshape(ngrid, ngrid, resize)}
    return metrics            


def generate_sample_2var(X, Y, n_samp=1000, frac=1):
    for i in range(n_samp):
        idx = np.random.choice(range(len(X)), size=int(len(X)*frac), replace=True)
        yield X[idx], Y[idx]


def bootstrap_fit(pfdb, n_samp=1000, frac=1, xlbl='L', ylbl='log_kf', use_all=False):
    pfdb = pfdb.copy()
    if use_all:
        pfdb['use'] = True
    X = np.array(np.log10(pfdb.loc[pfdb.use, xlbl]))
    Y = np.array(pfdb.loc[pfdb.use, ylbl].values)
    with Pool(N_PROC) as pool:
        return np.array(list(pool.starmap(linregress, generate_sample_2var(X, Y, frac=frac))))


def run_bootstrap_fitting_params(pdb):
    # Load PFDB and ACPro databases
    pfdb = asym_io.load_pfdb()
    acpro = asym_io.load_acpro()

    # Check how the position of maximum asymmetry (Fig 3)
    # changes with k_fold prediction parameters A and B
    met_fit = map_regression_space(pdb, pfdb, ngrid=51)
    pickle.dump(met_fit, open(PATH_FIG_DATA.joinpath("boot_fit_met.pickle"), 'wb'))

    # Check how the position of maximum asymmetry (Fig 3)
    # changes with bootstrapped samples using the restricted PFDB data set
    boot_fit = [bootstrap_fit(pfdb, frac=f) for f in [1, 1/2., 1/3.]]
    pickle.dump(boot_fit, open(PATH_FIG_DATA.joinpath("boot_fit_param.pickle"), 'wb'))
    
    # Check how the position of maximum asymmetry (Fig 3)
    # changes with bootstrapped samples using the full PFDB data set
    pfdb = pfdb.copy()
    pfdb.loc[pfdb.ln_kf_25.notnull(), 'log_kf'] = np.log10(np.exp(pfdb.loc[pfdb.ln_kf_25.notnull(), 'ln_kf_25']))
    boot_fit = [bootstrap_fit(pfdb, frac=f, use_all=True) for f in [1, 1/2., 1/3.]]
    pickle.dump(boot_fit, open(PATH_FIG_DATA.joinpath("boot_fit_param_useall.pickle"), 'wb'))
    
    # Check how the position of maximum asymmetry (Fig 3)
    # changes with bootstrapped samples using the full ACPro data set
    boot_fit = [bootstrap_fit(acpro, frac=f, use_all=True, xlbl='Protein Length', ylbl='ln kf') for f in [1, 1/2., 1/3.]]
    pickle.dump(boot_fit, open(PATH_FIG_DATA.joinpath("boot_fit_param_acpro.pickle"), 'wb'))
    


#---------------------------------------------------------#
# Contact Order


def generate_CO_params(configs, cut_vals, k_vals):
    for con in configs:
        idx = np.where(np.isnan(con)==False)[0]
        for cut in cut_vals:
            for k in k_vals:
                yield con[idx], idx, cut, k


def pdb_CO_different_params(config_list_trimmed):
    with Pool(N_PROC) as pool:                           
        inputs = generate_CO_params(config_list_trimmed, [6, 8, 10, 12], [0,1,2])
        res = pool.starmap(structure.calc_contact_order, inputs, 10)
    pdb_CO = np.array(res).reshape(len(config_list_trimmed), 4, 3)
    np.save(PATH_FIG_DATA.joinpath("pdb_config_CO.npy"), pdb_CO)
    return pdb_CO


#---------------------------------------------------------#
# SCOP fold independent


def run_scop_analysis(pdb, N=50, cat='SH.D'):
    Nvals = {x:np.zeros(N) for x in cat}
    Cvals = {x:np.zeros(N) for x in cat}
    asym  = {x:np.zeros(N) for x in cat}
    countN, countC, a = pdb_ss_new_counter(pdb)
    tot = np.mean([countN[x] for x in cat], axis=0) 
    for x in cat:
        Nvals[x] = countN[x] / tot
        Cvals[x] = countC[x] / tot
        asym[x] = a[x]

    enrich_edges, enrich_vals = utils.calculate_enrichment((pdb, False))
    return Nvals, Cvals, asym, enrich_edges, enrich_vals


def generate_scop_params(pdb, key, n_rep=1000):
    for i in range(n_rep):
        idx = [np.random.choice(v) for v in key.values()]
        yield pdb.loc[idx]


def boot_scop(pdb, ci=0.95, N=50, cat='SH.D'):
    key = scop.get_SCOP_redundant_key(pdb)
    with Pool(N_PROC) as pool:
        res = list(pool.imap_unordered(run_scop_analysis, generate_scop_params(pdb, key), 10))

    Nvals = {x: np.array([r[0][x] for r in res]) for x in cat}
    Cvals = {x: np.array([r[1][x] for r in res]) for x in cat}
    asym  = {x: np.array([r[2][x] for r in res]) for x in cat}
    enrich_edges = np.array([r[3] for r in res])
    enrich_vals  = np.array([r[4] for r in res])

    Nboot = {x:{} for x in cat}
    Cboot = {x:{} for x in cat}
    asymB = {x:{} for x in cat}
    for x in cat:
        Nboot[x]['lo'] = np.array([np.quantile(Nvals[x][:,i], (1-ci)/2.) for i in range(N)])
        Nboot[x]['hi'] = np.array([np.quantile(Nvals[x][:,i], 1 - (1-ci)/2.) for i in range(N)])
        Nboot[x]['mean'] = np.array([np.mean(Nvals[x][:,i]) for i in range(N)])
        Cboot[x]['lo'] = np.array([np.quantile(Cvals[x][:,i], (1-ci)/2.) for i in range(N)])
        Cboot[x]['hi'] = np.array([np.quantile(Cvals[x][:,i], 1 - (1-ci)/2.) for i in range(N)])
        Cboot[x]['mean'] = np.array([np.mean(Cvals[x][:,i]) for i in range(N)])
        asymB[x]['lo'] = np.array([np.quantile(asym[x][:,i], (1-ci)/2.) for i in range(N)])
        asymB[x]['hi'] = np.array([np.quantile(asym[x][:,i], 1 - (1-ci)/2.) for i in range(N)])
        asymB[x]['mean'] = np.array([np.mean(asym[x][:,i]) for i in range(N)])

    out = [Nboot, Cboot, asymB, enrich_edges, enrich_vals] 
    pickle.dump(out, open(PATH_FIG_DATA.joinpath(f"pdb_scop_indep.pickle"), 'wb'))

    return out


def boot_ss_max_asym_region(pdb, ci=0.95, N=50, cat='SH.D'):
    idx = (pdb.AA_PDB>=179)&(pdb.AA_PDB<=416)&(pdb.CO>=21)&(pdb.CO<=35)
    SS_asym = pdb_end_stats_bootstrap(pdb.loc[idx], N=100)
    pickle.dump(SS_asym, open(PATH_FIG_DATA.joinpath(f"pdb_ss_max_asym.pickle"), 'wb'))




