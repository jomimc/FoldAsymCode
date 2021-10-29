from collections import defaultdict, Counter
from itertools import product, permutations
from glob import glob
import json
import os
from pathlib import Path
import pickle
import sqlite3
import string
import sys
import time

import matplotlib as mpl
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from multiprocessing import Pool
import numpy as np
import pandas as pd
from palettable.colorbrewer.qualitative import Paired_12
from palettable.colorbrewer.diverging import PuOr_5, RdYlGn_6, PuOr_10, RdBu_10
from palettable.scientific.diverging import Cork_10
from scipy.spatial import distance_matrix, ConvexHull, convex_hull_plot_2d
from scipy.stats import linregress, pearsonr, lognorm
import seaborn as sns
import svgutils.compose as sc

import asym_io
from asym_io import PATH_BASE, PATH_ASYM, PATH_ASYM_DATA
import asym_utils as utils
import folding_rate
import paper_figs
import structure

PATH_FIG = PATH_ASYM.joinpath("Figures")
PATH_FIG_DATA = PATH_FIG.joinpath("Data")


custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
c_helix = custom_cmap[2]
c_sheet = custom_cmap[10]
col = [c_helix, c_sheet, "#CB7CE6", "#79C726"]


####################################################################
### SI Figures

####################################################################
### FIG 1

def fig1(df, nx=3, ny=3, N=50):
    fig, ax = plt.subplots(nx,ny, figsize=(12,12))
    ax = ax.reshape(ax.size)
    fig.subplots_adjust(hspace=.5)

    lbls = ['Helix', 'Sheet', 'Coil', 'Disorder']
    cat = 'HS.D'

    scop_desc = {row[1]:row[2] for row in pd.read_csv(PATH_BASE.joinpath('SCOP/scop-des-latest.txt')).itertuples()}
    CF_count = sorted(df.CF.value_counts().items(), key=lambda x:x[1], reverse=True)[1:]

    bold_idx = [0, 1, 2, 6, 8]

    for i in range(nx*ny):
        cf_id, count = CF_count[i]
        countN, countC = utils.pdb_end_stats_disorder_N_C(df.loc[df.CF==cf_id], N=N, s1='SEQ_PDB2', s2='SS_PDB2')
        base = np.zeros(len(countN['S']), dtype=float)
        Yt = np.array([[sum(p.values()) for p in countN[s]] for s in cat]).sum(axis=0)
        X = np.arange(base.size)
        for j, s in enumerate(cat):
            YN = np.array([sum(p.values()) for p in countN[s]])
            YC = np.array([sum(p.values()) for p in countC[s]])
            ax[i].plot(YN/Yt, '-', c=col[j], label=f"{s} N")
            ax[i].plot(YC/Yt, ':', c=col[j], label=f"{s} C")
        if i in bold_idx:
            ax[i].set_title(f"{scop_desc[int(cf_id)][:40]}\nTotal sequences: {count}", fontweight='bold')
        else:
            ax[i].set_title(f"{scop_desc[int(cf_id)][:40]}\nTotal sequences: {count}")
        ax[i].set_xlabel('Sequence distance from ends')
        if not i%3:
            ax[i].set_ylabel('Secondary\nstructure\nprobability')
    handles = [Line2D([0], [0], ls=ls, c=c, label=l) for ls, c, l in zip(['-', '--'], ['k']*2, ['N', 'C'])] + \
              [Line2D([0], [0], ls='-', c=c, label=l) for l, c in zip(lbls, col)]
    ax[1].legend(handles=handles, bbox_to_anchor=(1.40, 1.45), frameon=False,
                 ncol=6, columnspacing=1.5, handlelength=2.0)


    fig.savefig(PATH_FIG.joinpath("si1.pdf"), bbox_inches='tight')


####################################################################
### FIG 2


def fig2():
    pfdb = asym_io.load_pfdb()
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    fig.subplots_adjust(wspace=0.3)
    X1 = np.log10(pfdb.loc[pfdb.use, 'L'])
    X2 = np.log10(pfdb.loc[pfdb.use, 'CO'])
    Y = pfdb.loc[pfdb.use, 'log_kf']
    sns.regplot(X1, Y, ax=ax[0])
    sns.regplot(X2, Y, ax=ax[1])

    print(pearsonr(X1, Y))
    print(pearsonr(X2, Y))

    ax[0].set_ylabel(r'$\log_{10} k_f$')
    ax[1].set_ylabel(r'$\log_{10} k_f$')
    ax[0].set_xlabel(r'$\log_{10}$ Sequence Length')
    ax[1].set_xlabel(r'$\log_{10}$ Contact Order')

    fs = 14
    for i, b in zip([0,1], list('ABCDEFGHI')):
        ax[i].text( -0.10, 1.05, b, transform=ax[i].transAxes, fontsize=fs)
    

    fig.savefig(PATH_FIG.joinpath("si2.pdf"), bbox_inches='tight')




####################################################################
### FIG 3

def fig3(pdb, Y='S_ASYM'):
    LO = folding_rate.get_folding_translation_rates(pdb.copy(), which='lo')
    HI = folding_rate.get_folding_translation_rates(pdb.copy(), which='hi')
    fig, ax = plt.subplots()
    lbls = ['Fit', r"$95\% CI$", r"$95\% CI$"]
    for i, d in enumerate([pdb, LO, HI]):
        print(f"{i}:  frac R less than 0 = {utils.R_frac_1(d)}")
        print(f"{i}:  Euk frac (.1 < R < 10) = {utils.R_frac_2(d, k=5)}")
        print(f"{i}:  Prok frac (.1 < R < 10) = {utils.R_frac_2(d, k=10)}")
        print(f"{i}:  frac R faster than 'speed-limit' = {utils.R_frac_3(d)}")
        print(f"{i}:  frac R slower than 20 minutes = {utils.R_frac_4(d)}")
        print()
        sns.distplot(d['REL_RATE'], label=lbls[i], color=col[i])
    ax.legend(loc='best', frameon=False)
    ax.set_xlim(-6, 6)
    ax.set_xlabel(r'$\log_{10}R$')
    ax.set_ylabel('Density')

    fig.savefig(PATH_FIG.joinpath("si3.pdf"), bbox_inches='tight')



####################################################################
### FIG 4

def fig4(pdb, Y='S_ASYM'):
    LO = folding_rate.get_folding_translation_rates(pdb.copy(), which='lo')
    HI = folding_rate.get_folding_translation_rates(pdb.copy(), which='hi')
    # For the results using only 2-state proteins... 
#   HI = folding_rate.get_folding_translation_rates(pdb.copy(), which='best', only2s=True)
    fig = plt.figure(figsize=(8,10.5))
    gs  = GridSpec(5,12, wspace=0.5, hspace=0.0, height_ratios=[1,0.5,1,0.5,1.5])
    ax = [fig.add_subplot(gs[i*2,j*4:(j+1)*4]) for i in [0,1] for j in [0,1,2]] + \
         [fig.add_subplot(gs[4,:5]), fig.add_subplot(gs[4,7:])]

    X = np.arange(10)
    width = .35
    ttls = [r'$\alpha$ Helix', r'$\beta$ Sheet']
    lbls = [r'$E_{\alpha}$', r'$E_{\beta}$']
    custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
    c_helix = custom_cmap[0]
    c_sheet = custom_cmap[12]
    col = [c_helix, c_sheet]
    bins = np.linspace(-0.20, 0.20, 80)
    width = np.diff(bins[:2])
    X = bins[:-1] + width * 0.5
    mid = 39
    sep = 0.05

    for k, pdb in enumerate([LO, HI]):
        quantiles = pdb['REL_RATE'].quantile(np.arange(0,1.1,.1)).values
        pdb['quant'] = pdb['REL_RATE'].apply(lambda x: utils.assign_quantile(x, quantiles))

        enrich_data = pickle.load(open(PATH_FIG_DATA.joinpath("fig3_enrich.pickle"), 'rb'))
        for i, Y in enumerate(['H_ASYM', 'S_ASYM']):
            for j in range(len(quantiles)-1):
                hist, bins = np.histogram(pdb.loc[pdb.quant==j, Y], bins=bins)
                hist = hist / hist.sum()
                if i:
                    ax[k*3+i].bar(X[:mid], (hist/hist.sum())[:mid], width, bottom=[sep*j]*mid, color='grey', alpha=.5)
                    ax[k*3+i].bar(X[-mid:], (hist/hist.sum())[-mid:], width, bottom=[sep*j]*mid, color=col[i], alpha=.5)
                else:
                    ax[k*3+i].bar(X[:mid], (hist/hist.sum())[:mid], width, bottom=[sep*j]*mid, color=col[i], alpha=.5)
                    ax[k*3+i].bar(X[-mid:], (hist/hist.sum())[-mid:], width, bottom=[sep*j]*mid, color='grey', alpha=.5)
                ax[k*3+i].plot(X[:mid], (hist/hist.sum()+sep*j)[:mid], '-', c='k', alpha=.5)
                ax[k*3+i].plot(X[-mid:], (hist/hist.sum()+sep*j)[-mid:], '-', c='k', alpha=.5)

            mean = np.mean(enrich_data[Y[0]], axis=0)
            lo   = np.abs(mean - np.quantile(enrich_data[Y[0]], 0.025, axis=0))
            hi   = np.abs(mean - np.quantile(enrich_data[Y[0]], 0.975, axis=0))

            ax[k*3+2].barh([sep*j+(i+.7)*sep/3 for j in range(10)], mean, sep/3, xerr=(lo, hi), color=col[i], ec='k', alpha=.5, label=lbls[i], error_kw={'lw':.8})
            ax[k*3+2].plot([0,0], [-0.05, 0.5], '-', c='k', lw=.1)
        for i in [0,2]:
            ax[k*3+i].set_yticks(np.arange(len(quantiles))*sep)
            ax[k*3+i].set_yticklabels([round(x,1) for x in quantiles])
        for i in range(2):
            ax[k*3+i].spines['top'].set_visible(False)
            ax[k*3+i].spines['right'].set_visible(False)
        for i in range(1,3):
            ax[k*3+i].spines['left'].set_visible(False)
            ax[k*3+i].spines['top'].set_visible(False)
        for i in range(3):
            ax[k*3+i].set_ylim(0-sep/4, (0.5+sep/4)*1.05)
        ax[k*3+1].set_yticks([])
        ax[k*3+2].yaxis.set_label_position('right')
        ax[k*3+2].yaxis.tick_right()
        ax[k*3+0].set_xlabel(r"asym$_{\alpha}$")
        ax[k*3+1].set_xlabel(r"asym$_{\beta}$")
        ax[k*3+0].set_ylabel(r'$\log_{10}R$')
        ax[k*3+2].set_xlabel('N terminal\nEnrichment')

    plot_metric_space(fig, ax[6:])

    fs = 14
    for i, b in zip([0,3,6], list('ABCDEFGHI')):
        ax[i].text( -0.20, 1.05, b, transform=ax[i].transAxes, fontsize=fs)
    
    fig.savefig(PATH_FIG.joinpath("si4.pdf"), bbox_inches='tight')


def get_ci_index(X, Y):
    xlo = np.quantile(X, 0.025)
    xhi = np.quantile(X, 0.975)
    ylo = np.quantile(Y, 0.025)
    yhi = np.quantile(Y, 0.975)
    return np.where((X>=xlo)&(X<=xhi)&(Y>=ylo)&(Y<=yhi))[0]


def plot_hull(boot_fit, patt, ax='', c='k', lw=1):
    idx = get_ci_index(*boot_fit[:,:2].T)
    tmp = boot_fit[idx].copy()
    hull = ConvexHull(np.array([boot_fit[idx,1], boot_fit[idx, 0]]).T)
    for simplex in hull.simplices:
        if not isinstance(ax, str):
            ax.plot(tmp[simplex, 1], tmp[simplex, 0], patt, c=c, lw=lw)
        else:
            plt.plot(tmp[simplex, 1], tmp[simplex, 0], patt, c=c, lw=lw)


def plot_metric_space(fig, ax):
    fit = pickle.load(open(PATH_FIG_DATA.joinpath("boot_fit_met.pickle"), 'rb'))['AA']
    boot_fit = pickle.load(open(PATH_FIG_DATA.joinpath("boot_fit_param.pickle"), 'rb'))
    boot_fit_0 = pickle.load(open(PATH_FIG_DATA.joinpath("boot_fit_param_useall.pickle"), 'rb'))
    X, Y = np.meshgrid(fit["c1"], fit["c2"])
    cmap = colors.ListedColormap(sns.diverging_palette(230, 22, s=100, l=47, n=8))
    bounds = np.linspace(-2, 2, 9)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    im = []
    ttls = ['Helices', 'Sheets']
    for i in range(2):
        im = ax[i].contourf(X, Y, fit['met'][:,:,i], bounds, cmap=cmap, vmin=-2, vmax=2, norm=norm)
        cbar = fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04, norm=norm, boundaries=bounds, ticks=bounds)
        cbar.set_label(r"$R_{\mathregular{max}}$", labelpad=-5)
        ax[i].set_xlabel('A')
        ax[i].set_xlim(X.min(), X.max())
        ax[i].set_ylabel('B')
        ax[i].set_ylim(Y.max(), Y.min())
        ax[i].invert_yaxis()
        ax[i].set_aspect((np.max(X)-np.min(X))/(np.max(Y)-np.min(Y)))
        ax[i].set_title(ttls[i])

    col = ['k', '#79C726']
    for i, boofi in enumerate([boot_fit, boot_fit_0]):
        for j in range(2):
            for bf, p in zip(boofi, ['-', ':']):
                plot_hull(bf, p, ax[j], c=col[i])
    c1  = [13.77, -6.07]
    c1a = [11.36553036, -4.87716477]
    c1b = [16.17819934, -7.27168306]
    patt = ['*', 'o', 'o']
    lbls = ['Fit', r"$95\% CI$", r"$95\% CI$"]
    col = "#CB7CE6"
    for i in range(2):
        for coef, p, l in zip([c1, c1a, c1b], patt, lbls):
            ax[i].plot([coef[0]], [coef[1]], p, label=l, fillstyle='none', ms=10, c=col, mew=2)
        ax[i].legend(loc='best', frameon=False)



####################################################################
### FIG 5

def fig5():
    fig, ax = plt.subplots(2,1)
    fig.subplots_adjust(hspace=0.3)
    bins = np.arange(0,620,20)
    X = [bins[:-1] + np.diff(bins[:2])]
    bins = np.arange(0,61,2.0)
    X.append(bins[:-1] + np.diff(bins[:2]))
    yellows = sns.diverging_palette(5, 55, s=95, l=77, n=13)
    pinks = sns.diverging_palette(5, 55, s=70, l=52, n=13)
    col = [yellows[12], pinks[0]]
    col2 = [yellows[10], pinks[3]]

    data = [pickle.load(open(PATH_FIG_DATA.joinpath(f"dom_{x}_dist_boot.pickle"), 'rb')) for x in ['aa', 'smco']]
    for j in range(2):
        for i in [1,2]:
            MEAN, LO, HI = [np.array(x) for x in data[j][f"pos{i}"]]
            ax[j].plot(X[j], MEAN, '--', c=col[i-1], label=f'position {i}')
            ax[j].fill_between(X[j], LO, HI, color=col2[i-1], alpha=0.5)
    ax[0].set_xlabel('Sequence Length')
    ax[1].set_xlabel('Contact Order')
    ax[0].set_ylabel('Density')
    ax[1].set_ylabel('Density')
    ax[0].legend(loc='upper right', frameon=False)

    fig.savefig(PATH_FIG.joinpath("si5.pdf"), bbox_inches='tight')



####################################################################
### FIG 6

def fig6(X='REL_RATE', Y='S_ASYM'):
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    fig.subplots_adjust(hspace=0.7, wspace=0.3)
    sep = 0.40
    col = Paired_12.hex_colors[5]
    ttls = [f"Position {i}" for i in range(1,3)]
    dom_pos_boot = pickle.load(open(PATH_FIG_DATA.joinpath("dom_pos_boot.pickle"), 'rb'))
    custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
    c_helix = custom_cmap[2]
    c_sheet = custom_cmap[11]
    col = [c_helix, c_sheet, "#CB7CE6", "#79C726"]
#   ttls = ["Two-domain", "Three-domain"]
    xlbls = [r'$E_{\alpha}$', r'$E_{\beta}$']
    for i in range(2):
        for j, (pos, dat) in enumerate(dom_pos_boot[2].items()):
            quantiles = dat[0].mean(axis=0)
            mean = dat[1][:,i,:].mean(axis=0)
            lo   = np.abs(np.quantile(dat[1][:,i,:], 0.025, axis=0) - mean)
            hi   = np.abs(np.quantile(dat[1][:,i,:], 0.975, axis=0) - mean)
            ax[j].bar(np.arange(10)+(i+1)*sep, mean, sep, yerr=(lo, hi), color=col[i], label=xlbls[i], alpha=0.7, error_kw={'lw':.8})
            ax[j].set_xticks(np.arange(len(quantiles)))
            ax[j].set_xticklabels(np.round(quantiles, 1), rotation=90)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_title(ttls[i], loc='left')
        ax[i].set_xlabel(r'$\log_{10}R$')
#       ax[i,k].set_ylabel('N terminal\nEnrichment')
        ax[i].set_ylabel("N Terminal Enrichment")
    ax[0].legend(bbox_to_anchor=(1.17, 1.12), frameon=False, ncol=3)

    fig.savefig(PATH_FIG.joinpath("si6.pdf"), bbox_inches='tight')



####################################################################
### FIG 7

def fig7(pdb, Y='D_ASYM'):
    fig, ax = plt.subplots(3,3, figsize=(12,8))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    sep = 0.05
    col = Paired_12.hex_colors[7]
    xlbls = [r'$\log_{10} R$', 'Sequence Length', 'Contact Order']
    ttls = ['Full sample', 'Eukaryotes', 'Prokaryotes']
    for k, df in enumerate([pdb, pdb.loc[pdb.k_trans==5], pdb.loc[pdb.k_trans==10]]):
        for i, X in enumerate(['REL_RATE', 'AA_PDB', 'CO']):
            quantiles = df[X].quantile(np.arange(0,1.1,.1)).values
            df['quant'] = df[X].apply(lambda x: utils.assign_quantile(x, quantiles))
            ratio = []
            for j in range(len(quantiles)-1):
                left  = len(df.loc[(df.quant==j)&(df[Y]<0)]) / max(1, len(df.loc[(df.quant==j)]))
                right = len(df.loc[(df.quant==j)&(df[Y]>0)]) / max(1, len(df.loc[(df.quant==j)]))
                ratio.append((right - left))
#           print(ratio)

            ax[i,k].bar([sep*j+sep/2 for j in range(10)], ratio, sep/2, color=[col if r > 0 else 'grey' for r in ratio], alpha=.5)
            ax[i,k].set_xticks(np.arange(len(quantiles))*sep)
            if i == 1:
                ax[i,k].set_xticklabels([int(x) for x in quantiles], rotation=90)
            else:
                ax[i,k].set_xticklabels([round(x,1) for x in quantiles], rotation=90)
            ax[i,k].set_xlabel(xlbls[i])
            ax[i,k].set_ylabel('N terminal\nEnrichment')
        ax[0,k].set_title(ttls[k])

    fig.savefig(PATH_FIG.joinpath("si7.pdf"), bbox_inches='tight')



####################################################################
### FIG 8

def fig8(df_pdb):
    fig = plt.figure()
    gs  = GridSpec(2,1, wspace=0.0, height_ratios=[.5,1])
    ax = [fig.add_subplot(gs[1,0]), fig.add_subplot(gs[0,0])]
    X = np.arange(-3, 3, 0.01)
    Y = np.array([(10**x + 1)/max(10**x, 1) for x in X])
    Y2 = (1+10**X) / np.array([max(1, 10**x+30./100.) for x in X]) 
    ax[0].plot(X, Y, '-', label=r"$\tau_{ribo}=0$")
    ax[0].plot(X, Y2, ':', label=r"$\tau_{ribo}=0.3\tau_{trans}$")

    lbls = ['1ILO', '2OT2', '3BID']
    patt = ['o', 's', '^']
    for l, p in zip(lbls, patt):
        X, Y = np.load(PATH_FIG_DATA.joinpath(f"{l}.npy"))
        ax[0].plot(X, Y, p, label=l, alpha=0.5, mec='k', ms=7)

    ax[0].set_xlim(-2.3, 2.3)
    ax[0].set_ylim(1, 2.05)
    ax[0].set_xlabel(r'$\log_{10} R$')
    ax[0].set_ylabel("Speed-up")
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].legend(loc='upper right', frameon=False, bbox_to_anchor=(1.05, 1.00), ncol=1, labelspacing=.1)

    fig8a(df_pdb, ax[1])

    fig.savefig(PATH_FIG.joinpath("si8.pdf"), bbox_inches='tight')


def fig8a(df_pdb, ax):
    lbls = ['2OT2', '1ILO', '3BID']
    idx = [98212, 19922, 127370]
    SS = df_pdb.loc[idx, 'SS_PDB2'].values
    custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
    col_key = {'.':'grey', 'D':'grey', 'H':custom_cmap[3], 'S':custom_cmap[9]}
    ec_key = {'.':'grey', 'D':'grey', 'H':custom_cmap[1], 'S':custom_cmap[11]}
    wid_key = {'.':0.1, 'D':0.1, 'H':0.3, 'S':0.3}
    lw_key = {'.':0.7, 'D':0.7, 'H':1.5, 'S':1.5}

    for i, ss in enumerate(SS):
        left = 0.
        for j, strand in enumerate(new_figs.generate_strand(ss)):
            s = strand[0]
            ax.barh([i], [len(strand)], wid_key[s], left=[left], color=col_key[s], ec=ec_key[s], linewidth=lw_key[s])
            left += len(strand) + 0.20

    ax.annotate("N", xy=(-0.01, 1.0), xycoords='axes fraction')
    ax.annotate("C", xy=(0.59, 1.0), xycoords='axes fraction')

    for pos in ['left', 'right', 'top', 'bottom']:
        ax.spines[pos].set_visible(False)

    col = np.array(custom_cmap)[[3,9,1,11]]
    ax.legend(handles=[mpatches.Patch(fc=c1, ec=c2, label=l) for c1, c2, l in zip(col[:2], col[2:], ['Helix', 'Sheet'])],
                 loc='upper right', frameon=False, ncol=1, bbox_to_anchor=(0.95, 1.10))
    ax.set_xticks([])
    ax.set_yticks(range(3))
    ax.set_yticklabels(lbls)
    ax.tick_params(axis='y', which='major', length=0, pad=10)



####################################################################
### FIG 9

def fig9(pdb, s='S'):
    pdb = pdb.loc[(pdb.USE_RSA)]
    pdb = pdb.loc[(pdb.SS_PDB2.str.len()==pdb.RSA.apply(len))]
    
    path = PATH_FIG_DATA.joinpath("RSA_quantiles.pickle")
    if path.exists():
        quantiles, euk_quantiles, prok_quantiles = pickle.load(open(path, 'rb'))
    else:
        quantiles = [np.quantile([x for y in pdb['RSA'] for x in y if np.isfinite(x)], x/3) for x in range(1,4)]
        euk_quantiles = [np.quantile([x for y in pdb.loc[pdb.k_trans==5, 'RSA'] for x in y if np.isfinite(x)], x/3) for x in range(1,4)]
        prok_quantiles = [np.quantile([x for y in pdb.loc[pdb.k_trans==10, 'RSA'] for x in y if np.isfinite(x)], x/3) for x in range(1,4)]
        pickle.dump([quantiles, euk_quantiles, prok_quantiles], open(path, 'wb'))

    print(quantiles)

#   fig, ax = plt.subplots(4,3, figsize=(8,8))
#   fig.subplots_adjust(wspace=0.5)
    fig = plt.figure(figsize=(12,9))
    gs  = GridSpec(5,3, wspace=0.3, height_ratios=[1,1,1,1,1])
    ax = [fig.add_subplot(gs[j,i]) for i in range(3) for j in [0,1]] + \
         [fig.add_subplot(gs[j,i]) for i in range(3) for j in [3,4]]

    print("All proteins, all SS")
    fig9a(pdb['RSA'], pdb['SS_PDB2'], quantiles, ax[:2], s='SH.D')
    print("euk proteins, all ss")
    fig9a(pdb.loc[pdb.k_trans==5, 'RSA'], pdb.loc[pdb.k_trans==5, 'SS_PDB2'], euk_quantiles, ax[2:4], s='SH.D')
    print("Prok proteins, all SS")
    fig9a(pdb.loc[pdb.k_trans==10, 'RSA'], pdb.loc[pdb.k_trans==10, 'SS_PDB2'], prok_quantiles, ax[4:6], s='SH.D')
 
    print("Euk proteins, only SHC")
    fig9a(pdb.loc[pdb.k_trans==5, 'RSA'], pdb.loc[pdb.k_trans==5, 'SS_PDB2'], euk_quantiles, ax[6:8], s='SH.')
    print("Euk proteins, only S")
    fig9a(pdb.loc[pdb.k_trans==5, 'RSA'], pdb.loc[pdb.k_trans==5, 'SS_PDB2'], euk_quantiles, ax[8:10], s='S')
    print("Prok proteins, only S")
    fig9a(pdb.loc[pdb.k_trans==10, 'RSA'], pdb.loc[pdb.k_trans==10, 'SS_PDB2'], prok_quantiles, ax[10:12], s='S')


    ttls = ['All proteins\nAll residues', 'Eukaryotic proteins\nAll residues', 'Prokaryotic proteins\nAll residues',
            'Eukaryotic proteins\nHelix, sheet and coil', 'Eukaryotic proteins\nOnly Sheets', 'Prokaryotic proteins\nOnly Sheets']
    col = np.array(list(Paired_12.hex_colors))[[0,2,4,6]]
    lbls = ['Buried', 'Middle', 'Exposed']
    ax[0].set_ylabel('Solvent accessibility\nprobability')
    ax[1].set_ylabel('Solvent accessibility\nasymmetry\n$\\log_2 (N / C)$')
    ax[6].set_ylabel('Solvent accessibility\nprobability')
    ax[7].set_ylabel('Solvent accessibility\nasymmetry\n$\\log_2 (N / C)$')
    handles = [Line2D([0], [0], ls=ls, c=c, label=l) for ls, c, l in zip(['-', '--'], ['k']*2, ['N', 'C'])] + \
              [Line2D([0], [0], ls='-', c=c, label=l) for l, c in zip(lbls, col)]
    ax[8].legend(handles=handles, bbox_to_anchor=(1.30, 1.85), frameon=False,
                 ncol=5, columnspacing=1.5, handlelength=2.0, labelspacing=2.0)


    for i, a in enumerate(ax):
        if i % 2:
            ax[i].set_xticks(range(0, 60, 10))
            ax[i].set_xlabel('Sequence distance from ends')
        else:
            ax[i].set_xticks([])
            ax[i].set_title(ttls[i//2])
        ax[i].set_xlim(0, 50)

    fig.savefig(PATH_FIG.joinpath("si9.pdf"), bbox_inches='tight')
    

def fig9a(rsa_list, ss_list, quantiles, ax, s='S'):
    cat = 'BME'
    countN, countC = utils.sheets_rsa_seq_dist(rsa_list, ss_list, quantiles, ss_key=s)
    col = np.array(list(Paired_12.hex_colors))[[0,2,4,6]]
    base = np.zeros(len(countN[cat[0]]), dtype=float)
    YtN = np.array(list(countN.values())).sum(axis=0)
    YtC = np.array(list(countC.values())).sum(axis=0)

    X = np.arange(base.size)
    for i, s in enumerate(cat):
        YN = countN[s]
        YC = countC[s]
        ax[0].plot(YN/YtN, '-', c=col[i], label=f"{s} N")
        ax[0].plot(YC/YtC, ':', c=col[i], label=f"{s} C")
        ax[1].plot(np.log2(YN/YC*YtC/YtN), '-', c=col[i], label=f"{s}")
        print(s, np.round((np.sum(YN[:20]) / np.sum(YtN[:20])) / (np.sum(YC[:20]) / np.sum(YtC[:20])), 2))
    ax[1].plot([0]*base.size, ':', c='k')
    ax[0].set_ylim(0,1)
    ax[1].set_ylim(-1,1)
    for a in ax:
        a.set_xlim(X[0], X[-1])



####################################################################
### FIG 10

def fig10(pdb):
    pfdb = asym_io.load_pfdb()
    acpro = asym_io.load_acpro()
    fig = plt.figure(figsize=(12,9))
    gs  = GridSpec(3,7, wspace=0.0, width_ratios=[5,0.2,5,0.4,3,1.0,6], height_ratios=[1,.3,1])
    ax = [fig.add_subplot(gs[2,i*2]) for i in range(4)] + \
         [fig.add_subplot(gs[0,0:3]), fig.add_subplot(gs[0,5:])]

#   sns.distplot(pdb.ln_kf, ax=ax[5], label='PDB - PFDB fit', hist=False)

    pdb = pdb.copy()
    coef = folding_rate.linear_fit(np.log10(acpro['L']), acpro['log_kf']).params
    pdb['ln_kf'] = folding_rate.pred_fold(np.log10(pdb.AA_PDB), coef)
    pdb = utils.get_rel_rate(pdb)

    fig10a(fig, ax[4])
    fig10b(fig, ax[:4], pdb)

#   sns.distplot(pdb.ln_kf, ax=ax[5], label='PDB - ACPro fit', hist=False)
#   sns.distplot(pfdb.log_kf, ax=ax[5], label='PFDB data', kde=False, norm_hist=True)
#   sns.distplot(acpro["ln kf"], ax=ax[5], label='KDB data', kde=False, norm_hist=True)
    sns.regplot(np.log10(acpro['L']), acpro['log_kf'], label='ACPro data', scatter_kws={"alpha":0.5})
    sns.regplot(np.log10(pfdb.loc[pfdb.use, 'L']), pfdb.loc[pfdb.use, 'log_kf'], label='PFDB data', scatter_kws={"alpha":0.5})
    ax[5].legend(loc='best', frameon=False)

    ax[5].set_xlabel(r"$\log_{10}L$")
    ax[5].set_ylabel(r"$\log_{10}k_f$")

    fs = 14
    for i, b in zip([4,5,0,2,3], list('ABCDEFGHI')):
        ax[i].text( -0.20, 1.16, b, transform=ax[i].transAxes, fontsize=fs)
    
    fig.savefig(PATH_FIG.joinpath("si10.pdf"), bbox_inches='tight')


def fig10a(fig, ax):
    Rdist_data = pickle.load(open(PATH_FIG_DATA.joinpath("R_dist_acpro.pickle"), 'rb'))
    custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
    c_helix = custom_cmap[2]
    c_sheet = custom_cmap[10]
    col = [c_helix, c_sheet, "#CB7CE6", "#79C726"]
    lbls = ['All', 'Prokaryotes', 'Eukaryotes']

    for i, k in enumerate(['All', 'Prok', 'Euk']):
        ax.plot(Rdist_data['grid'], Rdist_data[k][0], '-', c=col[i], label=lbls[i])
        ax.fill_between(Rdist_data['grid'], Rdist_data[k][1], Rdist_data[k][2], color=col[i], alpha=0.5)
    ax.plot([0,0], [0, 0.60], ':', c='k', alpha=0.7)
    ax.set_xlabel(r'$\log_{10} R$')
    ax.set_ylabel('Density')
    ax.set_xticks(np.arange(-6, 5, 2))
    ax.set_xlim(-7, 2)
    ax.set_ylim(0, 0.60)
    ax.legend(loc='upper center', bbox_to_anchor=(0.55, 1.17), frameon=False, ncol=3, columnspacing=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def fig10b(fig, ax, pdb, Y='S_ASYM'):
    ft = 12

    X = np.arange(10)
    width = .35
    ttls = [r'$\alpha$ Helix', r'$\beta$ Sheet']
    lbls = [r'$E_{\alpha}$', r'$E_{\beta}$']
#   col = np.array(Paired_12.hex_colors)[[1,5]]
    custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
    c_helix = custom_cmap[0]
    c_sheet = custom_cmap[12]
    col = [c_helix, c_sheet]
    bins = np.linspace(-0.20, 0.20, 80)
    width = np.diff(bins[:2])
    X = bins[:-1] + width * 0.5
    mid = 39
    sep = 0.05


    enrich_data = pickle.load(open(PATH_FIG_DATA.joinpath("fig3_enrich_acpro.pickle"), 'rb'))
    quantiles = enrich_data['edges'].mean(axis=0)
    for i, Y in enumerate(['H_ASYM', 'S_ASYM']):
        for j in range(len(quantiles)-1):
            hist, bins = np.histogram(pdb.loc[pdb.quant==j, Y], bins=bins)
            hist = hist / hist.sum()
#           total = len(pdb)/10
#           left  = len(pdb.loc[(pdb.quant==j)&(pdb[Y]<0)]) / total
#           right = len(pdb.loc[(pdb.quant==j)&(pdb[Y]>0)]) / total
#           print(Y, j, ''.join([f"{x:6.3f}" for x in [left, right, left/right, right / left]]))
            if i:
                ax[i].bar(X[:mid], (hist/hist.sum())[:mid], width, bottom=[sep*j]*mid, color='grey', alpha=.5)
                ax[i].bar(X[-mid:], (hist/hist.sum())[-mid:], width, bottom=[sep*j]*mid, color=col[i], alpha=.5)
            else:
                ax[i].bar(X[:mid], (hist/hist.sum())[:mid], width, bottom=[sep*j]*mid, color=col[i], alpha=.5)
                ax[i].bar(X[-mid:], (hist/hist.sum())[-mid:], width, bottom=[sep*j]*mid, color='grey', alpha=.5)
            ax[i].plot(X[:mid], (hist/hist.sum()+sep*j)[:mid], '-', c='k', alpha=.5)
            ax[i].plot(X[-mid:], (hist/hist.sum()+sep*j)[-mid:], '-', c='k', alpha=.5)

        mean = np.mean(enrich_data[Y[0]], axis=0)
        lo   = np.abs(mean - np.quantile(enrich_data[Y[0]], 0.025, axis=0))
        hi   = np.abs(mean - np.quantile(enrich_data[Y[0]], 0.975, axis=0))

        ax[2].barh([sep*j+(i+.7)*sep/3 for j in range(10)], mean, sep/3, xerr=(lo, hi), color=col[i], ec='k', alpha=.5, label=lbls[i], error_kw={'lw':.8})
        ax[2].plot([0,0], [-0.05, 0.5], '-', c='k', lw=.1)
    ax[0].set_yticks(np.arange(len(quantiles))*sep)
    ax[0].set_yticklabels([round(x,1) for x in quantiles])
    ax[2].legend(loc='upper center', ncol=2, columnspacing=1.5, frameon=False,
                 bbox_to_anchor=(0.52, 1.15))



    for i, t in zip([0,1], ttls):
        ax[i].set_title(t)
        ax[i].set_xlim(-.15, .15)
        ax[i].set_xticks([-.1, 0, .1])

    for i in range(3):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_ylim(0-sep/4, 0.5+sep)
    for i in [1,2]:
        ax[i].spines['left'].set_visible(False)
        ax[i].set_yticks([])

    ax[0].set_xlabel(r"asym$_{\alpha}$")
    ax[1].set_xlabel(r"asym$_{\beta}$")
    ax[0].set_ylabel(r'$\log_{10}R$')
    ax[2].set_xlabel('N terminal\nEnrichment')


    pdb = pdb.loc[pdb.OC!='Viruses']
    X = np.arange(10)
    X = np.array([sep*j+(i+.7)*sep/3 for j in range(10)])
    width = .175
    ttls = ['Eukaryote ', 'Prokaryote ']
    lbls = [r'$E_{\alpha}$', r'$E_{\beta}$']
    custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
    col = [custom_cmap[i] for i in [3, 9, 0, 12]]
    paths = [f"fig3_enrich_{a}_acpro.pickle" for a in ['eukaryote', 'prokaryote']]

    for i, path in enumerate(paths):
        enrich_data = pickle.load(open(PATH_FIG_DATA.joinpath(path), 'rb'))
        for j, Y in enumerate(['H_ASYM', 'S_ASYM']):
#           adjust = (j - 1 + i*2)*width
            adjust = (j*2 - 4.0 + i)*(sep/5)
            mean = np.mean(enrich_data[Y[0]], axis=0)
            lo   = np.abs(mean - np.quantile(enrich_data[Y[0]], 0.025, axis=0))
            hi   = np.abs(mean - np.quantile(enrich_data[Y[0]], 0.975, axis=0))
            print(i, Y, max(np.abs(mean)))
            ax[3].barh(X+adjust, mean, sep/5.0, ec='k', xerr=(lo, hi), color=col[i*2+j],
                       label=ttls[i]+lbls[j], lw=0.001, error_kw={'lw':.2})
    ax[3].plot([0,0], [-0.05, 0.5], '-', c='k', lw=.1)
    ax[3].set_yticks(np.arange(len(quantiles))*sep)
    ax[3].set_ylabel(r'$\log_{10} R$')
    ax[3].set_yticklabels([round(x,1) for x in quantiles])

    ax[3].set_xlabel('N terminal\nEnrichment')
    ax[3].set_xlim(-.42, .42)
    ax[3].set_ylim(0-sep/4, 0.5+sep)

    ax[3].spines['top'].set_visible(False)
    ax[3].spines['left'].set_visible(False)
    handles = [mpatches.Patch([], [], color=col[j*2+i], label=ttls[j]+lbls[i]) for i in [0,1] for j in [1,0]]
    ax[3].legend(handles=handles, bbox_to_anchor=(1.05, 1.25), frameon=False,
                 loc='upper right', ncol=2, columnspacing=1.0, handlelength=1.5)
    ax[3].yaxis.set_label_position('right')
    ax[3].yaxis.tick_right()




####################################################################
### FIG 11


def fig11(pdb, X='AA_PDB', Y='CO', w=.1, ax='', fig=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots(4,2, figsize=(9,12))
        fig.subplots_adjust(wspace=0.0, hspace=0.65)
#       ax = ax.reshape(ax.size)

    pdb_CO = np.load(PATH_FIG_DATA.joinpath("pdb_config_CO.npy"))[:,:,0]
    
    df = pdb.copy()
    q = np.arange(w,1+w,w)
    lbls = ['Helix', 'Sheet']
#   cb_lbl = [r"$E_{\alpha}$", r"$E_{\beta}$"]
    cb_lbl = [r"$asym_{\alpha}$", r"$asym_{\beta}$"]
    vmax = 0.03
    vmin = -vmax

    for j, co in enumerate(pdb_CO.T):
        df['CO'] = co
        quant1 = [df[X].min()] + list(df[X].quantile(q).values)
        quant2 = [df[Y].min()] + list(df[Y].quantile(q).values)
        for i, Z in enumerate(['H_ASYM', 'S_ASYM']):
            mean = []
            for l1, h1 in zip(quant1[:-1], quant1[1:]):
                for l2, h2 in zip(quant2[:-1], quant2[1:]):
                    samp = df.loc[(df[X]>=l1)&(df[X]<h1)&(df[Y]>=l2)&(df[Y]<h2), Z]
                    mean.append(samp.mean())
#                   left  = len(df.loc[(df[X]>=l1)&(df[X]<h1)&(df[Y]>=l2)&(df[Y]<h2)&(df[Z]<0)])
#                   right = len(df.loc[(df[X]>=l1)&(df[X]<h1)&(df[Y]>=l2)&(df[Y]<h2)&(df[Z]>0)])
#                   tot = max(len(df.loc[(df[X]>=l1)&(df[X]<h1)&(df[Y]>=l2)&(df[Y]<h2)]), 1)
#                   mean.append((right - left)/tot)

            cmap = sns.diverging_palette(230, 22, s=100, l=47, as_cmap=True)
            norm = colors.BoundaryNorm([vmin, vmax], cmap.N)
            bounds = np.linspace(vmin, vmax, 3)
            im = ax[j,i].imshow(np.array(mean).reshape(q.size, q.size).T, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, cmap=cmap, ticks=bounds, ax=ax[j,i], fraction=0.046, pad=0.04)

            cbar.set_label(cb_lbl[i], labelpad=-5)
            ax[j,i].set_title(lbls[i])
            ax[j,i].set_xticks(np.arange(q.size+1)-0.5)
            ax[j,i].set_yticks(np.arange(q.size+1)-0.5)
            ax[j,i].set_xticklabels([int(x) for x in quant1], rotation=90)
            ax[j,i].set_yticklabels([int(round(x,0)) for x in quant2])

    for a in ax.ravel():
        a.invert_yaxis()
        a.set_xlabel('Sequence Length')
        a.set_ylabel('Contact Order')
        a.tick_params(axis='both', which='major', direction='in')

    fs = 14
    for i, b in zip(range(4), list('ABCDEFGHI')):
        ax[i,0].text( -0.20, 1.16, b, transform=ax[i,0].transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath("si11.pdf"), bbox_inches='tight')



def fig12(pdb, X='REL_RATE', Y='S_ASYM', w=0.1):
    fig = plt.figure(figsize=(8,12))
    gs  = GridSpec(3,2, wspace=0.4, hspace=0.5, width_ratios=[1,1])
    ax_all = [[fig.add_subplot(gs[j,i]) for i in [0,1]] for j in range(3)]

    custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
    c_helix = custom_cmap[2]
    c_sheet = custom_cmap[10]
    col = [c_helix, c_sheet]

    bins = np.linspace(-0.20, 0.20, 80)
    width = np.diff(bins[:2])
    mid = 39
    sep = 0.05
    lbls = ['Sheet', 'Helix']
    quantiles = pdb[X].quantile(np.arange(0,1+w,w)).values
#   print(np.round(quantiles, 2))
    pdb['quant'] = pdb[X].apply(lambda x: utils.assign_quantile(x, quantiles))
#   pdb['quant'] = np.random.choice(pdb['quant'], len(pdb), replace=False)
    for ax, threshold in zip(ax_all, [0, 0.025, 0.05]):
        print(f"threshold = {threshold}")
        for i, Y in enumerate(['S_ASYM', 'H_ASYM']):
            ratio1 = []
            ratio2 = []
            lefts = []
            rights = []
            for j in range(len(quantiles)-1):
                hist, bins = np.histogram(pdb.loc[pdb.quant==j, Y], bins=bins)
                hist = hist / hist.sum()
                left  = len(pdb.loc[(pdb.quant==j)&(pdb[Y]<-threshold)]) / max(len(pdb.loc[(pdb.quant==j)]), 1)
                right = len(pdb.loc[(pdb.quant==j)&(pdb[Y]>threshold)]) / max(len(pdb.loc[(pdb.quant==j)]), 1)
                lefts.append(left)
                rights.append(right)
                ratio1.append((right - left))
                ratio2.append(np.log2(right / left))
                print(Y, j, left, right)
                xgrid = [sep*j+(i+1.0)*sep/3 for j in range(len(quantiles)-1)]
            ax[0].barh(xgrid, ratio1, sep/3, color=col[i], alpha=.5)
            ax[1].barh(xgrid, ratio2, sep/3, color=col[i], alpha=.5)
            ax[0].set_xticks(np.arange(-0.3, 0.4, 0.1))

        for a in ax:
            a.set_yticks(np.arange(len(quantiles))*sep)
            a.set_yticklabels([round(x,1) for x in quantiles])

            a.plot([0,0], [-0.05, 0.5], '-', c='k', lw=.1)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.set_ylim(0, 0.5)

            a.set_ylabel(r'$\log_{10}R$')

        ax[0].set_xlim(-0.35, 0.35)
        ax[1].set_xlim(-1.50, 1.50)

        ax[0].set_xlabel(r'$P(\mathregular{{asym}} \geq {0}) - P(\mathregular{{asym}} \leq -{0})$'.format(*[threshold]*2))
        ax[1].set_xlabel(r'$\log_{{2}} \frac{{P(\mathregular{{asym}} \geq {0})}}{{P(\mathregular{{asym}} \leq -{0})}} $'.format(*[threshold]*2))

    fig.savefig(PATH_FIG.joinpath("si12.pdf"), bbox_inches='tight')


def fig13(df, X='AA_PDB', Y='CO', w=.1, ax='', fig=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots(1,3, figsize=(15,4))
        fig.subplots_adjust(wspace=0.5)
    
    q = np.arange(w,1+w,w)
    quant1 = [df[X].min()] + list(df[X].quantile(q).values)
    quant2 = [df[Y].min()] + list(df[Y].quantile(q).values)
    lbls = ['Helix', 'Sheet']
    cb_lbl = [r"$asym_{\alpha}$", r"$asym_{\beta}$"]
    vmax = 0.03
    vmin = -vmax

    count = []

    for i, Z in enumerate(['H_ASYM', 'S_ASYM']):
        mean = []
        for l1, h1 in zip(quant1[:-1], quant1[1:]):
            for l2, h2 in zip(quant2[:-1], quant2[1:]):
                samp = df.loc[(df[X]>=l1)&(df[X]<h1)&(df[Y]>=l2)&(df[Y]<h2), Z]
                mean.append(samp.mean())
#               left  = len(df.loc[(df[X]>=l1)&(df[X]<h1)&(df[Y]>=l2)&(df[Y]<h2)&(df[Z]<0)])
#               right = len(df.loc[(df[X]>=l1)&(df[X]<h1)&(df[Y]>=l2)&(df[Y]<h2)&(df[Z]>0)])
#               tot = max(len(df.loc[(df[X]>=l1)&(df[X]<h1)&(df[Y]>=l2)&(df[Y]<h2)]), 1)
#               mean.append((right - left)/tot)
                if not i:
                    count.append(len(samp))
#                   print(len(samp))
        mean = np.array(mean).reshape(q.size, q.size)
        count = np.array(count).reshape(q.size, q.size)

        cmap = sns.diverging_palette(230, 22, s=100, l=47, as_cmap=True)
        norm = colors.BoundaryNorm([vmin, vmax], cmap.N)
        bounds = np.linspace(vmin, vmax, 3)
        im = ax[i].imshow(mean.T, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, cmap=cmap, ticks=bounds, ax=ax[i], fraction=0.046, pad=0.04)

        cbar.set_label(cb_lbl[i], labelpad=-5)
        ax[i].set_title(lbls[i])
        ax[i].set_xticks(np.arange(q.size+1)-0.5)
        ax[i].set_yticks(np.arange(q.size+1)-0.5)
        ax[i].set_xticklabels([int(x) for x in quant1], rotation=90)
        ax[i].set_yticklabels([int(round(x,0)) for x in quant2])

    for i in [2]:
        cmap = plt.cm.Greys
#       norm = colors.BoundaryNorm([-.04, .04], cmap.N)
#       bounds = np.linspace(-.04, .04, 5)
        im = ax[i].imshow(np.array(count).reshape(q.size, q.size).T, cmap=cmap, vmin=0)
        cbar = fig.colorbar(im, cmap=cmap, ax=ax[i], fraction=0.046, pad=0.04)

        cbar.set_label('Count')
        ax[i].set_title('Distribution')
        ax[i].set_xticks(np.arange(q.size+1)-0.5)
        ax[i].set_yticks(np.arange(q.size+1)-0.5)
        ax[i].set_xticklabels([int(x) for x in quant1], rotation=90)
        ax[i].set_yticklabels([int(round(x,0)) for x in quant2])

    for a in ax:
        a.invert_yaxis()
        a.set_xlabel('Sequence Length')
        a.set_ylabel('Contact Order')
        a.tick_params(axis='both', which='major', direction='in')

    fs = 14
    for i, b in zip([0,1,2], list('ABCDEFGHI')):
        ax[i].text( -0.20, 1.05, b, transform=ax[i].transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath("si13.pdf"), bbox_inches='tight')


def scop_ss():
    fig, ax = plt.subplots(2,1)
    cat = 'HS.D'
    N = 50
    X = np.arange(50)
    Nboot, Cboot, asym, enrich_edges, enrich_vals = pickle.load(open(PATH_FIG_DATA.joinpath(f"pdb_scop_indep.pickle"), 'rb'))
    data = [Nboot, Cboot, asym]
    custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
    c_helix = custom_cmap[2]
    c_sheet = custom_cmap[10]
    col = [c_helix, c_sheet, "#CB7CE6", "#79C726"]
    lbls = ['Helix', 'Sheet', 'Coil', 'Disorder']

    for j, s in enumerate(cat):
        ax[0].plot(X, data[0][s]['mean']/4, '-', c=col[j], label=f"{s} N")
        ax[0].fill_between(X, data[0][s]['hi']/4, data[0][s]['lo']/4, color="grey", label=f"{s} N", alpha=0.5)

        ax[0].plot(X, data[1][s]['mean']/4, '--', c=col[j], label=f"{s} N")
        ax[0].fill_between(X, data[1][s]['hi']/4, data[1][s]['lo']/4, color="grey", label=f"{s} N", alpha=0.2)

        print(s, round(np.mean(data[2][s]['mean']), 2), round(np.mean(data[2][s]['mean'][:20]), 2), round(np.mean(data[2][s]['mean'][20:]), 2))

        ax[1].plot(X, np.log2(data[2][s]['mean']), '-', c=col[j], label=lbls[j])
        ax[1].fill_between(X, np.log2(data[2][s]['hi']), np.log2(data[2][s]['lo']), color="grey", label=f"{s} N", alpha=0.2)

    ax[1].set_ylim(-1, 1.3)
    ax[1].plot([0]*50, '-', c='k')
    ax[1].set_yticks(np.arange(-1,1.5,0.5))
    ax[0].set_ylim(0, 0.6)

    ax[1].set_xlabel('Sequence distance from ends')
    ax[0].set_ylabel('Secondary structure\nprobability')
    ax[1].set_ylabel('Structural asymmetry\n$\\log_2 (N / C)$')


    fs = 14
    for i, b in zip([0,1], list('ABCDEFGHI')):
        ax[i].text( -0.10, 1.05, b, transform=ax[i].transAxes, fontsize=fs)
    fig.savefig(PATH_FIG.joinpath("si14.pdf"), bbox_inches='tight')


def percentage_asym(x):
    return np.sign(x) * 100*2**(abs(x)) - np.sign(x) * 100


def fig15():
    fig, ax = plt.subplots(3,1, figsize=(10,10))
    cat = 'HS.D'
    N = 100
    X = np.arange(N)
    Nboot, Cboot, asym, = pickle.load(open(PATH_FIG_DATA.joinpath(f"pdb_ss_max_asym.pickle"), 'rb'))
    data = [Nboot, Cboot, asym]
    custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
    c_helix = custom_cmap[2]
    c_sheet = custom_cmap[10]
    col = [c_helix, c_sheet, "#CB7CE6", "#79C726"]
    lbls = ['Helix', 'Sheet', 'Coil', 'Disorder']
    X2 = np.arange(5)

    for j, s in enumerate(cat):
        ax[0].plot(X, data[0][s]['mean']/2, '-', c=col[j], label=f"{s} N")
        ax[0].fill_between(X, data[0][s]['hi']/2, data[0][s]['lo']/2, color="grey", label=f"{s} N", alpha=0.5)

        ax[0].plot(X, data[1][s]['mean']/2, '--', c=col[j], label=f"{s} N")
        ax[0].fill_between(X, data[1][s]['hi']/2, data[1][s]['lo']/2, color="grey", label=f"{s} N", alpha=0.2)

        for k in range(5):
            print(s, round(np.mean(data[2][s]['mean']), 2), round(np.mean(data[2][s]['mean'][k*20:(k+1)*20]), 2))

        ax[1].plot(X, np.log2(data[2][s]['mean']), '-', c=col[j], label=lbls[j])
        ax[1].fill_between(X, np.log2(data[2][s]['hi']), np.log2(data[2][s]['lo']), color="grey", label=f"{s} N", alpha=0.2)

        if s in 'HS':
            Y2 = [percentage_asym(np.log2(data[2][s]['mean'])[k*20:(k+1)*20].mean()) for k in range(5)]
            ax[2].bar(X2, Y2, 0.5, color=col[j], label=lbls[j], ec='k')

    ax[1].set_ylim(-1.5, 2.0)
    ax[1].plot([0]*100, '-', c='k')
    ax[2].plot([0]*5, '-', c='k')
    ax[1].set_yticks(np.arange(-1,2.5,0.5))
    ax[0].set_ylim(0, 0.6)
    ax[2].set_xticks(np.arange(5))
    ax[2].set_xticklabels([f"{i*20} - {(i+1)*20}" for i in range(5)])

    ax[0].set_xlabel('Sequence distance from ends')
    ax[1].set_xlabel('Sequence distance from ends')
    ax[2].set_xlabel('Sequence distance from ends')
    ax[0].set_ylabel('Secondary structure\nprobability')
    ax[1].set_ylabel('Structural asymmetry\n$\\log_2 (N / C)$')
    ax[2].set_ylabel('Percentage asymmetry')


    fs = 14
    for i, b in zip([0,1,2], list('ABCDEFGHI')):
        ax[i].text( -0.10, 1.05, b, transform=ax[i].transAxes, fontsize=fs)
    fig.savefig(PATH_FIG.joinpath("si15.pdf"), bbox_inches='tight')


def oligomer(pdb, X='REL_RATE', Y='S_ASYM', w=0.1):
    pdb = pdb.copy()
    fig = plt.figure(figsize=(8,8))
    gs  = GridSpec(2,2, wspace=0.4, hspace=0.5, width_ratios=[1,1])
    ax_all = [[fig.add_subplot(gs[j,i]) for i in [0,1]] for j in range(2)]

    custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
    c_helix = custom_cmap[2]
    c_sheet = custom_cmap[10]
    col = [c_helix, c_sheet]
    bins = np.linspace(-0.20, 0.20, 80)
    width = np.diff(bins[:2])
    mid = 39
    sep = 0.05
    threshold = 0
    lbls = [r'$E_{\beta}$', r'$E_{\alpha}$']
    ttls = ['Monomers', 'Oligomers']
    for ax, idx, ttl in zip(ax_all, [pdb.NPROT==1, pdb.NPROT>1], ttls):
        quantiles = pdb.loc[idx, X].quantile(np.arange(0,1+w,w)).values
        pdb['quant'] = pdb.loc[idx, X].apply(lambda x: utils.assign_quantile(x, quantiles))
        for i, Y in enumerate(['S_ASYM', 'H_ASYM']):
            ratio1 = []
            ratio2 = []
            lefts = []
            rights = []
            for j in range(len(quantiles)-1):
                hist, bins = np.histogram(pdb.loc[(idx)&(pdb.quant==j), Y], bins=bins)
                hist = hist / hist.sum()
                left  = len(pdb.loc[(idx)&(pdb.quant==j)&(pdb[Y]<-threshold)]) / max(len(pdb.loc[(idx)&(pdb.quant==j)]), 1)
                right = len(pdb.loc[(idx)&(pdb.quant==j)&(pdb[Y]>threshold)]) / max(len(pdb.loc[(idx)&(pdb.quant==j)]), 1)
                lefts.append(left)
                rights.append(right)
                ratio1.append((right - left))
                ratio2.append(np.log2(right / left))
                xgrid = [sep*j+(i+1.0)*sep/3 for j in range(len(quantiles)-1)]
            ax[0].barh(xgrid, ratio1, sep/3, color=col[i], alpha=.5, label=lbls[i])
            ax[1].barh(xgrid, ratio2, sep/3, color=col[i], alpha=.5)
            ax[0].set_xticks(np.arange(-0.3, 0.4, 0.1))

        for a in ax:
            a.set_yticks(np.arange(len(quantiles))*sep)
            a.set_yticklabels([round(x,1) for x in quantiles])

            a.plot([0,0], [-0.05, 0.5], '-', c='k', lw=.1)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.set_ylim(0, 0.5)

            a.set_ylabel(r'$\log_{10}R$')
            a.set_title(f"{ttl}, N={np.sum(idx)}")

        ax[0].set_xlim(-0.35, 0.35)
        ax[1].set_xlim(-1.50, 1.50)

        ax[0].set_xlabel(r'$P(\mathregular{{asym}} \geq {0}) - P(\mathregular{{asym}} \leq -{0})$'.format(*[threshold]*2))
        ax[1].set_xlabel(r'$\log_{{2}} \frac{{P(\mathregular{{asym}} \geq {0})}}{{P(\mathregular{{asym}} \leq -{0})}} $'.format(*[threshold]*2))

        ax[0].legend(loc='upper center', ncol=2, columnspacing=3, frameon=False,
                     bbox_to_anchor=(1.20, 1.20))

    fig.savefig(PATH_FIG.joinpath("si16.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("oligomers.png"), bbox_inches='tight')


def scop2(X='REL_RATE', Y='S_ASYM', w=0.1):
    fig, ax = plt.subplots(figsize=(10,6))
    edges, data = pickle.load(open(PATH_FIG_DATA.joinpath("pdb_scop_indep.pickle"), 'rb'))[3:]
    edges = edges[0]
    sep = 0.05
    lbls = [r'$E_{\alpha}$', r'$E_{\beta}$']

    for i, Y in enumerate(['H_ASYM', 'S_ASYM']):
        mean = np.mean(data[:,i], axis=0)
        lo   = np.abs(mean - np.quantile(data[:,i], 0.025, axis=0))
        hi   = np.abs(mean - np.quantile(data[:,i], 0.975, axis=0))
        ax.barh([sep*j+(i+.7)*sep/3 for j in range(10)], mean, sep/3, xerr=(lo, hi), color=col[i], ec='k', alpha=.5, label=lbls[i], error_kw={'lw':.8})
        ax.plot([0,0], [-0.05, 0.5], '-', c='k', lw=.1)

    ax.set_yticks(np.arange(len(edges))*sep)
    ax.set_yticklabels([round(x,1) for x in edges])
    ax.legend(loc='upper center', ncol=2, columnspacing=3, frameon=False,
                 bbox_to_anchor=(0.52, 1.06))

    ax.set_xlim(-.38, .38)
    ax.set_xticks(np.arange(-.3, .4, .1))


# To create this figure, you need to download the complete
# Human and E. coli proteomes at:
# https://alphafold.ebi.ac.uk/download
# and then change the code so that "base" points to the
# folder that contains the downloaded ".pdb" files
def disorder_proteome(N=100):
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    lbls = ["Human", "Ecoli"]
    ttls = ["Human", "E. coli"]
    for i, l in enumerate(lbls):
        path = PATH_FIG_DATA.joinpath(f"alphafold_{l}.npy")
        if not path.exists():
            base = PATH_BASE.joinpath(f"AlphaFold/{l}")
            countN = np.zeros(N, float)
            countC = np.zeros(N, float)
            tot = np.zeros(N, float)
            with Pool(50) as pool:
                dis = list(pool.imap_unordered(utils.get_disorder_from_conf, base.glob("*pdb"), 10))
                for d in dis:
                    n = min(int(len(d)/2), N)
                    countN[:n] = countN[:n] + d[:n]
                    countC[:n] = countC[:n] + d[-n:][::-1]
                    tot[:n] = tot[:n] + 1
            fracN = countN / tot
            fracC = countC / tot
            np.save(path, np.array([fracN, fracC]))
        else:
            fracN, fracC = np.load(path)

        ax[i].plot(np.arange(N)+1, fracN, '-', label='N')
        ax[i].plot(np.arange(N)+1, fracC, '--', label='C')
        ax[i].set_title(ttls[i])
        ax[i].set_xlabel("Sequence distance from ends")
        ax[i].set_ylabel("Disorder probability")
        ax[i].set_ylim(0, 1)
        ax[i].legend(loc='best', frameon=False)
 
    fig.savefig(PATH_FIG.joinpath("si17.pdf"), bbox_inches='tight')


def kfold_vs_ss():
    pfdb = asym_io.load_pfdb()
    fig, ax = plt.subplots(figsize=(8,8))
    for c in pfdb.Class.unique():
        X = np.log10(pfdb.loc[pfdb.Class==c, 'L'])
        Y = pfdb.loc[pfdb.Class==c, 'log_kf']
        sns.regplot(X, Y, label=c)
    ax.set_xlabel(r"$\log_{10}$ Sequence Length")
    ax.set_ylabel(r"$\log_{10} k_f$")
    ax.legend(loc='best', frameon=False)

    fig.savefig(PATH_FIG.joinpath("si18.pdf"), bbox_inches='tight')


def hbond_asym(pdb, Xl='REL_RATE', Y='hb_asym', w=0.1):
    fig = plt.figure(figsize=(9,6))
    gs  = GridSpec(1,2, wspace=0.2, hspace=0.0, width_ratios=[1,.3])
    ax = [fig.add_subplot(gs[i]) for i in [0,1]]

    col = np.array(Paired_12.hex_colors)[[1,3]]
    bins = np.linspace(-0.20, 0.20, 80)
    width = np.diff(bins[:2])
    X = bins[:-1] + width * 0.5
    mid = 39
    sep = 0.05
    quantiles = pdb[Xl].quantile(np.arange(0,1+w,w)).values
    ratio = []
    lefts = []
    rights = []
    threshold = 0.00
    for j in range(len(quantiles)-1):
        hist, bins = np.histogram(pdb.loc[pdb.quant==j, Y], bins=bins)
        hist = hist / hist.sum()
        left  = len(pdb.loc[(pdb.quant==j)&(pdb[Y]<-threshold)]) / max(len(pdb.loc[(pdb.quant==j)]), 1)
        right = len(pdb.loc[(pdb.quant==j)&(pdb[Y]>threshold)]) / max(len(pdb.loc[(pdb.quant==j)]), 1)
        lefts.append(left)
        rights.append(right)
        ratio.append((right - left))
        ax[0].bar(X[:mid], (hist/hist.sum())[:mid], width, bottom=[sep*j]*mid, color='grey', alpha=.5)
        ax[0].bar(X[-mid:], (hist/hist.sum())[-mid:], width, bottom=[sep*j]*mid, color=col[0], alpha=.5)
        ax[0].plot(X[:mid], (hist/hist.sum()+sep*j)[:mid], '-', c='k', alpha=.5)
        ax[0].plot(X[-mid:], (hist/hist.sum()+sep*j)[-mid:], '-', c='k', alpha=.5)

    ax[0].set_yticks(np.arange(len(quantiles))*sep)
    ax[0].set_yticklabels([round(x,1) for x in quantiles])

    ax[1].barh([sep*j+sep/2 for j in range(len(quantiles)-1)], ratio, sep/2, color=[col[0] if r > 0 else 'grey' for r in ratio], alpha=.5)
    ax[1].plot([0,0], [-0.05, 0.5], '-', c='k', lw=.1)

    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].set_yticks([])
    for a in ax:
        a.set_ylim(0, 0.60)

    ax[0].set_xlabel('Asymmetry in mean hydrogen bond length')
    ax[0].set_ylabel(r'$\log_{10}R$')
    ax[1].set_xlabel('N terminal enrichment')

    fig.savefig(PATH_FIG.joinpath("si19.pdf"), bbox_inches='tight')


def hyd_asym(pdb, Xl='REL_RATE', Y='hyd_asym', w=0.1):
    fig = plt.figure(figsize=(9,6))
    gs  = GridSpec(1,2, wspace=0.2, hspace=0.0, width_ratios=[1,.3])
    ax = [fig.add_subplot(gs[i]) for i in [0,1]]

    col = np.array(Paired_12.hex_colors)[[1,3]]
    bins = np.linspace(-4.5, 4.5, 80)
    width = np.diff(bins[:2])
    X = bins[:-1] + width * 0.5
    mid = 39
    sep = 0.05
    quantiles = pdb[Xl].quantile(np.arange(0,1+w,w)).values
    ratio = []
    lefts = []
    rights = []
    threshold = 0.00
    for j in range(len(quantiles)-1):
        hist, bins = np.histogram(pdb.loc[pdb.quant==j, Y], bins=bins)
        hist = hist / hist.sum()
        left  = len(pdb.loc[(pdb.quant==j)&(pdb[Y]<-threshold)]) / max(len(pdb.loc[(pdb.quant==j)]), 1)
        right = len(pdb.loc[(pdb.quant==j)&(pdb[Y]>threshold)]) / max(len(pdb.loc[(pdb.quant==j)]), 1)
        lefts.append(left)
        rights.append(right)
        ratio.append((right - left))
        ax[0].bar(X[:mid], (hist/hist.sum())[:mid], width, bottom=[sep*j]*mid, color='grey', alpha=.5)
        ax[0].bar(X[-mid:], (hist/hist.sum())[-mid:], width, bottom=[sep*j]*mid, color=col[0], alpha=.5)
        ax[0].plot(X[:mid], (hist/hist.sum()+sep*j)[:mid], '-', c='k', alpha=.5)
        ax[0].plot(X[-mid:], (hist/hist.sum()+sep*j)[-mid:], '-', c='k', alpha=.5)

    ax[0].set_yticks(np.arange(len(quantiles))*sep)
    ax[0].set_yticklabels([round(x,1) for x in quantiles])

    ax[1].barh([sep*j+sep/2 for j in range(len(quantiles)-1)], ratio, sep/2, color=[col[0] if r > 0 else 'grey' for r in ratio], alpha=.5)
    ax[1].plot([0,0], [-0.05, 0.5], '-', c='k', lw=.1)

    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].set_yticks([])
    for a in ax:
        a.set_ylim(0, 0.60)

    ax[0].set_xlabel('Asymmetry in mean hydrophobicity')
    ax[0].set_ylabel(r'$\log_{10}R$')
    ax[1].set_xlabel('N terminal enrichment')

    fig.savefig(PATH_FIG.joinpath("si20.pdf"), bbox_inches='tight')


