from collections import defaultdict, Counter
from itertools import product, permutations
from glob import glob
import json
import os
from pathlib import Path
import pickle
import sqlite3
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
from scipy.stats import linregress, pearsonr
import seaborn as sns
import svgutils.compose as sc

from asym_io import PATH_BASE, PATH_ASYM
import asym_utils as utils

PATH_FIG = PATH_ASYM.joinpath("Figures")
PATH_FIG_DATA = PATH_FIG.joinpath("Data")


####################################################################
### FIG 1


def fig1(pdb):
    fig = plt.figure(figsize=(11, 10))
    gs  = GridSpec(5,48, wspace=1.0, hspace=0.0, height_ratios=[1,0.3,1,0.6,1.5])
    ax = [fig.add_subplot(gs[i*2,:14]) for i in [0,1]] + \
         [fig.add_subplot(gs[i*2,17:31]) for i in [0,1]] + \
         [fig.add_subplot(gs[i*2,34:]) for i in [0,1]] + \
         [fig.add_subplot(gs[3:,i:i+j]) for i,j in zip([0,18,36],[12,12,12])]

    custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
    c_helix = custom_cmap[2]
    c_sheet = custom_cmap[10]
    col = [c_helix, c_sheet, "#CB7CE6", "#79C726"]

    ttls = ['Full sample', 'Eukaryotes', 'Prokaryotes']
    lbls = ['Helix', 'Sheet', 'Coil', 'Disorder']
    cat = 'HS.D'
    ss_stats = [pickle.load(open(PATH_FIG_DATA.joinpath(f"pdb_ss_dist_boot{s}.pickle"), 'rb')) for s in ['', '_euk', '_prok']]
    X = np.arange(50)
    for i, data in enumerate(ss_stats):
        for j, s in enumerate(cat):
            ax[i*2].plot(X, data[0][s]['mean']/2, '-', c=col[j], label=f"{s} N")
            ax[i*2].fill_between(X, data[0][s]['hi']/2, data[0][s]['lo']/2, color="grey", label=f"{s} N", alpha=0.5)

            ax[i*2].plot(X, data[1][s]['mean']/2, '--', c=col[j], label=f"{s} N")
            ax[i*2].fill_between(X, data[1][s]['hi']/2, data[1][s]['lo']/2, color="grey", label=f"{s} N", alpha=0.2)

            print(i, s, round(np.mean(data[2][s]['mean']), 2), round(np.mean(data[2][s]['mean'][:20]), 2))

            ax[i*2+1].plot(X, np.log2(data[2][s]['mean']), '-', c=col[j], label=lbls[j])
            ax[i*2+1].fill_between(X, np.log2(data[2][s]['hi']), np.log2(data[2][s]['lo']), color="grey", label=f"{s} N", alpha=0.2)

    for i in range(3):
        ax[i*2].set_title(ttls[i])
        ax[i*2+1].set_ylim(-1, 1.3)
        ax[i*2+1].plot([0]*50, '-', c='k')
        ax[i*2+1].set_yticks(np.arange(-1,1.5,0.5))
        ax[i*2].set_ylim(0, 0.6)

#   ax[0].plot(X, (X+1)**-0.78, '-k')
#   ax[2].plot(X, (X+1)**-0.5, '-k')
#   ax[2].plot(X, (X+1)**-0.6, '-k')
#   ax[4].plot(X, (X+1)**-1.0, '-k')


    ax[1].set_ylabel('Structural asymmetry\n$\\log_2 (N / C)$')
    handles = [Line2D([0], [0], ls=ls, c=c, label=l) for ls, c, l in zip(['-', '--'], ['k']*2, ['N', 'C'])] + \
              [Line2D([0], [0], ls='-', c=c, label=l) for l, c in zip(lbls, col)]
    ax[4].legend(handles=handles, bbox_to_anchor=(0.20, 1.43), frameon=False,
                 ncol=6, columnspacing=1.5, handlelength=2.0)

    ax[0].set_ylabel('Secondary structure\nprobability')

    for i in range(6):
        ax[i].set_xticks(range(0, 60, 10))
        ax[i].set_xlabel('Sequence distance from ends')
    

    fig1b(pdb, ax=ax[6:], fig=fig)

    fs = 14
    for i, b in zip([0,1,6,7,8], list('ABCDEFGHI')):
        ax[i].text( -0.20, 1.10, b, transform=ax[i].transAxes, fontsize=fs)
    for a in ax:
        a.tick_params(axis='both', which='major', direction='in')

    fig.savefig(PATH_FIG.joinpath("fig1.pdf"), bbox_inches='tight')


def ss_by_seq(pdb, cat='SH.D'):
    countN, countC = utils.pdb_end_stats_disorder_N_C(pdb, N=50, s1='SEQ_PDB', s2='SS_PDB')
    base = np.zeros(len(countN['S']), dtype=float)
    Yt = np.array([[sum(p.values()) for p in countN[s]] for s in cat]).sum(axis=0)
    X = np.arange(base.size)
    out_dict = {}
    for i, s in enumerate(cat):
        YN = np.array([sum(p.values()) for p in countN[s]])
        YC = np.array([sum(p.values()) for p in countC[s]])
        out_dict[s] = {'N':YN/Yt, 'C':YC/Yt, 'asym':YN/YC}
    return out_dict


def fig1b(df, X='AA', Y='CO', w=.1, ax='', fig=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots(1,3, figsize=(15,4))
        fig.subplots_adjust(wspace=0.5)
    
    q = np.arange(w,1+w,w)
    quant1 = [df[X].min()] + list(df[X].quantile(q).values)
    quant2 = [df[Y].min()] + list(df[Y].quantile(q).values)
    lbls = ['Helix', 'Sheet']
    cb_lbl = [r"$E_{\alpha}$", r"$E_{\beta}$"]
    vmax = 0.50
    vmin = -vmax

    count = []

    for i, Z in enumerate(['H_ASYM', 'S_ASYM']):
        mean = []
        for l1, h1 in zip(quant1[:-1], quant1[1:]):
            for l2, h2 in zip(quant2[:-1], quant2[1:]):
                samp = df.loc[(df[X]>=l1)&(df[X]<h1)&(df[Y]>=l2)&(df[Y]<h2), Z]
#               mean.append(samp.mean())
                left  = len(df.loc[(df[X]>=l1)&(df[X]<h1)&(df[Y]>=l2)&(df[Y]<h2)&(df[Z]<0)])
                right = len(df.loc[(df[X]>=l1)&(df[X]<h1)&(df[Y]>=l2)&(df[Y]<h2)&(df[Z]>0)])
                tot = max(len(df.loc[(df[X]>=l1)&(df[X]<h1)&(df[Y]>=l2)&(df[Y]<h2)]), 1)
                mean.append((right - left)/tot)
                if not i:
                    count.append(len(samp))
#                   print(len(samp))

#       cmap = plt.cm.RdBu
#       cmap = sns.diverging_palette(230, 25, s=98, l=55, as_cmap=True)
        cmap = sns.diverging_palette(230, 22, s=100, l=47, as_cmap=True)
        norm = colors.BoundaryNorm([vmin, vmax], cmap.N)
        bounds = np.linspace(vmin, vmax, 3)
        im = ax[i].imshow(np.array(mean).reshape(q.size, q.size).T, cmap=cmap, vmin=vmin, vmax=vmax)
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


    


####################################################################
### FIG 2


def fig2():
    fs = 16
    
    if not PATH_FIG.joinpath("fig1_bottom.svg").exists() or 1:
#       fig = plt.figure()
#       gs  = GridSpec(2,2, wspace=0.5, hspace=0.5, height_ratios=[1,1])
#       ax = [fig.add_subplot(gs[0,i]) for i in [0,1]] + [fig.add_subplot(gs[1,:])]
        fig, ax = plt.subplots(2,1, figsize=(3,6))
        fig.subplots_adjust(hspace=0.50)
        fig2b(fig, ax[0])
        fig2d(fig, ax[1])

        for i, b in enumerate('BD'):
            ax[i].text( -0.28, 1.08, b, transform=ax[i].transAxes, fontsize=fs)
        fig.savefig(PATH_FIG.joinpath("fig1_bottom.svg"), bbox_inches='tight')
    

    sc.Figure("14.8cm", "7.50cm", 
        sc.Panel(sc.SVG(PATH_FIG.joinpath("fig1_dia_ver2.svg")).scale(1.400).move(5,0)),
        sc.Panel(sc.SVG(PATH_FIG.joinpath("fig1_bottom.svg")).scale(0.700).move(390,0))
        ).save(PATH_FIG.joinpath("fig1.svg"))



def fig2b(fig, ax):
    Rdist_data = pickle.load(open(PATH_FIG_DATA.joinpath("R_dist.pickle"), 'rb'))
    custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
    c_helix = custom_cmap[2]
    c_sheet = custom_cmap[10]
    col = [c_helix, c_sheet, "#CB7CE6", "#79C726"]
    lbls = ['All', 'Prokaryotes', 'Eukaryotes']

    for i, k in enumerate(['All', 'Prok', 'Euk']):
        ax.plot(Rdist_data['grid'], Rdist_data[k][0], '-', c=col[i], label=lbls[i])
        ax.fill_between(Rdist_data['grid'], Rdist_data[k][1], Rdist_data[k][2], color=col[i], alpha=0.5)
    ax.plot([0,0], [0, 0.35], ':', c='k', alpha=0.7)
    ax.set_xlabel(r'$\log_{10} R$')
    ax.set_ylabel('Density')
    ax.set_xticks(np.arange(-4, 5, 2))
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.35)
    ax.legend(loc='upper center', bbox_to_anchor=(0.55, 1.27), frameon=False, ncol=2, columnspacing=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def fig2d(fig, ax):
    X = np.arange(-3, 3, 0.01)
    Y = np.array([(10**x + 1)/max(10**x, 1) for x in X])
    Y2 = (1+10**X) / np.array([max(1, 10**x+0.3) for x in X]) 
    ax.plot(X, Y, '-', label=r"$\tau_{\mathregular{ribo}}=0$")
    ax.plot(X, Y2, ':', label=r"$\tau_{\mathregular{ribo}}=0.3\tau_{\mathregular{trans}}$")

    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(1, 2.05)
    ax.set_xlabel(r'$\log_{10} R$')
    ax.set_ylabel("Speed-up")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.15, 1.20), ncol=1, labelspacing=.1)




####################################################################
### FIG 3

def fig3(pdb, Y='S_ASYM'):
    fig = plt.figure(figsize=(12.8, 4.8))
    gs  = GridSpec(1,7, wspace=0.0, width_ratios=[5,0.2,5,0.4,3,1.0,6])
    ax = [fig.add_subplot(gs[0,i*2]) for i in range(4)]
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

    quantiles = pdb['REL_RATE'].quantile(np.arange(0,1.1,.1)).values
    pdb['quant'] = pdb['REL_RATE'].apply(lambda x: utils.assign_quantile(x, quantiles))
#   pdb['quant'] = np.random.choice(pdb['quant'], len(pdb), replace=False)

    enrich_data = pickle.load(open(PATH_FIG_DATA.joinpath("fig3_enrich.pickle"), 'rb'))
    for i, Y in enumerate(['H_ASYM', 'S_ASYM']):
        for j in range(len(quantiles)-1):
            hist, bins = np.histogram(pdb.loc[pdb.quant==j, Y], bins=bins)
            hist = hist / hist.sum()
            total = len(pdb)/10
            left  = len(pdb.loc[(pdb.quant==j)&(pdb[Y]<0)]) / total
            right = len(pdb.loc[(pdb.quant==j)&(pdb[Y]>0)]) / total
            print(Y, j, ''.join([f"{x:6.3f}" for x in [left, right, left/right, right / left]]))
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
    ax[2].legend(loc='upper center', ncol=2, columnspacing=3, frameon=False,
                 bbox_to_anchor=(0.52, 1.06))



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
    paths = [f"fig3_enrich_{a}.pickle" for a in ['eukaryote', 'prokaryote']]

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
    ax[3].set_ylim(0-sep/4, 0.5+sep/4)

    ax[3].spines['top'].set_visible(False)
    ax[3].spines['left'].set_visible(False)
    handles = [mpatches.Patch([], [], color=col[j*2+i], label=ttls[j]+lbls[i]) for i in [0,1] for j in [1,0]]
    ax[3].legend(handles=handles, bbox_to_anchor=(1.00, 1.16), frameon=False,
                 loc='upper right', ncol=2, columnspacing=1.3, handlelength=1.5)
    ax[3].yaxis.set_label_position('right')
    ax[3].yaxis.tick_right()

    fs = 14
    for i, b in zip([0,2,3], list('ABCDEFGHI')):
        ax[i].text( -0.20, 1.08, b, transform=ax[i].transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath("fig3.pdf"), bbox_inches='tight')




####################################################################
### FIG 4

def fig4(dom, Y='S_ASYM'):
    fig = plt.figure(figsize=(4.7,4.9))
    gs  = GridSpec(5,1, wspace=0.00, hspace=0.0,
                   height_ratios=[.45, .5, .45, .5, .8])# .8, .3, 1])
    ax = [fig.add_subplot(gs[i*2,0]) for i in range(2)] + \
         [fig.add_subplot(gs[4,0])] 
    ax = np.array(ax)
    ft = 14

    bins = np.arange(0,31,1.0)
    X = bins[:-1] + np.diff(bins[:2])
    yellows = sns.diverging_palette(5, 55, s=95, l=77, n=13)
    pinks = sns.diverging_palette(5, 55, s=70, l=52, n=13)
    col = [yellows[12], pinks[0]]
    col2 = [yellows[10], pinks[3]]

    custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
    col = [custom_cmap[0], custom_cmap[12]]
    X = np.arange(10)
    width = .35
    ttls = ['Single-domain', 'Multi-domain']
    lbls = [r'$E_{\alpha}$', r'$E_{\beta}$']
    paths = [f"fig4_enrich_{a}.pickle" for a in ['single', 'multi']]

    for i, dat in enumerate([dom.loc[(dom.NDOM==1)&(np.isfinite(dom.REL_RATE))], dom.loc[(dom.NDOM>1)&(np.isfinite(dom.REL_RATE))]]):
        enrich_data = pickle.load(open(PATH_FIG_DATA.joinpath(paths[i]), 'rb'))
        quantiles = dat['REL_RATE'].quantile(np.arange(0,1.1,.1)).values
#       dat['quant'] = dat['REL_RATE'].apply(lambda x: utils.assign_quantile(x, quantiles))
        for j, Y in enumerate(['H_ASYM', 'S_ASYM']):
#           ratio = []
#           for k in range(10):
#               total = len(dat)/10
#               left  = len(dat.loc[(dat.quant==k)&(dat[Y]<0)]) / total
#               right = len(dat.loc[(dat.quant==k)&(dat[Y]>0)]) / total
#               ratio.append(right - left)
#           mean = np.array([np.mean(dat.loc[dat.quant==j, Y]) for j in range(len(quantiles)-1)])
            adjust = (j-.5)*width
            mean = np.mean(enrich_data[Y[0]], axis=0)
            lo   = np.abs(mean - np.quantile(enrich_data[Y[0]], 0.025, axis=0))
            hi   = np.abs(mean - np.quantile(enrich_data[Y[0]], 0.975, axis=0))
            ax[i].bar(X+adjust, mean, width, ec='k', yerr=(lo, hi), color=col[j], label=lbls[j], alpha=.5, error_kw={'lw':.8})
        ax[i].set_xticks(np.arange(len(quantiles))-0.5)
        ax[i].set_yticks(np.arange(-.3, .4, .3))
        ax[i].set_ylabel('N terminal   \nEnrichment   ', labelpad=0, fontsize=ft-4)
        ax[i].set_xlabel(r'$\log_{10} R$', fontsize=ft-4)
        ax[i].set_xticklabels([round(x,1) for x in quantiles], rotation=90)
        ax[i].set_title(ttls[i], loc='left')
    for a in ax[:2]:
        a.set_ylim(-.33, .40)
    ax[1].legend(loc='upper right', bbox_to_anchor=(1.00,1.75), frameon=False)


    sns.boxplot(y='CO', x='NDOM', data=dom.loc[dom.NDOM<6], ax=ax[2], showfliers=False,
                color=(0.8, 0.8, 0.8))
    ax[2].set_xlabel("Number of domains", fontsize=ft-4)
    ax[2].set_ylabel("Contact Order", fontsize=ft-4)
    plt.setp(ax[2].artists, edgecolor=[.3]*3, facecolor=[.8]*3, lw=1.3)
    plt.setp(ax[2].lines, color=[.3]*3, lw=0.5)


    
    top = [1.00] + [1.20]*2
    bot = [-.19] + [-.20]*2
    for i, (a, b) in enumerate(zip(ax[[2,0,1]], list('DBCFGH'))):
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.text(bot[i], top[i], b, transform=a.transAxes, fontsize=ft+2)
        a.tick_params(axis='both', which='major', direction='in')

    fig.savefig(PATH_FIG.joinpath("fig3_bottom.svg"), bbox_inches='tight')
    

    sc.Figure("8.05cm", "10.65cm", 
        sc.Panel(sc.SVG(PATH_FIG.joinpath("fig3_bottom.svg")).scale(0.930).move(000,100)),
        sc.Panel(sc.SVG(PATH_FIG.joinpath("fig3_dia.svg")).scale(1.200).move(5,00))
        ).save(PATH_FIG.joinpath("fig3.svg"))



####################################################################
### PDB Domain plots


def circ_perm(df_pdb):
    fig, ax = plt.subplots(figsize=(6,2))
    SS = df_pdb.loc[1834, 'SS_PDB']
    ss_var = [utils.circ_perm(SS, 67), SS]
    lbls = ['1ASK-CP67', '1ASK']
    custom_cmap = sns.diverging_palette(230, 22, s=100, l=47, n=13)
    col_key = {'.':'grey', 'D':'grey', 'H':custom_cmap[3], 'S':custom_cmap[9]}
    ec_key = {'.':'grey', 'D':'grey', 'H':custom_cmap[1], 'S':custom_cmap[11]}
    wid_key = {'.':0.1, 'D':0.1, 'H':0.3, 'S':0.3}
    lw_key = {'.':0.7, 'D':0.7, 'H':1.5, 'S':1.5}
    for i, ss in enumerate(ss_var):
        left = 0.
        for j, strand in enumerate(generate_strand(ss)):
            s = strand[0]
            ax.barh([i], [len(strand)], wid_key[s], left=[left], color=col_key[s], ec=ec_key[s], linewidth=lw_key[s])
            left += len(strand) + 0.20
    for pos in ['left', 'right', 'top', 'bottom']:
        ax.spines[pos].set_visible(False)

    col = np.array(custom_cmap)[[3,9,1,11]]
    ax.legend(handles=[mpatches.Patch(fc=c1, ec=c2, label=l) for c1, c2, l in zip(col[:2], col[2:], ['Helix', 'Sheet'])],
                 loc='lower right', frameon=False, ncol=2, bbox_to_anchor=(0.95, 0.35))
    ax.set_xticks([])
    ax.set_yticks(range(2))
    ax.set_yticklabels(lbls)
    ax.tick_params(axis='y', which='major', length=0, pad=10)

    fig.savefig(PATH_FIG.joinpath("fig5.pdf"), bbox_inches='tight')


def generate_strand(ss):
    for i, s in enumerate(ss):
        if not i:
            strand = s
        elif s == strand[-1]:
            strand += s
        else:
            yield strand
            strand = s
    yield strand




