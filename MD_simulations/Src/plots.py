from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.nonparametric.api as smnp 

import anal_probe as AP

PATH_BASE = Path("/home/jmcbride/CotransFold/")
#ATH_BASE = Path("/home/jmcbride/CotransFold/3BID")
PATH_DATA = Path("/home/jmcbride/CotransFold/Test/Init")
PATH_RES  = PATH_BASE.joinpath('Results')



def plot_fold_time_dist(fold_time):
    fig, ax = plt.subplots(2,1)
    bins = np.arange(0, 1000, 20)
    temp = np.arange(80, 112, 2)
    X = bins[:-1] + np.diff(bins[:2])/2.
    for t, ft in zip(temp, fold_time):
        hist = np.histogram([x for x in ft if np.isfinite(x)], bins=bins)[0]
        print('  '.join([f"{t:10.4f}{fn([x for x in ft if np.isfinite(x)]):10.4f}" for fn in [np.mean, np.std]]))
        ax[0].plot(X, hist/hist.sum(), label=t)
    ax[0].legend(loc='best', frameon=False)
    mean = [np.mean(x[np.isfinite(x)]) for x in fold_time]
    std  = [np.std(x[np.isfinite(x)]) for x in fold_time]
    ax[1].errorbar(temp, mean, std)


def get_maximum(X, ngrid=1000):
    if not len(X):
        return np.nan
    kde = smnp.KDEUnivariate(np.array(X, dtype=float))
    kde.fit(kernel='gau', bw='scott', fft=1, gridsize=10000, cut=20)
    grid = np.linspace(min(X), max(X), num=ngrid)
    y = np.array([kde.evaluate(x) for x in grid]).reshape(ngrid)
    return grid[np.argmax(y)]


def plot_forbak_dist(steps, forbak, x=5, y=5):
#   fig, ax = plt.subplots(x,y, sharex=True)
    fig, ax = plt.subplots(x,y)
    ax = ax.reshape(ax.size)
    for i in range(forbak.shape[1]):
        sns.distplot([x for x in forbak[0,i,:] if np.isfinite(x)], ax=ax[i])
        sns.distplot([x for x in forbak[1,i,:] if np.isfinite(x)], ax=ax[i])
        ax[i].set_title(steps[i])


def plot_forbak_ratio(steps, forbak, alg='mean', ax='', fold_time=324500, seqlen=131, pt='o', lbl=''):
    ratio = []
    for i in range(forbak.shape[1]):
        if alg=='mean':
            forw = np.mean([x for x in forbak[0,i,:] if np.isfinite(x)])
            back = np.mean([x for x in forbak[1,i,:] if np.isfinite(x)])
        elif alg=='max':
            forw = get_maximum([x for x in forbak[0,i,:] if np.isfinite(x)])
            back = get_maximum([x for x in forbak[1,i,:] if np.isfinite(x)])
#       print(steps[i], len([x for x in forbak[0,i,:] if np.isfinite(x)]), len([x for x in forbak[1,i,:] if np.isfinite(x)]))
#       print(i, forw, back)
        ratio.append(back/forw) 
#       ratio.append(forw-back) 
    if isinstance(ax, str):
        fig, ax = plt.subplots()
    ax.plot(np.log10(fold_time / (np.array(steps)*seqlen)), ratio, pt, label=lbl, alpha=0.7)
    return np.array([np.log10(fold_time / (np.array(steps)*seqlen)), np.array(ratio)])


def plot_Qtraj(end, st, k, path_idx=3):
    path = PATH_BASE.joinpath(f"{end}_first_{path_idx:02d}")
    path_ext = path.joinpath(f"trans_time_{st:05d}", f"{k+1:04d}")
    Q = AP.load_and_unravel(path_ext, 58, 59)
    fig, ax = plt.subplots()
    ax.plot(Q)


def plot_ratio_all_prot():
    pdb_idx = [127234, 98113, 19921]
    pdb_id = ['3BID', '2OT2', '1ILO']
    ft = [98000, 167000, 189000]
    sl = [58, 90, 77]
    pt = ['o', 's', '^']

    fig, ax = plt.subplots()
    for i in range(len(pdb_id)):
        steps = [int(x.strip('\n')) for x in open(PATH_BASE.joinpath(pdb_id[i], 'N_first_01', 'steps.txt'))]
        ratio = np.load(PATH_BASE.joinpath(pdb_id[i], 'rates_01.npy'))
        data = plot_forbak_ratio(steps, ratio, ax=ax, fold_time=ft[i], seqlen=sl[i], pt=pt[i], lbl=pdb_id[i])
        np.save(PATH_RES.joinpath(f'{pdb_id[i]}.npy'), data)
    ax.legend(loc='best', frameon=False)
    X = np.arange(-2, 2.01, 0.01)
    Y  = (1+10**X) / np.array([max(1, 10**x) for x in X])
    ax.plot(X, Y, '-', c='k')

    tunnel = 30.
    L = 77.
    Y2 = (1+10**X) / np.array([max(1, 10**x+tunnel/L) for x in X])
    ax.plot(X, Y2, '-', c='grey')

    L = 100.
    Y3 = (1+10**X) / np.array([max(1, 10**x+tunnel/L) for x in X])
    ax.plot(X, Y3, '-', c='grey')

    ax.plot([-2, 2], [1,1], ':', c='k')

    ax.set_ylabel('Reverse folding time / Forward folding time')
    ax.set_xlabel('Folding time / translation time')





