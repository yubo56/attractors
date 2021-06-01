import pickle, lzma
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
from utils import *

s_c_vals = {
    20: [
        0.7,
        0.2,
        0.06,
        2.0,
        1.2,
        1.0,
        0.65,
        0.6,
        0.55,
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.1,
        0.03,
        # 0.01,
    ],
    5: [
        0.7,
        0.2,
        0.06,
        2.0,
        1.2,
        1.0,
        0.85,
        0.8,
        0.75,
        0.65,
        0.6,
        0.55,
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.1,
        0.03,
        0.01,
    ]
}

def plot_anal():
    etasyncs = np.geomspace(1e-4, 0.4, 100)

    fig = plt.figure(figsize=(6, 5))
    Ids = [5, 20]
    colors = ['k', 'r']
    for c, Id in zip(colors, Ids):
        I = np.radians(Id)
        ptce2s = 4 * np.sqrt(etasyncs * np.sin(I)) / np.pi * (
            np.sqrt(1 / 10) + 3 / (2 * (1 + np.sqrt(1 / 10))))
        plt.loglog(etasyncs, ptce2s,
                   c=c,
                   label=r'$I = %d^\circ$' % Id)
        ptce2s_2 = 4 * np.sqrt(etasyncs * np.sin(I)) / np.pi * (
            np.sqrt(1 / 2.5) + 3 / (2 * (1 + np.sqrt(1 / 2.5))))
        plt.loglog(etasyncs, ptce2s_2, c=c, ls='--', lw=1.0)
        pkl_fn = '5cum_probs_%d.pkl' % Id
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            counts = pickle.load(f)
        probs_dat = np.array(counts) / 19968
        s_c_vals_I = np.array(s_c_vals[Id])
        idxs = np.where(s_c_vals_I < np.max(etasyncs))[0]
        plt.loglog(s_c_vals_I[idxs], probs_dat[idxs], c=c, ls='', marker='x', ms=5)
    plt.legend(loc='upper left')
    plt.xlabel(r'$\eta_{\rm sync}$')
    plt.ylabel(r'$P_{\rm tCE2}$')
    plt.tight_layout()
    plt.savefig('7anal_ptce', dpi=300)
    plt.close()

def pcaps_compare():
    '''
    plot P_{III --> II} as a function of eta for analytic / semi-analytic
    expressions

    NB: Pcap never increases again for sufficiently large eta, has to do with
    asymmetry of contour probably
    '''
    s_c = 0.06
    I = np.radians(10)
    s = np.linspace(0.3, 10, 100)
    eta=s_c/s
    top, bot = get_ps_anal(I, s_c, s)
    plt.plot(s_c/s, (top + bot) / bot, 'k', label='An')
    top, bot = get_ps_numinterp(I, s_c, s)
    plt.plot(s_c/s, (top(s) + bot(s)) / bot(s), 'r', label='SA')
    plt.ylim(bottom=0)
    plt.xlabel(r'$\eta_{\rm cross}$')
    plt.ylabel(r'$P_{\rm III \to II}$')
    plt.tight_layout()
    plt.savefig('7pcaps_compare', dpi=100)
    plt.close()

if __name__ == '__main__':
    # plot_anal()
    pcaps_compare()
