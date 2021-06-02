'''
Dong's suggestion: parameterize the evolution of vec{L} first, rather than doing
full n-body type simulations.

for simplicity, we simulate in the inertial frame initially, though perhaps the
corotating frame would be easier to think about eventually
'''
from multiprocessing import Pool
import os, pickle, lzma
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=1.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
from scipy.integrate import solve_ivp

from utils import *

def dydt_inertial(t, y, I1, I2, g2, phi=0, eps_alpha=0):
    '''
    y = [*svec, alpha]

    convention: g1 = -1
    '''
    svec = y[ :3]
    alpha = y[3]
    lx = np.sin(I1) * np.cos(-t) + np.sin(I2) * np.cos(-g2 * t + phi)
    ly = np.sin(I1) * np.sin(-t) + np.sin(I2) * np.sin(-g2 * t + phi)
    lz = np.sqrt(1 - lx**2 - ly**2)
    lvec = [lx, ly, lz]
    return np.concatenate((
        alpha * np.dot(svec, lvec) * np.cross(svec, lvec),
        [eps_alpha * alpha]
    ))

def get_CS_angles(s_vec, l_vec):
    '''
    from svecs & lvecs, calculate obliquities & precessional phases relative to
    jhat = [0, 0, 1]
    '''
    jhat = np.array([0, 0, 1])

    # obliquities are easy to calculate, but precessional phases? we need to
    # define the coordinate system with:
    # zhat = l[i]
    # xhat = jtot - proj(jtot, l), (i.e. unit vector in the j-l plane)
    # yhat = zhat cross xhat
    # then phi = np.arctan2(s_vec . xhat, s_vec . yhat)
    obliquities = np.degrees(np.arccos(ts_dot(s_vec, l_vec)))
    phi_arr = []
    for t_idx in range(s_vec.shape[-1]):
        s_vec_now = s_vec[:, t_idx]
        l_vec_now = l_vec[:, t_idx]
        xvec = jhat - np.dot(jhat, l_vec_now) * l_vec_now
        xhat = xvec / np.sqrt(np.sum(xvec**2))
        yhat = np.cross(l_vec_now, xhat)
        phi = np.arctan2(
            np.dot(s_vec_now, yhat),
            np.dot(s_vec_now, xhat)) + np.pi
        phi_arr.append(phi)
    return obliquities, np.degrees(np.unwrap(phi_arr))

def run_example_on_cs(I2=0, g2=0):
    '''
    an example where the svec is initialized directly on CS2
    '''
    alpha0 = 10
    eta = 1 / alpha0
    I1 = np.radians(10)
    phi_args = 0
    q_cs2 = roots(I1, eta)[1]
    svec = [-np.sin(q_cs2 - I1),
            0,
            np.cos(q_cs2 - I1)]
    args = [I1, I2, g2, phi_args]
    ret = solve_ivp(dydt_inertial, (0, 300), [*svec, alpha0], args=args,
                    method='DOP853', atol=1e-9, rtol=1e-9)
    lx = np.sin(I1) * np.cos(-ret.t) + np.sin(I2) * np.cos(-g2 * ret.t + phi_args)
    ly = np.sin(I1) * np.sin(-ret.t) + np.sin(I2) * np.sin(-g2 * ret.t + phi_args)
    lz = np.sqrt(1 - lx**2 - ly**2)
    lvec = np.array([lx, ly, lz])
    obliquities, phirot_arr = get_CS_angles(ret.y[ :3], lvec)
    # phi = np.degrees(np.unwrap(np.arctan2(ret.y[1], ret.y[0])))
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 8),
        sharex=True)
    ax1.plot(ret.t, obliquities)
    ax2.plot(ret.t, phirot_arr)
    ax2.set_xlabel(r'$g_1t$')
    ax1.set_ylabel(r'$\theta$')
    ax2.set_ylabel(r'$\phi_{\rm rot}$')
    plt.tight_layout()
    plt.savefig('/tmp/foo', dpi=200)
    plt.close()

def scan_runner(alpha0=10, I1=np.radians(10), I2=0, g2=0, phi_args=0, dq=0):
    eta = 1 / alpha0
    q_cs2 = roots(I1, eta)[1]
    svec = [-np.sin(q_cs2 + dq - I1),
            0,
            np.cos(q_cs2  + dq- I1)]
    args = [I1, I2, g2, phi_args]
    ret = solve_ivp(dydt_inertial, (0, 100), [*svec, alpha0], args=args,
                    method='DOP853', atol=1e-9, rtol=1e-9)
    lx = (
        np.sin(I1) * np.cos(-ret.t)
        + np.sin(I2) * np.cos(-g2 * ret.t + phi_args))
    ly = (
        np.sin(I1) * np.sin(-ret.t)
        + np.sin(I2) * np.sin(-g2 * ret.t + phi_args))
    lz = np.sqrt(1 - lx**2 - ly**2)
    lvec = np.array([lx, ly, lz])
    obliquities, phirot_arr = get_CS_angles(ret.y[ :3], lvec)
    return obliquities, phirot_arr

def scan_runner_dphi(*args):
    _, phirot_arr = scan_runner(*args)
    return phirot_arr.max() - phirot_arr.min()

def scan_dq(g2s, fns, I2s):
    '''
    scan over distance to CS2
    '''
    alpha0 = 10
    eta = 1 / alpha0
    I1 = np.radians(10)
    sepwidth = np.sqrt(2 * eta * np.sin(I1))
    dqs = np.linspace(-1.7 * sepwidth, 1.7 * sepwidth, 201)
    for g2, fn, I2 in zip(g2s, fns, I2s):
        pkl_fn = fn + '.pkl'
        if not os.path.exists(pkl_fn):
            print('Running %s' % pkl_fn)
            dphis = []
            pool_args = [
                (alpha0, I1, I2, g2, 0, dq)
                for dq in dqs
            ]
            with Pool(8) as p:
                dphis = p.starmap(scan_runner_dphi, pool_args)
            with lzma.open(pkl_fn, 'wb') as f:
                pickle.dump(dphis, f)
        else:
            with lzma.open(pkl_fn, 'rb') as f:
                print('Loading %s' % pkl_fn)
                dphis = pickle.load(f)
        label = r'$g_2 / g_1 = %.1f$, $I_2 = %d^\circ$' % (
            g2, np.degrees(I2))
        plt.semilogy(np.degrees(dqs), dphis, label=label, alpha=0.7)
    plt.title(r'$\alpha / g_1 = %d$, $I_1 = %d^\circ$' %
              (alpha0, np.degrees(I1)))
    plt.ylim(bottom=1)
    plt.axvline(-np.degrees(sepwidth), c='k')
    plt.axvline(np.degrees(sepwidth), c='k')
    plt.legend(loc='lower right', fontsize=10)
    plt.axhline(360, c='k', ls='--')
    plt.xlabel(r'$\theta_{\rm i} - \theta_{\rm CS2}$')
    plt.ylabel(r'$\phi_{\max} - \phi_{\min}$')
    plt.tight_layout()
    plt.savefig('3paramsim/scan_dq', dpi=300)
    plt.close()

def scan_frequency(dqs, fns, I2s):
    '''
    scan over frequency of perturbation
    '''
    alpha0 = 10
    eta = 1 / alpha0
    I1 = np.radians(10)
    g2s = np.geomspace(0.01, 10, 121)
    for dq, fn, I2 in zip(dqs, fns, I2s):
        pkl_fn = fn + '.pkl'
        if not os.path.exists(pkl_fn):
            print('Running %s' % pkl_fn)
            dphis = []
            pool_args = [
                (alpha0, I1, I2, g2, 0, dq)
                for g2 in g2s
            ]
            with Pool(8) as p:
                dphis = p.starmap(scan_runner_dphi, pool_args)
            with lzma.open(pkl_fn, 'wb') as f:
                pickle.dump(dphis, f)
        else:
            with lzma.open(pkl_fn, 'rb') as f:
                print('Loading %s' % pkl_fn)
                dphis = pickle.load(f)
        label = r'$\Delta \theta_{\rm i} = %.1f$, $I_2 = %d^\circ$' % (
            np.degrees(dq), np.degrees(I2))
        plt.loglog(g2s, dphis, label=label, alpha=0.7)
    plt.title(r'$\alpha / g_1 = %d$, $I_1 = %d^\circ$' %
              (alpha0, np.degrees(I1)))
    plt.legend(loc='upper left', fontsize=14)
    plt.ylim(bottom=1)
    plt.axhline(360, c='k', ls='--')
    plt.axvline(1, c='k')
    plt.xlabel(r'$g_2 / g_1$')
    plt.ylabel(r'$\phi_{\max} - \phi_{\min}$')
    plt.tight_layout()
    plt.savefig('3paramsim/scan_frequency', dpi=300)
    plt.close()

def scan_I2(dqs, fns, g2s):
    '''
    scan over frequency of perturbation
    '''
    alpha0 = 10
    eta = 1 / alpha0
    I1 = np.radians(10)
    I2s = np.linspace(0, np.radians(20), 161)
    for dq, fn, g2 in zip(dqs, fns, g2s):
        pkl_fn = fn + '.pkl'
        if not os.path.exists(pkl_fn):
            print('Running %s' % pkl_fn)
            dphis = []
            pool_args = [
                (alpha0, I1, I2, g2, 0, dq)
                for I2 in I2s
            ]
            with Pool(8) as p:
                dphis = p.starmap(scan_runner_dphi, pool_args)
            with lzma.open(pkl_fn, 'wb') as f:
                pickle.dump(dphis, f)
        else:
            with lzma.open(pkl_fn, 'rb') as f:
                print('Loading %s' % pkl_fn)
                dphis = pickle.load(f)
        label = r'$\Delta \theta_{\rm i} = %.1f$, $g_2 / g_1 = %.1f$' % (
            np.degrees(dq), g2)
        plt.semilogy(np.degrees(I2s), dphis, label=label, alpha=0.7)
    plt.title(r'$\alpha / g_1 = %d$, $I_1 = %d^\circ$' %
              (alpha0, np.degrees(I1)))
    plt.legend(loc='lower right', fontsize=10, ncol=2)
    # plt.ylim(bottom=1)
    plt.axhline(360, c='k', ls='--')
    plt.xlabel(r'$I_2$ (Deg)')
    plt.ylabel(r'$\phi_{\max} - \phi_{\min}$')
    plt.tight_layout()
    plt.savefig('3paramsim/scan_I2', dpi=300)
    plt.close()

if __name__ == '__main__':
    os.makedirs('3paramsim', exist_ok=True)
    # run_example_on_cs(I2=np.radians(1), g2=0.1)
    # run_example_on_cs(I2=np.radians(1), g2=10)
    scan_dq(
        [0, 0.1, 1, 3, 10],
        [
            '3paramsim/3scan_none',
            '3paramsim/3scan_slow',
            '3paramsim/3scan_res',
            '3paramsim/3scan_semi',
            '3paramsim/3scan_fast',
        ],
        [0, *([np.radians(3)] * 4)])
    scan_frequency(
        np.radians([0, 2, 6, 9]),
        [
            '3paramsim/3freq0',
            '3paramsim/3freq2',
            '3paramsim/3freq6',
            '3paramsim/3freq9',
        ],
        [np.radians(1)] * 4)
    scan_I2(
        np.radians([2, 2, 2, 2, 2, 2]),
        [
            '3paramsim/3I22',
            '3paramsim/3I20',
            '3paramsim/3I21',
            '3paramsim/3I25',
            '3paramsim/3I23',
            '3paramsim/3I24',
        ],
        [1, 1.5, 2, 2.5, 3, 3.5])
