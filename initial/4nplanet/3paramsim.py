'''
Dong's suggestion: parameterize the evolution of vec{L} first, rather than doing
full n-body type simulations.

for simplicity, we simulate in the inertial frame initially, though perhaps the
corotating frame would be easier to think about eventually
'''
from collections import defaultdict
from scipy.interpolate import interp1d
from multiprocessing import Pool
import os, pickle, lzma
import numpy as np
# plt = None
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    plt.rc('lines', lw=1.5)
    plt.rc('xtick', direction='in', top=True, bottom=True)
    plt.rc('ytick', direction='in', left=True, right=True)
except ModuleNotFoundError:
    plt = None
from scipy.integrate import solve_ivp

from utils import *

def dydt_inertial(t, y, I1, I2, g2, phi=0, eps_alpha=0, eps_tide=0):
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
    so_precession = alpha * np.dot(svec, lvec) * np.cross(svec, lvec)

    tide_disp = eps_tide * (np.cross(svec, np.cross(lvec, svec)))
    return np.concatenate((
        so_precession + tide_disp,
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

def scan_runner(alpha0=10, I1=np.radians(10), I2=0, g2=0, phi_args=0, dq=0,
                return_ret=False, tf=100):
    eta = 1 / alpha0
    q_cs2 = roots(I1, eta)[1]
    svec = [-np.sin(q_cs2 + dq - I1) * np.cos(phi_args),
            -np.sin(q_cs2 + dq - I1) * np.sin(phi_args),
            np.cos(q_cs2  + dq- I1)]
    args = [I1, I2, g2, phi_args]
    ret = solve_ivp(dydt_inertial, (0, tf), [*svec, alpha0], args=args,
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
    if return_ret:
        return obliquities, phirot_arr, ret
    return obliquities, phirot_arr

def scan_runner_dphi(*args):
    _, phirot_arr = scan_runner(*args)
    return phirot_arr.max() - phirot_arr.min()

def scan_dq(g2s, fns, I2s, suffix='', fs=10, **kwargs):
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
    plt.legend(loc='lower right', fontsize=fs, **kwargs)
    plt.axhline(360, c='k', ls='--')
    plt.xlabel(r'$\theta_{\rm i} - \theta_{\rm CS2}$')
    plt.ylabel(r'$\phi_{\max} - \phi_{\min}$')
    plt.tight_layout()
    plt.savefig('3paramsim/scan_dq' + suffix, dpi=300)
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

def plot_phase(g2=3, fn='3paramsim/phase_portrait3', I2=np.radians(1), nq=48,
               stride=5):
    '''
    scan over distance to CS2
    '''
    alpha0 = 10
    eta = 1 / alpha0
    I1 = np.radians(10)
    sepwidth = np.sqrt(2 * eta * np.sin(I1))
    dqs = np.linspace(0, 1.7 * sepwidth, nq)
    pkl_fn = fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        dphis = []
        pool_args = [
            (alpha0, I1, I2, g2, 0, dq)
            for dq in dqs
        ]
        with Pool(8) as p:
            trajs = p.starmap(scan_runner, pool_args)
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(trajs, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            trajs = pickle.load(f)
    plt_trajs = trajs[::stride]
    print(dqs[::stride])
    colors = np.linspace(0.2, 0.9, len(plt_trajs))
    for c, (obliquities, phis) in zip(colors, plt_trajs):
        plt.plot(phis % 360, np.cos(np.radians(obliquities)),
                 c=(c, 1 - c, 0.5),
                 marker='o', ms=1.0, ls='')
        print(np.cos(np.radians(obliquities))[0])
        plt.plot(phis[0] % 360, np.cos(np.radians(obliquities))[0],
                 mfc=(c, 1 - c, 0.5),
                 mec='k',
                 marker='o', ms=3.0, ls='')
    q_cs2 = roots(I1, eta)[1]
    plt.plot(180, np.cos(q_cs2), 'rx', ms=5)
    plt.xlabel(r'$\phi_{\rm rot}$')
    plt.ylabel(r'$\cos \theta$')
    plt.tight_layout()
    plt.savefig(fn, dpi=300)
    plt.close()

def get_lib_frequency(alpha0=10, I1=np.radians(10), phi_args=0,
                      dq=0, tf=20, I2=0, g2=0):
    '''
    returns [0, circ_period] if circulating, else [1, libperiod] if lib

    NB: is in the absence of perturbation
    '''
    _, phirot, ret = scan_runner(alpha0=alpha0, I1=I1, phi_args=phi_args, dq=dq,
                                 return_ret=True, tf=tf, I2=I2, g2=g2)
    if phirot.max() > phirot[0] + 360:
        # circulating right
        all_last_lt = np.where(phirot < phirot[0] + 360)[0]
        # find last idx that is contiguous (some complicated evolutions if I2,
        # g2 \neq 0
        last_lt = np.where(all_last_lt == np.arange(len(all_last_lt)))[0][-1]
        # do a small interp
        t_prev = ret.t[last_lt]
        t_next = ret.t[last_lt + 1]
        phi_prev = phirot[last_lt]
        phi_next = phirot[last_lt + 1]
        t_cross = (phirot[0] + 360 - phi_prev) / (phi_next + 360 - phi_prev) * (
            t_next - t_prev) + t_prev
        ret = 0, t_cross, phirot
    elif phirot.min() < phirot[0] - 360:
        # circulating left
        all_last_lt = np.where(phirot > phirot[0] - 360)[0]
        last_lt = np.where(all_last_lt == np.arange(len(all_last_lt)))[0][-1]
        t_prev = ret.t[last_lt]
        t_next = ret.t[last_lt + 1]
        phi_prev = phirot[last_lt]
        phi_next = phirot[last_lt + 1]
        t_cross = (phirot[0] - 360 - phi_prev) / (phi_next - 360 - phi_prev) * (
            t_next - t_prev) + t_prev
        ret = 0, t_cross, phirot
    elif phirot[1] > phirot[0]:
        # librating cw
        cross_idx = np.where(np.logical_and(
            phirot[ :-1] < phirot[0],
            phirot[1: ] > phirot[0]
        ))[0][0]
        t_prev = ret.t[cross_idx]
        t_next = ret.t[cross_idx + 1]
        phi_prev = phirot[cross_idx]
        phi_next = phirot[cross_idx + 1]
        t_cross = (phirot[0] - phi_prev) / (phi_next - phi_prev) * (
            t_next - t_prev) + t_prev
        ret = 1, t_cross, phirot
    else:
        # librating ccw
        cross_idx = np.where(np.logical_and(
            phirot[ :-1] > phirot[0],
            phirot[1: ] < phirot[0]
        ))[0][0]
        t_prev = ret.t[cross_idx]
        t_next = ret.t[cross_idx + 1]
        phi_prev = phirot[cross_idx]
        phi_next = phirot[cross_idx + 1]
        t_cross = (phirot[0] - phi_prev) / (phi_next - phi_prev) * (
            t_next - t_prev) + t_prev
        ret = 1, t_cross, phirot
    return ret

def get_lyapunov(alpha0=10, I1=np.radians(10), I2=np.radians(1), phi_args=0,
                 dq=0, tf=20, g2=1.25):
    _, phirot1, ret1 = scan_runner(alpha0=alpha0, I1=I1, phi_args=phi_args, dq=dq,
                            return_ret=True, tf=tf, I2=I2, g2=g2)
    _, phirot2, ret2 = scan_runner(alpha0=alpha0, I1=I1, phi_args=phi_args + 1e-5,
                             dq=dq, return_ret=True, tf=tf, I2=I2, g2=g2)
    def get_interp_vec(rett, rety, tnew):
        ynew = np.zeros((3, len(tnew)))
        for i in range(3):
            ynew[i] = interp1d(rett, rety[i])(tnew)
        return ynew
    tnew = np.linspace(0, tf, 10 * max(len(ret1.t), len(ret2.t)))
    yinterp1 = get_interp_vec(ret1.t, ret1.y, tnew)
    yinterp2 = get_interp_vec(ret2.t, ret2.y, tnew)
    ydiff_mag = np.sqrt(np.sum((yinterp2 - yinterp1)**2, axis=0))
    return tnew, ydiff_mag
    # print(phirot1.max() - phirot1.min(), phirot2.max() - phirot2.min(), ydiff_mag[0])
    # plt.semilogy(tnew, ydiff_mag)
    # plt.savefig('/tmp/foo')
    # plt.close()

def plot_resonances(fn='3paramsim/resonances', g2=2, phi_args=0):
    dqs = np.linspace(-20, 20, 201)
    pkl_fn = fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        dq_plots = [[], []]
        periods = [[], []]
        periods_r = [[], []]
        ydiffmaxes = [[], []]
        alpha0 = 10
        tf = 30
        args1 = [
            (alpha0, np.radians(10), phi_args, np.radians(dq), tf, 0, 0)
            for dq in dqs
        ]
        args1_realg2 = [
            (alpha0, np.radians(10), phi_args, np.radians(dq), tf, np.radians(1), g2)
            for dq in dqs
        ]
        args2 = [
            (10, np.radians(10), np.radians(1), 0, np.radians(dq), 100, g2)
            for dq in dqs
        ]
        with Pool(8) as p:
            rets1 = p.starmap(get_lib_frequency, args1)
            rets1_real = p.starmap(get_lib_frequency, args1_realg2)
            rets2 = p.starmap(get_lyapunov, args2)
        for dq, ret, ret1real, ret2 in zip(dqs, rets1, rets1_real, rets2):
            cycle_type, cycle_period, phirot = ret
            _, cycle_periodr, phirotr = ret1real

            periods[cycle_type].append(cycle_period)
            periods_r[cycle_type].append(cycle_periodr)
            dq_plots[cycle_type].append(dq)

            tnew, ydiff_mag = ret2
            ydiffmaxes[cycle_type].append(np.max(ydiff_mag))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump((dq_plots, periods, periods_r, ydiffmaxes), f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            dq_plots, periods, periods_r, ydiffmaxes = pickle.load(f)
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 11), sharex=True)
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(7, 11), sharex=True)
    for i, (c, lbl) in enumerate(zip(['b', 'g'], ['Circ', 'Lib'])):
        ax1.plot(dq_plots[i], periods[i], '%so' % c, ms=2, label=lbl)
        # ax2.plot(dq_plots[i], periods_r[i], '%so' % c, ms=2)
        ax3.semilogy(dq_plots[i], ydiffmaxes[i], '%so' % c, ms=2)
    ax1.legend(fontsize=16)
    ax1.axhline(2 * np.pi / (g2 - 1), c='k')
    ax3.set_xlabel(r'$\Delta \theta$ (deg)')
    ax1.set_ylabel(r'$P_{\rm lib, 0}$')
    # ax2.set_ylabel(r'$P_{\rm lib}$')
    ax3.set_ylabel(r'Max $\Delta \hat{\mathbf{s}}$')
    ax1.set_title(r'$g_2 / g_1 = %.1f$' % g2)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig(fn, dpi=300)
    plt.close()

def nondisp_explore():
    os.makedirs('3paramsim', exist_ok=True)
    # run_example_on_cs(I2=np.radians(1), g2=0.1)
    # run_example_on_cs(I2=np.radians(1), g2=10)
    # scan_dq(
    #     [0, 0.1, 1, 3, 10],
    #     [
    #         '3paramsim/3scan_none',
    #         '3paramsim/3scan_slow',
    #         '3paramsim/3scan_res2',
    #         '3paramsim/3scan_semi',
    #         '3paramsim/3scan_fast',
    #     ],
    #     [0, *([np.radians(3)] * 4)])
    # scan_dq(
    #     [1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2],
    #     [
    #         '3paramsim/3scan_res0',
    #         '3paramsim/3scan_res1',
    #         '3paramsim/3scan_res2',
    #         '3paramsim/3scan_res3',
    #         '3paramsim/3scan_res4',
    #         '3paramsim/3scan_res5',
    #         '3paramsim/3scan_res6',
    #         '3paramsim/3scan_res7',
    #     ],
    #     [np.radians(3)] * 8,
    #     suffix='_res',
    #     fs=8,
    #     ncol=2)
    # scan_frequency(
    #     np.radians([0, 2, 6, 9]),
    #     [
    #         '3paramsim/3freq0',
    #         '3paramsim/3freq2',
    #         '3paramsim/3freq6',
    #         '3paramsim/3freq9',
    #     ],
    #     [np.radians(1)] * 4)
    # scan_I2(
    #     np.radians([2, 2, 2, 2, 2, 2]),
    #     [
    #         '3paramsim/3I22',
    #         '3paramsim/3I20',
    #         '3paramsim/3I21',
    #         '3paramsim/3I25',
    #         '3paramsim/3I23',
    #         '3paramsim/3I24',
    #     ],
    #     [1, 1.5, 2, 2.5, 3, 3.5])

    # plot_phase(g2=3, fn='3paramsim/phase_portrait3', I2=np.radians(1), nq=48)
    # plot_phase(g2=1, fn='3paramsim/phase_portrait1', I2=np.radians(1), nq=48)
    # plot_phase(g2=0.1, fn='3paramsim/phase_portrait01', I2=np.radians(1), nq=48)
    # plot_phase(g2=2, fn='3paramsim/phase_portrait2', I2=np.radians(1), nq=48)
    # plot_phase(g2=10, fn='3paramsim/phase_portrait10', I2=np.radians(1), nq=48)
    # plot_phase(g2=0, fn='3paramsim/phase_portrait0', I2=np.radians(0), nq=48)

    # plot_resonances(fn='3paramsim/resonances13', g2=1.3)
    # plot_resonances(fn='3paramsim/resonances15', g2=1.5)
    # plot_resonances()
    # plot_resonances(fn='3paramsim/resonances3', g2=3)
    pass

TIDE_FLDR = '3paramtide/'
def disp_run_ex(I2=0, g2=0, fn='/tmp/foo', dq=0.05, tf=300, eps_tide=5e-2,
                plot=False, q0=None, phi0=np.pi, rot_mult=0,
                alpha0=10, plot_mode2cs2=False, plot_mode1cs2=False,
                plot_mixed=False):
    '''
    run an example of dissipation
    '''
    eta = 1 / alpha0
    I1 = np.radians(10)
    eps_alpha = 0
    phi_args = 0
    q_cs2 = roots(I1, eta)[1]
    if q0 is None:
        q0 = q_cs2 + dq
    svec = [np.sin(q0 - I1) * np.cos(phi0),
            np.sin(q0 - I1) * np.sin(phi0),
            np.cos(q0 - I1)]
    args = [I1, I2, g2, phi_args, eps_alpha, eps_tide]
    pkl_fn = fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        ret = solve_ivp(dydt_inertial, (0, tf), [*svec, alpha0], args=args,
                        method='DOP853', atol=1e-9, rtol=1e-9)
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump((ret), f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    lx = np.sin(I1) * np.cos(-ret.t) + np.sin(I2) * np.cos(-g2 * ret.t + phi_args)
    ly = np.sin(I1) * np.sin(-ret.t) + np.sin(I2) * np.sin(-g2 * ret.t + phi_args)
    lz = np.sqrt(1 - lx**2 - ly**2)
    lvec = np.array([lx, ly, lz])
    obliquities, phirot_arr = get_CS_angles(ret.y[ :3], lvec)
    if not plot:
        return obliquities, phirot_arr, ret

    phirot_arr += np.degrees(rot_mult * ret.t)
    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1,
        figsize=(8, 8),
        sharex=True)
    ax0.plot(ret.t, np.degrees(np.arccos(lz)))
    ax1.plot(ret.t, obliquities)
    ax2.plot(ret.t, phirot_arr)
    ax0.set_ylabel(r'$I$ (deg)')
    ax1.set_ylabel(r'$\theta_{\rm sl}$ (deg)')
    ax2.set_ylabel(r'$\phi_{\rm sl} + %.1fg_1t$ (deg)' % rot_mult)
    ax2.set_xlabel(r'$g_1t$')

    if plot_mode2cs2:
        eta = g2 / alpha0
        q_cs2 = find_cs(np.radians(1), eta, +np.pi / 2)
        ax1.axhline(np.degrees(np.arccos(q_cs2)), c='k', ls='-.', lw=2)
    if plot_mode1cs2:
        eta = 1 / alpha0
        q_cs2 = roots(I1, eta)[1]
        ax1.axhline(np.degrees(q_cs2), c='k', ls='-.', lw=2)
    if plot_mixed:
        mu_res = (1 + g2) / (2 * alpha0)
        ax1.axhline(np.degrees(np.arccos(mu_res)), c='k', ls='-.', lw=2)
        amp = np.sin(np.arccos(mu_res)) * np.degrees(I2) * 2
        ax1.axhline(np.degrees(np.arccos(mu_res)) + amp, c='r', ls='-.', lw=2)

    # plt.scatter(phirot_arr % 360, np.cos(np.radians(obliquities)),
    #             c=ret.t)
    # plt.colorbar()
    plt.tight_layout()
    plt.savefig(fn, dpi=200)

    ax2.set_xlim(left=ret.t[-1] - 10, right=ret.t[-1])
    leftidx = np.where(ret.t > ret.t[-1] - 10)[0][0]
    minylim = phirot_arr[leftidx:-1].min()
    maxylim = phirot_arr[leftidx:-1].max()
    dy = maxylim - minylim
    ax2.set_ylim(minylim - 0.1 * dy, maxylim + 0.1 * dy)
    plt.tight_layout()
    plt.savefig(fn + 'zoom', dpi=200)
    plt.close()

def outcome_runner_full(I2, g2, q0, phi0, tf, eps_tide):
    '''
    run an individual example from (q0, phi0) and return the final obliquity

    NB: phi0=pi = CS2, per usual convention
    '''
    alpha0 = 10
    eta = 1 / alpha0
    I1 = np.radians(10)
    phi_args = 0
    eps_alpha = 0
    svec = [np.sin(q0) * np.cos(phi0),
            np.sin(q0) * np.sin(phi0),
            np.cos(q0)]
    args = [I1, I2, g2, phi_args, eps_alpha, eps_tide]
    ret = solve_ivp(dydt_inertial, (0, tf), [*svec, alpha0], args=args,
                    method='DOP853', atol=1e-9, rtol=1e-9)

    lx = np.sin(I1) * np.cos(-ret.t) + np.sin(I2) * np.cos(-g2 * ret.t + phi_args)
    ly = np.sin(I1) * np.sin(-ret.t) + np.sin(I2) * np.sin(-g2 * ret.t + phi_args)
    lz = np.sqrt(1 - lx**2 - ly**2)
    lvec = np.array([lx, ly, lz])
    return get_CS_angles(ret.y[ :3], lvec)

def outcome_runner(*args):
    '''
    run an individual example from (q0, phi0) and return the final obliquity

    NB: phi0=pi = CS2, per usual convention
    '''
    obliquities, phirot_arr = outcome_runner_full(*args)
    return obliquities[-1], obliquities[0], phirot_arr[0]

def find_cs(I, eta, q0):
    f = lambda q: -eta * np.sin(q - I) + np.sin(q) * np.cos(q)
    fp = lambda q: -eta * np.cos(q - I) + np.cos(2 * q)
    return np.cos(opt.newton(f, q0, fprime=fp))

def plot_sep(I, eta, N=100, c='k'):
    # eta = g
    eta_c = (np.sin(I)**(2/3) + np.cos(I)**(2/3))**(-3/2)
    def H(mu, phi):
        return -0.5 * mu**2 + eta * (
            mu * np.cos(I) -
            np.sqrt(1 - mu**2) * np.sin(I) * np.cos(phi))
    mu_2 = find_cs(I, eta, +np.pi/2)
    if I == 0:
        return None, mu_2
    plt.plot(180, mu_2, '%so' % c, ms=6, ls='')
    if eta > eta_c:
        return None, mu_2

    mu_1 = find_cs(I, eta, 0)
    mu_4 = find_cs(I, eta, -np.pi/2)
    H4 = H(mu_4, 0)
    phi_grid, mu_grid = np.meshgrid(np.linspace(0, 360, N),
                                    np.linspace(-1, 1, N))
    H_grid = H(mu_grid, np.radians(phi_grid))
    plt.contour(phi_grid, mu_grid, H_grid,
                levels=[H4], colors=[c], linestyles=['-'])
    return mu_1, mu_2

def plot_outcomes(I2=0, g2=0, tf=1000, eps_tide=1e-2,
                  fn='{}outcomes0'.format(TIDE_FLDR),
                  num_pts=5000, to_plot=True):
    q_vals = np.arccos(np.random.uniform(-1, 1, num_pts))
    phi_vals = np.random.uniform(0, 2 * np.pi, num_pts)
    args = [
        (I2, g2, q0, phi0, tf, eps_tide)
        for q0, phi0 in zip(q_vals, phi_vals)
    ]
    pkl_fn2 = fn + '_short.pkl'
    if not os.path.exists(pkl_fn2):
        print('Running %s' % pkl_fn2)
        # pkl_fn = fn + '.pkl'
        # if not os.path.exists(pkl_fn):
        #     print('Running %s' % pkl_fn)
        #     with Pool(64) as p:
        #         rets = p.starmap(outcome_runner_full, args)
        #     with lzma.open(pkl_fn, 'wb') as f:
        #         pickle.dump((rets, q_vals, phi_vals), f)
        # else:
        #     with lzma.open(pkl_fn, 'rb') as f:
        #         print('Loading %s' % pkl_fn)
        #         rets, q_vals, phi_vals = pickle.load(f)

        with Pool(64) as p:
            rets = p.starmap(outcome_runner, args)
        q_fs, q_is, phi_is = np.array(rets).T
        with lzma.open(pkl_fn2, 'wb') as f:
            pickle.dump((q_fs, q_is, phi_is, q_vals, phi_vals), f)
    else:
        with lzma.open(pkl_fn2, 'rb') as f:
            print('Loading %s' % pkl_fn2)
            q_fs, q_is, phi_is, q_vals, phi_vals = pickle.load(f)
    if plt is not None and to_plot:
        plt.scatter(phi_is, np.cos(np.radians(q_is)), c=q_fs, s=2)
        c = plt.colorbar()
        ticks = c.get_ticks()
        c.set_ticks(ticks)
        c.set_ticklabels(['$%d^\circ$ ($%.2f$)' % (
            q, np.cos(np.radians(q))) for q in ticks])
        c.ax.tick_params(labelsize=10)
        c.set_label(r'$\theta_{\rm f}$')

        mu1_1, mu2_1 = plot_sep(np.radians(10), 0.1)
        mu1_2, mu2_2 = plot_sep(I2, g2 / 10, c='b')
        plt.xlabel(r'$\phi_{\rm sl, i}$')
        plt.ylabel(r'$\cos \theta_{\rm sl, i}$')
        plt.tight_layout()
        plt.savefig(fn, dpi=300)
        plt.close()

        plt.hist(q_fs, bins=np.linspace(0, 90, 101))
        plt.axvline(np.degrees(np.arccos(mu2_1)), c='k', ls='--')
        plt.axvline(np.degrees(np.arccos(mu2_2)), c='b', ls='--')
        if mu1_1 is not None:
            plt.axvline(np.degrees(np.arccos(mu1_1)), c='k', ls='-.')
        if mu1_2 is not None:
            plt.axvline(np.degrees(np.arccos(mu1_2)), c='b', ls='-.')
        plt.xlim(0, 90)
        plt.xlabel(r'$\theta_{\rm sl, f}$')
        plt.axvline(np.degrees(np.arccos((g2 + 1) / (20))), c='r', ls='--')
        plt.axvline(np.degrees(np.arccos((g2 + 1) / (20))) + 2 * np.degrees(I2),
                    c='r', ls=':')
        plt.axvline(np.degrees(np.arccos((g2 + 1) / (20))) - 2 * np.degrees(I2),
                    c='r', ls=':')

        # plt.hist(np.cos(np.radians(q_fs)), bins=np.linspace(0, 1, 101))
        # plt.axvline(mu2_1, c='k', ls='--')
        # plt.axvline(mu2_2, c='b', ls='--')
        # if mu1_1 is not None:
        #     plt.axvline(mu1_1, c='k', ls='-.')
        # if mu1_2 is not None:
        #     plt.axvline(mu1_2, c='b', ls='-.')
        # plt.axvline((g2 + 1) / (20), c='r', ls='--')
        # plt.xlim(0, 1)
        # plt.xlabel(r'$\cos \theta_{\rm sl, f}$')

        plt.tight_layout()
        plt.savefig(fn + '_hist', dpi=300)
        plt.close()
    return phi_is, q_is, q_fs

def plot_cum(plot_ind=True):
    kw_params = [
        dict(I2=np.radians(0), g2=00, tf=5000, eps_tide=2e-3,
             fn='{}outcomes00'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(1), g2=0.1, tf=5000, eps_tide=2e-3,
             fn='{}outcomes01'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(1), g2=1.5, tf=5000, eps_tide=2e-3,
             fn='{}outcomes15'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(1), g2=2.0, tf=5000, eps_tide=2e-3,
             fn='{}outcomes20'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(1), g2=2.5, tf=5000, eps_tide=2e-3,
             fn='{}outcomes25'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(1), g2=3.0, tf=5000, eps_tide=2e-3,
             fn='{}outcomes30'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(1), g2=3.5, tf=5000, eps_tide=2e-3,
             fn='{}outcomes35'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(1), g2=5, tf=5000, eps_tide=2e-3,
             fn='{}outcomes05'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(1), g2=7, tf=5000, eps_tide=2e-3,
             fn='{}outcomes07'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(1), g2=8, tf=5000, eps_tide=2e-3,
             fn='{}outcomes08'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(1), g2=10, tf=5000, eps_tide=2e-3,
             fn='{}outcomes010'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(2), g2=10, tf=5000, eps_tide=2e-3,
             fn='{}outcomes2_010'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(1), g2=12, tf=5000, eps_tide=2e-3,
             fn='{}outcomes012'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(1), g2=15, tf=5000, eps_tide=2e-3,
             fn='{}outcomes015'.format(TIDE_FLDR), num_pts=3000),

        dict(I2=np.radians(3), g2=0.1, tf=5000, eps_tide=2e-3,
             fn='{}3outcomes01'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(3), g2=1.5, tf=5000, eps_tide=2e-3,
             fn='{}3outcomes15'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(3), g2=2.0, tf=5000, eps_tide=2e-3,
             fn='{}3outcomes20'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(3), g2=2.5, tf=5000, eps_tide=2e-3,
             fn='{}3outcomes25'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(3), g2=3.0, tf=5000, eps_tide=2e-3,
             fn='{}3outcomes30'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(3), g2=3.5, tf=5000, eps_tide=2e-3,
             fn='{}3outcomes35'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(3), g2=5, tf=5000, eps_tide=2e-3,
             fn='{}3outcomes05'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(3), g2=7, tf=5000, eps_tide=2e-3,
             fn='{}3outcomes07'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(3), g2=8, tf=5000, eps_tide=2e-3,
             fn='{}3outcomes08'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(3), g2=10, tf=5000, eps_tide=2e-3,
             fn='{}3outcomes010'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(3), g2=12, tf=5000, eps_tide=2e-3,
             fn='{}3outcomes012'.format(TIDE_FLDR), num_pts=3000),
        dict(I2=np.radians(3), g2=15, tf=5000, eps_tide=2e-3,
             fn='{}3outcomes015'.format(TIDE_FLDR), num_pts=3000),
    ]
    dat_by_I = defaultdict(list)
    for kwargs in kw_params:
        phi_is, q_is, q_fs = plot_outcomes(**kwargs, to_plot=plot_ind)
        if plt is None:
            continue
        I2 = kwargs.get('I2')
        g2 = kwargs.get('g2')

        eta1 = 0.1
        eta2 = g2 / 10

        q1_2 = np.degrees(np.arccos(find_cs(np.radians(10), eta1, +np.pi/2)))
        q1_1 = np.degrees(np.arccos(find_cs(np.radians(10), eta1, 0)))
        q2_1 = np.degrees(np.arccos(find_cs(I2, eta2, 0)))
        q2_2 = np.degrees(np.arccos(find_cs(I2, eta2, +np.pi / 2)))
        mixed_mode = np.degrees(np.arccos((g2 + 1) / 20))

        mixed_amp = np.degrees(I2) * 2

        # This is so messy, but to get non-intersecting sets, many priorities
        # have to be implemented... should probably have made something
        # systematic
        cs1_2_idx = np.where(np.abs(q_fs - q1_2) < 10)[0]
        cs1_1_idx = np.where(np.logical_and(
            np.abs(q_fs - q1_1) < 6, np.logical_and(
                np.abs(q_fs - q1_1) < np.abs(q_fs - q2_1),
                np.abs(q_fs - q1_1) < np.abs(q_fs - q2_2),
            )
        ))[0]
        if eta2 > get_etac(I2):
            cs2_1_idx = []
        else:
            cs2_1_idx = np.where(np.logical_and(
                np.abs(q_fs - q2_1) < 6,
                np.abs(q_fs - q1_1) > np.abs(q_fs - q2_1)
            ))[0]
        cs2_2_idx = np.where(np.logical_and(
            np.logical_and(
                np.abs(q_fs - q2_2) < 10,
                np.abs(q_fs - q1_2) > 10, # M1-CS2 always has priority
            ),
            np.abs(q_fs - q1_1) > np.abs(q_fs - q2_2)
        ))[0]
        mixed_idx = np.where(np.logical_and(
            np.abs(q_fs - mixed_mode) < mixed_amp,
            np.abs(q_fs - q1_2) > 10
        ))[0]

        # print(q1_1, q1_2, q2_1, q2_2, eta2)
        # printlencs1_1_idx, len(cs1_2_idx), len(cs2_1_idx), len(cs2_2_idx),
        #       len(mixed_idx))
        idxs = [cs1_1_idx, cs1_2_idx, cs2_1_idx, cs2_2_idx, mixed_idx]
        for idx, idx_arr in enumerate(idxs):
            for idx2, idx_arr2 in enumerate(idxs[idx + 1: ]):
                assert len(np.intersect1d(idx_arr, idx_arr2)) == 0, \
                    '{} and {} indicies overlap'.format(idx, idx + 1 + idx2)
        dat_by_I[I2].append(
            (eta2, len(cs1_2_idx), len(cs1_1_idx), len(cs2_1_idx),
             len(cs2_2_idx), len(mixed_idx)))
    if plt is None:
        return
    fig, axs = plt.subplots(
        2, 1,
        figsize=(8, 8),
        sharex=True)
    ax1, ax2 = axs
    for ax, I_val in zip(axs, [np.radians(1), np.radians(3)]):
        dat = np.array(dat_by_I[I_val]).T
        eta2s = dat[0]
        ax.set_ylim(0, 1)
        for d, c, label in zip(
            dat[1: ],
            ['k', 'tab:olive', 'tab:blue', 'tab:orange', 'tab:green'],
            ['M1-CS2', 'M1-CS1', 'M2-CS1', 'M2-CS2', 'Mixed'],
        ):
            ax.plot(eta2s, d / len(q_fs), marker='o', linestyle='',
                    label=label, alpha=0.7, c=c, ms=5)
        ax.plot(eta2s, 1 - np.sum(dat[1: , :], axis=0) / len(q_fs), marker='o',
                linestyle='', label='Other', alpha=0.5, c='tab:red', ms=5)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(xlim[-1] * 0.95, ylim[-1] * 0.95,
                r'$I_2 = %d^\circ$' % np.degrees(I_val),
                ha='right', va='top')
        ax.legend(ncol=2, fontsize=14, loc='center left')
        ax.axvline(get_etac(I_val), c='k', ls='--')
    ax1.set_ylabel(r'Prob')
    ax2.set_ylabel(r'Prob')
    ax2.set_xlabel(r'$\eta_2$')
    plt.tight_layout()
    plt.savefig('3outcomes', dpi=300)
    plt.close()

def mm_phase_portrait(I2=np.radians(1), g2=10, fn='/tmp/foo', dq=0.05, tf=60,
                      alpha0=10, eps_tide=0, plot_mult=0):
    '''
    mixed mode phase portrait, determined numerically
    '''
    I1 = np.radians(10)
    eps_alpha = 0
    phi_args = 0
    nrow = 3
    ncol = 4
    q0s = np.arccos(np.linspace(np.cos(2 * I1), np.cos(np.pi / 2), nrow * ncol))
    pkl_fn = fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        rets = []
        for idx, q0 in enumerate(q0s):
            rets_curr = []
            for phi0 in np.linspace(0, 2 * np.pi, 4, endpoint=False):
                print('Running', q0, phi0)
                svec = [np.sin(q0 - I1) * np.cos(phi0),
                        np.sin(q0 - I1) * np.sin(phi0),
                        np.cos(q0 - I1)]
                args = [I1, I2, g2, phi_args, eps_alpha, eps_tide]
                ret = solve_ivp(dydt_inertial, (0, tf), [*svec, alpha0], args=args,
                                method='DOP853', atol=1e-9, rtol=1e-9)
                rets_curr.append(ret)
            rets.append(rets_curr)
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump((rets), f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            rets = pickle.load(f)
    fig, axs = plt.subplots(
        nrow, ncol,
        figsize=(16, 9),
        sharex=True, sharey=True)
    mu_res = (1 + g2) / (2 * alpha0)

    def get_phirot_mu(svec, freq, t):
        # phi_inertial = np.unwrap(np.arctan2(svec[1], svec[0]))
        # phirot = phi_inertial + freq * t
        # return svec[2], (phirot % (2 * np.pi))

        lx = np.sin(I1) * np.cos(-t) + np.sin(I2) * np.cos(-g2 * t + phi_args)
        ly = np.sin(I1) * np.sin(-t) + np.sin(I2) * np.sin(-g2 * t + phi_args)
        lz = np.sqrt(1 - lx**2 - ly**2)
        lvec = np.array([lx, ly, lz])
        phi_lvec = np.arctan2(ly, lx) # convention

        mu_sl = ts_dot(lvec, svec)
        yhat = np.array([
            np.cos(phi_lvec + np.pi / 2),
            np.sin(phi_lvec + np.pi / 2),
            np.zeros_like(phi_lvec),
        ])
        xhat = ts_cross(yhat, lvec)
        phi_sl = np.arctan2(
            ts_dot(yhat, svec),
            ts_dot(xhat, svec),
        )
        return mu_sl, (phi_sl + freq * t) % (2 * np.pi)

    # plot again but w/ different resonant angle
    for idx, (freq_str, freq) in enumerate(zip(
        [r'0', r'(g_2 - g_1)', r'(g_2 - g_1) / 2', r'(g_1 - g_2)3/2'],
        [0, g2 - 1, (g2 - 1) / 2, 3 * (g2 - 1) / 2]
    )):
        fig, axs = plt.subplots(
            nrow, ncol,
            figsize=(16, 9),
            sharex=True, sharey=True)
        for rets_curr, ax, q0 in zip(rets, axs.flat, q0s):
            for c, ret in zip(['k', 'r', 'b', 'g'], rets_curr):
                svec = ret.y[ :3]
                idxs = np.where(ret.t > plot_mult * tf)[0]
                mu_sl, phirot = get_phirot_mu(svec, freq, ret.t)
                ax.scatter(phirot[idxs], mu_sl[idxs], s=2, c=c,
                           alpha=0.3)
                ax.plot(phirot[0], mu_sl[0], '%so' % c, ms=10)
                # ax.plot(0, mu_res, 'bx', ms=10)
                # ax.plot(np.pi, mu_res, 'bx', ms=10)
                ax.axhline(mu_res, c='k', ls='--', lw=2)

                eta = 1 / alpha0
                q_cs2 = roots(I1, eta)[1]
                ax.axhline(np.cos(q_cs2), c='k', ls='-.', lw=2)
        axs[-1][0].set_xlabel(r'$\phi_{\rm sl} + t%s$' % freq_str)
        axs[-1][0].set_ylabel(r'$\cos \theta_{\rm sl}$')
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.02, wspace=0.02)
        plt.savefig(fn + '_mode{}'.format(idx), dpi=200)
        plt.close()

def testo():
    ''' plots of dot{I}, dot{W} cos/sin I '''
    t = np.linspace(0, 10, 10000)
    g2 = 5
    phi_args = 0
    I1= np.radians(10)
    I2 = np.radians(1)

    lx = np.sin(I1) * np.cos(-t) + np.sin(I2) * np.cos(-g2 * t + phi_args)
    ly = np.sin(I1) * np.sin(-t) + np.sin(I2) * np.sin(-g2 * t + phi_args)
    lz = np.sqrt(1 - lx**2 - ly**2)
    lvec = np.array([lx, ly, lz])

    I = np.arccos(lz)
    W = np.unwrap(np.arctan2(ly, lx))

    dIdt = np.diff(I) / np.diff(t)
    dWdt = np.diff(W) / np.diff(t)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 8),
        sharex=True)
    ax1.plot(t[1: ], dIdt)
    ax2.plot(t[1: ], dWdt * np.cos(I[1: ]))
    ax2.plot(t[1: ], dWdt * np.sin(I[1: ]))
    plt.savefig('/tmp/foo', dpi=200)
    plt.close()

if __name__ == '__main__':
    # testo()

    os.makedirs(TIDE_FLDR, exist_ok=True)
    # disp_run_ex(I2=np.radians(1), g2=2, tf=1000,
    #             fn='%sdisp_1' % TIDE_FLDR, dq=0, plot=True)
    # disp_run_ex(I2=np.radians(1), g2=2, tf=1000, eps_tide=0,
    #             fn='%sdisp_1_notide' % TIDE_FLDR, dq=0, plot=True)
    # disp_run_ex(I2=np.radians(2), g2=2, tf=1000, fn='%sdisp_2' % TIDE_FLDR,
    #             dq=np.radians(2), plot=True)
    # disp_run_ex(I2=np.radians(2), g2=2, tf=1000, eps_tide=0,
    #             fn='%sdisp_2_notide' % TIDE_FLDR, dq=np.radians(2), plot=True)
    # disp_run_ex(I2=np.radians(0), g2=0, tf=1000, eps_tide=1e-2,
    #             fn='%sdisp_zero' % TIDE_FLDR, dq=0, plot=True,
    #             q0=np.pi / 2 + np.radians(15), phi0 = 2 * np.pi - 1)
    # disp_run_ex(I2=np.radians(0), g2=0, tf=5000, eps_tide=2e-3,
    #             fn='%sdisp_zero_long' % TIDE_FLDR, dq=0, plot=True,
    #             q0=np.pi / 2 + np.radians(15), phi0 = 2 * np.pi - 1)

    plot_cum(plot_ind=True)

    # disp_run_ex(I2=np.radians(1), g2=10, tf=3000, eps_tide=4e-3,
    #             fn='%sdisp_10_res' % TIDE_FLDR, q0=np.radians(55), plot=True,
    #             rot_mult=9/2, plot_mixed=True)
    # disp_run_ex(I2=np.radians(1), g2=10, tf=3000, eps_tide=4e-3,
    #             fn='%sdisp_10_cs2' % TIDE_FLDR, q0=np.radians(90), plot=True,
    #             rot_mult=0, plot_mode1cs2=True)
    # disp_run_ex(I2=np.radians(1), g2=10, tf=3000, eps_tide=4e-3,
    #             fn='%sdisp_10_cs1' % TIDE_FLDR, q0=np.radians(10), plot=True,
    #             rot_mult=9, plot_mode2cs2=True)
    # disp_run_ex(I2=np.radians(1), g2=7, tf=3000, eps_tide=4e-3,
    #             fn='%sdisp_7_res' % TIDE_FLDR, q0=np.radians(66), plot=True,
    #             rot_mult=3, plot_mixed=True)
    # disp_run_ex(I2=np.radians(1), g2=15, tf=3000, eps_tide=4e-3,
    #             fn='%sdisp_15_res' % TIDE_FLDR, q0=np.radians(36), plot=True,
    #             rot_mult=7, plot_mixed=True)

    # disp_run_ex(I2=np.radians(1), g2=10, alpha0=15, tf=3000, eps_tide=4e-3,
    #             fn='%sdisp_10_res2' % TIDE_FLDR, q0=np.radians(68), plot=True,
    #             rot_mult=9/2, plot_mixed=True)
    # disp_run_ex(I2=np.radians(1), g2=10, alpha0=12, tf=3000, eps_tide=4e-3,
    #             fn='%sdisp_10_res3' % TIDE_FLDR, q0=np.radians(68), plot=True,
    #             rot_mult=9/2, plot_mixed=True)
    # disp_run_ex(I2=np.radians(1.5), g2=10, tf=3000, eps_tide=4e-3,
    #             fn='%sdisp_10_res4' % TIDE_FLDR, q0=np.radians(55), plot=True,
    #             rot_mult=9/2, plot_mixed=True)

    # mm_phase_portrait(fn='3paramtide/mm')
    # mm_phase_portrait(fn='3paramtide/mm_I22', I2=np.radians(2))
    # mm_phase_portrait(fn='3paramtide/mm_g2_7', g2=7)
    # mm_phase_portrait(fn='3paramtide/mm_I2_2', g2=7, I2=np.radians(2))
    # mm_phase_portrait(fn='3paramtide/mm_tide', eps_tide=5e-3, tf=1000,
    #                   plot_mult=0.99)
    # mm_phase_portrait(fn='3paramtide/mm_I22_tide', eps_tide=5e-3, tf=1000,
    #                   plot_mult=0.99, I2=np.radians(2))
    # mm_phase_portrait(fn='3paramtide/mm_g2_7_tide', eps_tide=5e-3, tf=1000,
    #                   plot_mult=0.99, g2=7)
    # mm_phase_portrait(fn='3paramtide/mm_I2_2_tide', eps_tide=5e-3, tf=1000,
    #                   plot_mult=0.99, g2=7, I2=np.radians(2))
