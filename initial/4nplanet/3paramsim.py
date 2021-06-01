'''
Dong's suggestion: parameterize the evolution of vec{L} first, rather than doing
full n-body type simulations.

for simplicity, we simulate in the inertial frame initially, though perhaps the
corotating frame would be easier to think about eventually
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
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
            np.dot(s_vec_now, xhat))
        phi_arr.append(phi)
    return obliquities, np.degrees(np.unwrap(phi_arr))

if __name__ == '__main__':
    alpha0 = 10
    g2 = 0
    eta = 1 / alpha0
    I1 = np.radians(10)
    I2 = 0
    phi_args = 0
    q_cs2 = roots(I1, eta)[1]
    svec = [-np.sin(q_cs2 - I1),
            0,
            np.cos(q_cs2 - I1)]
    args = [I1, I2, g2, phi_args]
    ret = solve_ivp(dydt_inertial, (0, 10), [*svec, alpha0], args=args,
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
