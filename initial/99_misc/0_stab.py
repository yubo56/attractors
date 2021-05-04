'''
look for existence/stability of CS2 as tidal dissipation increases
'''
import numpy as np
import scipy.optimize as opt

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

I = np.radians(20)
eta = 0.2
mu2 = eta * np.cos(I)# / (1 + eta * np.sin(I))
mu4 = eta * np.cos(I)# / (1 - eta * np.sin(I))
eps_crit = eta * np.sin(I) / (1 - (eta * np.cos(I))**2)

def get_dydt(tide):
    def dydt(s):
        x, y, z = s
        return np.array([
            y * z - eta * y * np.cos(I) - tide * z * x,
            -x * z + eta * (x * np.cos(I) - z * np.sin(I)) - tide * z * y,
            eta * y * np.sin(I) + tide * (1 - z**2),
        ])
    return dydt

def get_cs2(eps, eta=eta, tol=1e-10, log=False, method='hybr'):
    cs2 = [np.sqrt(1 - mu2**2), 0, mu2]
    ret = opt.root(get_dydt(eps), cs2, tol=tol, method=method)
    if log and not ret.success:
        print('get_cs2 did not succeed')
        print(ret)
    return ret

def get_eps_crit_num():
    left = 0
    right = eps_crit
    while right - left > 1e-12:
        eps = (right + left) / 2
        sol2 = get_cs2(eps) # either cs2 or cs4 work here, just check exist
        if sol2.success:
            left = eps
        else:
            right = eps
    return left # only left is guaranteed success at this point

eps_crit_num = get_eps_crit_num()

def get_jac(tide):
    ''' get jacobian d(dy/dt)_i/dy_j for params '''
    def jac(s):
        x, y, z = s
        return [
            [-tide * z, z - eta * np.cos(I), y - tide * x],
            [-z + eta * np.cos(I), -tide * z, -x - eta * np.sin(I) - tide * y],
            [0, eta * np.sin(I), -2 * tide * z],
        ]
    return jac

def to_ang(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    q = np.arccos(z / r)
    phi = (np.arctan2(y / np.sin(q), x / np.sin(q)) + np.pi)\
        % (2 * np.pi)
    return np.array([q, phi])

def to_cart(q, phi):
    return [
        -np.sin(q) * np.cos(phi),
        -np.sin(q) * np.sin(phi),
        np.cos(q),
    ]

def get_cs1(eps, tol=1e-10, log=False, method='hybr'):
    cs1 = [0, 0, 1]
    ret = opt.root(get_dydt(eps), cs1, tol=tol, method=method)
    if log and not ret.success:
        print('get_cs2 did not succeed')
        print(ret)
    return ret

def get_cs3(eps, tol=1e-10, log=False, method='hybr'):
    cs3 = [0, 0, -1]
    ret = opt.root(get_dydt(eps), cs3, tol=tol, method=method)
    if log and not ret.success:
        print('get_cs2 did not succeed')
        print(ret)
    return ret

def get_cs4(eps, tol=1e-10, log=False, method='hybr'):
    cs4 = [-np.sqrt(1 - mu4**2), 0, mu4 + 0.02]
    ret = opt.root(get_dydt(eps), cs4, tol=tol, method=method)
    if log and not ret.success:
        print('get_cs4 did not succeed')
        print(ret)
    return ret

def get_cs(eps):
    sol1 = get_cs1(eps)
    sol3 = get_cs3(eps)

    ang1 = to_ang(*sol1.x)
    ang3 = to_ang(*sol3.x)

    if eps > eps_crit_num:
        return np.array([
            sol1.x[2], 0, sol3.x[2], 0,
            ang1[1], 0, ang3[1], 0,
        ])

    sol2 = get_cs2(eps)
    sol4 = get_cs4(eps)
    ang2 = to_ang(*sol2.x)
    ang4 = to_ang(*sol4.x)

    # print(sol2.x[2], mu2)
    # print(sol4.x[2], mu4)
    return np.array([
        sol1.x[2], sol2.x[2], sol3.x[2], sol4.x[2],
        ang1[1], ang2[1], ang3[1], ang4[1],
    ])

def plot_cs_pts():
    '''
    plot the locations of CS2/4 for increasingly large tides
    '''
    eps_crit = eps_crit_num
    offsets = eps_crit * np.geomspace(0.9, 1e-3, 100)
    eps_vals = np.concatenate((
        eps_crit - offsets + min(offsets),
        (eps_crit + 10 * (offsets - min(offsets)))[1:][::-1],
    ))
    rets = []
    for eps in eps_vals:
        rets.append(get_cs(eps))
    mu1s, mu2s, mu3s, mu4s, phi1s, phi2s, phi3s, phi4s = np.array(rets).T

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(6, 6),
        gridspec_kw={'height_ratios': [1, 1]},
        sharex=True)

    idxs = np.where(eps_vals < eps_crit)
    eps_vals_norm = eps_vals / (eta * np.sin(I))
    mu1ds = np.degrees(np.arccos(mu1s))
    mu2ds = np.degrees(np.arccos(mu2s))
    mu3ds = np.degrees(np.arccos(mu3s))
    mu4ds = np.degrees(np.arccos(mu4s))
    ax1.plot(eps_vals_norm, mu1ds + 70, 'darkorange', alpha=0.8)
    ax1.plot(eps_vals_norm[idxs], mu2ds[idxs], 'tab:green', alpha=0.8)
    ax1.plot(eps_vals_norm, mu3ds - 95, 'tab:blue', alpha=0.8)
    ax1.plot(eps_vals_norm[idxs], mu4ds[idxs], 'tab:purple', alpha=0.8)
    ax1.set_ylabel(r'$\theta_{\rm cs}$ (deg)')
    ax1.text(eps_vals_norm[-1], mu1ds[-1] + 72, 'CS1 ($+70^\circ$)',
             color='darkorange', ha='right', va='top')
    ax1.text(eps_vals_norm[0], mu2ds[0] + 0.1, 'CS2', color='tab:green', va='bottom')
    ax1.text(eps_vals_norm[-1], mu3ds[-1] - 96.5, 'CS3 ($-95^\circ$)',
             color='tab:blue', ha='right')
    ax1.text(eps_vals_norm[0], mu4ds[0] - 0.3, 'CS4', color='tab:purple', va='top')

    ax2.plot(eps_vals_norm, np.degrees(phi1s), 'darkorange', alpha=0.8)
    ax2.plot(eps_vals_norm[idxs], np.degrees(phi2s[idxs]), 'tab:green', alpha=0.5)
    ax2.plot(eps_vals_norm, np.degrees(phi3s), 'tab:blue', alpha=0.8)
    ax2.plot(eps_vals_norm[idxs], np.degrees(phi4s[idxs]), 'tab:purple', alpha=0.5)
    ax2.set_ylabel(r'$\phi_{\rm cs}$ (deg)')
    ax2.set_xlabel(r'$|gt_{\rm al}\sin I|^{-1}$')
    ax2.set_yticks([0, 90, 180])
    ax2.set_yticklabels(['0', '$90$', '$180$'])
    ax2.set_xscale('log')

    # "exact" sols
    phi2_th = np.pi - np.arcsin(eps_vals[idxs] * (1 - mu2**2) / (eta * np.sin(I)))
    phi4_th = np.arcsin(eps_vals[idxs] * (1 - mu4**2) / (eta * np.sin(I)))
    # mu2_th = eta * np.cos(I) / (
    #     1 - eta * np.sin(I) * np.cos(phi2_th) / np.sqrt(1 - mu2**2))
    # mu4_th = eta * np.cos(I) / (
    #     1 - eta * np.sin(I) * np.cos(phi4_th) / np.sqrt(1 - mu4**2))
    # ax1.plot(eps_vals[idxs], mu2_th, 'r:')
    # ax1.plot(eps_vals[idxs], mu4_th, 'b:')
    ax2.plot(eps_vals_norm[idxs], np.degrees(phi2_th), 'g--')
    ax2.plot(eps_vals_norm[idxs], np.degrees(phi4_th), 'r--')

    # ax1.set_title(r'$\eta = %.1f$' % eta)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.06)
    plt.savefig('0_stab', dpi=300)

def get_stab():
    eps_crit_exact = get_eps_crit_num()
    delta_eps = eta**7 * np.sin(I)**3 * np.cos(I)**4 / 2

    def get_eigens(eps_frac):
        '''
        pick unit-vector offsets from CS2 to get matrix elements, then eigens
        eps used is eps_c - eps_frac * delta_eps
        '''
        eps = eps_crit_exact - eps_frac * delta_eps
        cs2 = get_cs2(eps, tol=1e-10, log=True, method='anderson')
        init = cs2.x
        dydt = get_dydt(eps)
        delta = 1e-5
        lin_mat = []
        init_ang = to_ang(*init)
        jac_ang = []

        dydt_init = dydt(init)
        for offset in [delta, 0], [0, delta]:
            # first-order derivative
            offset_ang = init_ang + np.array(offset)
            offset_init = to_cart(*offset_ang)
            diff = dydt(offset_init)

            # too lazy to coordinate change, dydt_ang = to_ang(y0_cart +
            # dydt_cart * delta) - to_ang(y0_cart)
            dydt_ang = to_ang(*(offset_init + diff)) - offset_ang
            lin_mat.append(dydt_ang / delta)
        print(eps_frac, np.linalg.eigvals(lin_mat))

    get_eigens(0.9)
    get_eigens(1.1)

if __name__ == '__main__':
    plot_cs_pts()
    # get_stab()
