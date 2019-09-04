'''
look for existence/stability of CS2 as tidal dissipation increases
'''
import numpy as np
import scipy.optimize as opt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

I = np.radians(20)
eta = 0.1
mu2 = eta * np.cos(I) / (1 + eta * np.sin(I))
mu4 = eta * np.cos(I) / (1 - eta * np.sin(I))
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

def get_cs2(eps):
    cs2 = [np.sqrt(1 - mu2**2), 0, mu2]
    return opt.root(get_dydt(eps), cs2, jac=get_jac(eps))

def get_cs4(eps):
    cs4 = [-np.sqrt(1 - mu4**2), 0, mu4]
    return opt.root(get_dydt(eps), cs4, jac=get_jac(eps))

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

def get_cs(eps):
    sol2 = get_cs2(eps)
    sol4 = get_cs4(eps)

    ang2 = to_ang(*sol2.x)
    ang4 = to_ang(*sol4.x)

    # print(sol2.x[2], mu2)
    # print(sol4.x[2], mu4)
    return np.array([sol2.x[2], sol4.x[2], ang2[1], ang4[1]])

def plot_cs_pts():
    '''
    plot the locations of CS2/4 for increasingly large tides
    '''
    offsets = 0.9 * eps_crit * np.exp(np.linspace(0, -5, 60))
    eps_vals = eps_crit - offsets + min(offsets)
    rets = []
    for eps in eps_vals:
        rets.append(get_cs(eps))
    mu2s, mu4s, phi2s, phi4s = np.array(rets).T

    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=(6, 9),
                                   sharex=True)
    fig.subplots_adjust(hspace=0)

    ax1.plot(eps_vals, mu2s, 'r', label=r'$\mu_2$')
    ax1.plot(eps_vals, mu4s, 'b', label=r'$\mu_4$')
    mu2_th = mu2 + eta**2 * np.sin(I) * np.cos(I) / (1 + eta * np.sin(I)) * \
        (eps_vals / (eta * np.sin(I)))**2 / 2
    mu4_th = mu4 - eta**2 * np.sin(I) * np.cos(I) / (1 - eta * np.sin(I)) * \
        (eps_vals / (eta * np.sin(I)))**2 / 2
    ax1.plot(eps_vals, np.ones_like(eps_vals) * mu2_th, 'r:')
    ax1.plot(eps_vals, np.ones_like(eps_vals) * mu4_th, 'b:')
    ax1.set_ylabel(r'$\mu$')
    ax1.legend()

    phi2_th = np.pi - eps_vals / (eta * np.sin(I))
    phi4_th = eps_vals / (eta * np.sin(I))
    ax2.plot(eps_vals, phi2s, 'r', label=r'$\phi_2$')
    ax2.plot(eps_vals, phi4s, 'b', label=r'$\phi_4$')
    ax2.plot(eps_vals, np.ones_like(eps_vals) * phi2_th, 'r:')
    ax2.plot(eps_vals, np.ones_like(eps_vals) * phi4_th, 'b:')
    ax2.set_ylabel(r'$\phi$')
    ax2.legend()

    for ax in [ax1, ax2]:
        ax.axvline(eps_crit, color='m')
        ax.axvline(get_eps_crit_num(), color='k')
    ax1.set_title(r'$\eta = %.1f$' % eta)
    plt.savefig('0_stab.png')

def get_stab():
    eps_crit_exact = get_eps_crit_num()
    delta_eps = eta**7 * np.sin(I)**3 * np.cos(I)**4 / 2

    eps_test = eps_crit_exact - eps_crit_exact * 100
    dydt = get_dydt(eps_test)

    def get_eigens(init):
        '''
        pick unit-vector offsets from CS2 to get matrix elements, then eigens
        '''
        delta = 1e-6
        lin_mat = []
        init_ang = to_ang(*init)
        jac_ang = []

        dydt_init = dydt(init)
        for offset in [delta, 0], [0, delta]:
            # first-order derivative
            offset_ang = init_ang + np.array(offset)
            offset_init = to_cart(*offset_ang)
            diff = dydt(offset_init) * delta

            # d_ang/dt = (to_ang(y + dydt * dt) - to_ang(y)) / dt
            d_ang = to_ang(*(offset_init + diff)) - to_ang(*offset_init)
            lin_mat.append(d_ang / delta)
        print(lin_mat)
        print(np.linalg.eigvals(lin_mat))

    cs2 = get_cs2(eps_test)
    get_eigens(cs2.x)

if __name__ == '__main__':
    # plot_cs_pts()
    get_stab()
