import os, pickle, lzma
import scipy.optimize as opt
import time
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

k = 39.4751488
MEARTH = 3e-6
REARTH = 4.7e-5
# length = AU
# unit of time = 499s * 6.32e4 = 1yr
# unit of mass = solar mass, solve for M using N1 + distance in correct units

def roots(I, eta):
    ''' returns theta roots from EOM '''
    eta_c = get_etac(I)

    # function to minimize and derivatives
    f = lambda q: -eta * np.sin(q - I) + np.sin(q) * np.cos(q)
    fp = lambda q: -eta * np.cos(q - I) + np.cos(2 * q)

    if eta < eta_c:
        roots = []
        inits = [0, np.pi / 2, -np.pi, -np.pi / 2]
        for qi in inits:
            roots.append(opt.newton(f, qi, fprime=fp))
        return np.array(roots)
    elif eta > 5: # buggy case, vanishing gradient?
        return np.array([I, np.pi - I])
    else:
        roots = []
        inits = [np.pi / 2 - I, -np.pi + I]
        for qi in inits:
            roots.append(opt.newton(f, qi, fprime=fp))
        return np.array(roots)

def get_etac(I):
    return (np.sin(I)**(2/3) + np.cos(I)**(2/3))**(-3/2)

def reg(z):
    return np.minimum(np.maximum(z, -1), 1)

def ts_dot(x, y):
    ''' dot product of two time series (is there a better way?) '''
    z = np.zeros(np.shape(x)[1])
    for x1, y1 in zip(x, y):
        z += x1 * y1
    return reg(z)

def ts_dot_hat(x, yhat):
    ''' dot product of time series w/ const vec '''
    z = np.zeros(np.shape(x)[1])
    for idx, x1 in enumerate(x):
        z += x1 * yhat[idx]
    return reg(z)

def ts_cross(x, y):
    return np.array([
        x[1] * y[2] - x[2] * y[1],
        -x[0] * y[2] + x[2] * y[0],
        x[0] * y[1] - x[1] * y[0],
    ])

def get_laplace(a, j=1):
    '''
    b_{3/2}^1(a) / 3 * a, determined numerically
    '''
    psi = np.linspace(0, 2 * np.pi, 10000)
    integrand = np.cos(j * psi) / (
        1 - 2 * a * np.cos(psi) + a**2)**(3/2)
    return np.mean(integrand) * 2 / (3 * a)

def get_y0(Id_lst, Wd_lst, qd_lst, fd_lst, Ws_ns):
    '''
    Id_lst - initial orbital inclinations [deg]
    Wd_lst - initial orbital \Omega's [deg]
    q_lst - initial spin orientations [deg] (in inertial frame!)
    f_lst - initial spin precessional phases [deg] (in inertial frame)
    '''
    I_lst = np.radians(Id_lst)
    W_lst = np.radians(Wd_lst)
    q_lst = np.radians(qd_lst)
    f_lst = np.radians(fd_lst)
    return np.concatenate((
        *[
            [np.sin(q) * np.cos(f),
             np.sin(q) * np.sin(f),
             np.cos(q)]
            for q, f in zip(q_lst, f_lst)
        ],
        *[
            [np.sin(I) * np.cos(W),
             np.sin(I) * np.sin(W),
             np.cos(I)]
            for I, W in zip(I_lst, W_lst)
        ],
        Ws_ns,
    ))

def to_array(x, arr):
    if type(x) == np.ndarray:
        return x
    elif type(x) == list:
        return np.array(x)
    else:
        return np.full_like(arr, x)

def dydt(t, y, a_lst=[0.035, 0.05], m_lst=[1, 10], m_star=1,
         kq_ks=1/3, R0s=0, eps=0, tide=0):
    '''
    y = [
        [3 x N]: N svecs,
        [3 x N]: N lvecs,
        [N]: Ws_n (Omega_s / n)
    ]
    a_lst [AU]
    m_lst [MEARTH]
    m_star [MSun]
    kq_k [k_q/k] - if passed scalar, auto-converted into full_like
    R0 [REARTH] - if passed scalar, auto-converted into full_like
    '''
    n_pl = len(a_lst)
    kq_ks = to_array(kq_ks, a_lst)
    R0s = to_array(R0s, a_lst)

    Ws_ns = y[6 * n_pl: ]
    n_lst = np.sqrt(k * m_star / a_lst**3)
    s_vecs = np.reshape(y[ :3 * n_pl], (n_pl, 3))
    l_vecs = np.reshape(y[3 * n_pl:6 * n_pl], (n_pl, 3))
    dydt = np.zeros_like(y)
    for i in range(n_pl):
        # spin precession
        alpha = (
            3/2 * kq_ks[i] / (m_lst[i] * MEARTH)
            * (R0s[i] * REARTH / a_lst[i])**3
            * Ws_ns[i] * n_lst[i]
        )
        dydt[i * 3:(i + 1) * 3] = alpha * (
            np.dot(s_vecs[i], l_vecs[i]) * np.cross(s_vecs[i], l_vecs[i])
        )
        if tide > 0:
            dydt[i * 3:(i + 1) * 3] += tide * (
                np.cross(s_vecs[i], np.cross(l_vecs[i], s_vecs[i])))

        # orbital precession
        for j in range(n_pl):
            if i == j:
                continue
            outer = max(i, j)
            inner = min(i, j)
            coeff = (
                3 * m_lst[outer] * MEARTH / (4 * m_star)
                * (a_lst[inner] / a_lst[outer])**3
                * n_lst[inner]
                * get_laplace(a_lst[inner] / a_lst[outer])
            )
            if i > j:
                coeff *= (
                    m_lst[j] / m_lst[i]
                    * np.sqrt(a_lst[j] / a_lst[i])
                )
            dydt[(n_pl + i) * 3:(n_pl + i + 1) * 3] += coeff * (
                np.dot(l_vecs[i], l_vecs[j]) * np.cross(l_vecs[i], l_vecs[j]))
        # spin evolution
    dydt[6 * n_pl: ] = eps * Ws_ns
    return dydt

def get_eigs(y0, args, toprint=True):
    ''' print out just the etas for the first N - 1 planets assuming
    perturbation is domintated by the Nth planet '''
    a_lst, m_lst, m_star, kq_ks, R0s, _, _ = args
    kq_ks = to_array(kq_ks, a_lst)
    R0s = to_array(R0s, a_lst)

    n_pl = len(a_lst)
    Ws_ns = y0[6 * n_pl: ]
    n_lst = np.sqrt(k * m_star / a_lst**3)
    s_vecs = np.reshape(y0[ :3 * n_pl], (n_pl, 3))
    l_vecs = np.reshape(y0[3 * n_pl:6 * n_pl], (n_pl, 3))
    alphas = []
    for i in range(n_pl - 1):
        # spin precession
        alpha = (
            3/2 * kq_ks[i] / (m_lst[i] * MEARTH)
            * (R0s[i] * REARTH / a_lst[i])**3
            * Ws_ns[i] * n_lst[i]
        )
        if alpha > 0:
            alphas.append(alpha)
            if toprint:
                print('Alpha & Alpha * cos(q) for planet (%d):' % i,
                      alpha,
                      alpha * np.dot(s_vecs[i], l_vecs[i]))
                # print('spin', Ws_ns[i], 'S/L norms', np.sum(s_vecs[i]**2),
                #       np.sum(l_vecs[i]**2))
    B_mat = np.zeros((n_pl, n_pl))
    for i in range(n_pl):
        def get_val(i, j):
            alpha = min(a_lst[i], a_lst[j]) / max(a_lst[i], a_lst[j])
            alpha_bar = min(a_lst[i] /  a_lst[j], 1)
            return (
                3 * m_lst[j] * MEARTH / (4 * m_star)
                * alpha**2 * alpha_bar
                * n_lst[i]
                * get_laplace(alpha)
            )

        # off diagonal terms
        for j in range(n_pl):
            if i != j:
                B_mat[i, j] = -get_val(i, j)
        # diagonal terms
        for j in range(n_pl):
            if j != i:
                B_mat[i, i] += get_val(i, j)
    _eigs = np.abs(np.linalg.eigvals(B_mat))
    eigs = _eigs[np.where(_eigs > 1e-10)[0]]
    if toprint:
        print('Eigs', eigs)
    return eigs, alphas

def test_dydt_jupsat():
    '''
    check that the orbital precession code is correct: compare against M&D
    Jupiter&Saturn
    '''
    n_pl = 2
    y0 = get_y0([1.3, 2.48], [100, 113], [0, 0], [0, 0], [0, 0])
    args = [np.array([5.20, 9.55]), [317, 95], 1]
    start = time.time()
    ret = solve_ivp(dydt, [0, 1e5], y0, method='BDF', atol=1e-10, rtol=1e-10,
                    args=args)
    print('Integration took %f s' % (time.time() - start))
    plt.plot(ret.t, np.degrees(np.arccos(ret.y[8])), label='J')
    plt.plot(ret.t, np.degrees(np.arccos(ret.y[11])), label='S')
    plt.legend()
    plt.savefig('/tmp/foo', dpi=200)
    plt.close()

def get_CS_angles(s_vecs, l_vecs, a_lst, m_lst):
    '''
    from svecs & lvecs, calculate obliquities & precessional phases relative to
    jhat
    '''
    jtot = np.zeros(3)
    for idx, (a, m) in enumerate(zip(a_lst, m_lst)):
        jtot += l_vecs[idx, :, 0] * m * np.sqrt(a)
    jhat = jtot / np.sqrt(np.sum(jtot**2))

    # obliquities are easy to calculate, but precessional phases? we need to
    # define the coordinate system with:
    # zhat = l[i]
    # xhat = jtot - proj(jtot, l), (i.e. unit vector in the j-l plane)
    # yhat = zhat cross xhat
    # then phi = np.arctan2(s_vec . xhat, s_vec . yhat)
    obliquities = []
    phis = []
    for s_vec, l_vec in zip(s_vecs, l_vecs):
        obliquities.append(np.degrees(np.arccos(ts_dot(s_vec, l_vec))))
        phi_arr = []
        for t_idx in range(s_vec.shape[-1]):
            s_vec_now = s_vec[:, t_idx]
            l_vec_now = l_vec[:, t_idx]
            xvec = jhat - np.dot(jhat, l_vec_now) * l_vec_now
            xhat = xvec / np.sqrt(np.sum(xvec**2))
            yhat = np.cross(jhat, xhat)
            phi = np.arctan2(
                np.dot(s_vec_now, xhat),
                np.dot(s_vec_now, yhat))
            phi_arr.append(phi)
        phis.append(phi_arr)
    return obliquities, phis, jhat

def test_2p_cassini_state():
    '''
    Verify that 2-planet CS shows up, with both circulating and librating cases
    for high obliquities (inside/outside separatrix)
    '''
    n_pl = 2
    y0 = get_y0([1, 5], [0, 0], [70, 0], [0, 0], [1, 1])
    args = [np.array([0.4, 5]), [4, 317], 1, [1, 1], [2, 0], 0]
    start = time.time()
    ret = solve_ivp(dydt, [0, 1e6], y0, method='BDF', atol=1e-10, rtol=1e-10,
                    args=args)
    print('Integration took %f s' % (time.time() - start))

    s_vecs = np.reshape(ret.y[ :3 * n_pl], (n_pl, 3, len(ret.t)))
    l_vecs = np.reshape(ret.y[3 * n_pl:6 * n_pl], (n_pl, 3, len(ret.t)))
    obliquities, phis, jhat = get_CS_angles(s_vecs, l_vecs, *args[ :2])
    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1,
        figsize=(8, 10),
        sharex=True)
    ax0.plot(ret.t, np.degrees(np.arccos(ts_dot_hat(l_vecs[0], jhat))))
    ax1.plot(ret.t, obliquities[0])
    ax2.plot(ret.t, np.degrees(np.unwrap(phis[0])))
    ax2.set_xlabel(r'$t$ [yr]')
    ax0.set_ylabel(r'$i_0$')
    ax1.set_ylabel(r'Obliquity')
    ax2.set_ylabel(r'$\phi$')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig('2_2p_lib', dpi=200)
    plt.close()

    y0 = get_y0([1, 5], [0, 0], [30, 0], [0, 0], [1, 1])
    args = [np.array([0.4, 5]), [4, 317], 1, [1, 1], [2, 0], 0]
    start = time.time()
    ret = solve_ivp(dydt, [0, 1e6], y0, method='BDF', atol=1e-10, rtol=1e-10,
                    args=args)
    print('Integration took %f s' % (time.time() - start))

    s_vecs = np.reshape(ret.y[ :3 * n_pl], (n_pl, 3, len(ret.t)))
    l_vecs = np.reshape(ret.y[3 * n_pl:6 * n_pl], (n_pl, 3, len(ret.t)))
    obliquities, phis, jhat = get_CS_angles(s_vecs, l_vecs, *args[ :2])
    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1,
        figsize=(8, 10),
        sharex=True)
    ax0.plot(ret.t, np.degrees(np.arccos(ts_dot_hat(l_vecs[0], jhat))))
    ax1.plot(ret.t, obliquities[0])
    ax2.plot(ret.t, np.degrees(np.unwrap(phis[0])))
    ax2.set_xlabel(r'$t$ [yr]')
    ax0.set_ylabel(r'$i_0$')
    ax1.set_ylabel(r'Obliquity')
    ax2.set_ylabel(r'$\phi$')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig('2_2p_circ', dpi=200)
    plt.close()

def test_3p_cs_mode1():
    '''
    3-planet case, 2 precessional modes. Example of finding libration about
    larger mode
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [85, 0, 0], [0, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1, 0, 0], # Rpl
            0] # dWs_n/dt
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_3p_mode1.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 1e3], y0, method='BDF', atol=1e-10,
                        rtol=1e-10, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)

    s_vecs = np.reshape(ret.y[ :3 * n_pl], (n_pl, 3, len(ret.t)))
    l_vecs = np.reshape(ret.y[3 * n_pl:6 * n_pl], (n_pl, 3, len(ret.t)))
    obliquities, phis, jhat = get_CS_angles(s_vecs, l_vecs, *args[ :2])
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(
        4, 1,
        figsize=(8, 10),
        sharex=True)
    ax0.plot(ret.t, np.degrees(np.arccos(ts_dot_hat(l_vecs[0], jhat))))
    ax1.plot(ret.t, obliquities[0])
    ax2.plot(ret.t, np.degrees(np.unwrap(phis[0])))
    sign = np.sign(np.unwrap(phis[0])[-1] - phis[0][0])
    shifted0 = np.degrees(np.unwrap(phis[0]) - sign * eigs[0] * ret.t)
    ax3.plot(ret.t, shifted0, label=r'$g = %f$' % eigs[0])
    shifted1 = np.degrees(np.unwrap(phis[0]) - sign * eigs[1] * ret.t)
    ax3.plot(ret.t, shifted1, label=r'$g = %f$' % eigs[1])
    ax3.legend()
    min_traj = shifted0 if abs(shifted0[-1]) < abs(shifted1[-1]) else shifted1
    ax3.set_ylim(min_traj.min() - 180, min_traj.max() + 180)
    ax3.set_xlabel(r'$t$ [yr]')
    ax0.set_ylabel(r'$i_0$')
    ax1.set_ylabel(r'Obliquity')
    ax2.set_ylabel(r'$\phi$')
    ax3.set_ylabel(r'$\phi - g_it$')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig('2_3p_mode1', dpi=200)
    plt.close()

def test_3p_cs_mode1_2():
    '''
    3-planet case, 2 precessional modes. Example of finding libration about
    larger mode
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [85, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1, 0, 0], # Rpl
            0] # dWs_n/dt
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_3p_mode1_2.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 1e3], y0, method='BDF', atol=1e-10,
                        rtol=1e-10, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)

    s_vecs = np.reshape(ret.y[ :3 * n_pl], (n_pl, 3, len(ret.t)))
    l_vecs = np.reshape(ret.y[3 * n_pl:6 * n_pl], (n_pl, 3, len(ret.t)))
    obliquities, phis, jhat = get_CS_angles(s_vecs, l_vecs, *args[ :2])
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(
        4, 1,
        figsize=(8, 10),
        sharex=True)
    ax0.plot(ret.t, np.degrees(np.arccos(ts_dot_hat(l_vecs[0], jhat))))
    ax1.plot(ret.t, obliquities[0])
    ax2.plot(ret.t, np.degrees(np.unwrap(phis[0])))
    sign = np.sign(np.unwrap(phis[0])[-1] - phis[0][0])
    shifted0 = np.degrees(np.unwrap(phis[0]) - sign * eigs[0] * ret.t)
    ax3.plot(ret.t, shifted0, label=r'$g = %f$' % eigs[0])
    shifted1 = np.degrees(np.unwrap(phis[0]) - sign * eigs[1] * ret.t)
    ax3.plot(ret.t, shifted1, label=r'$g = %f$' % eigs[1])
    ax3.legend()
    min_traj = shifted0 if abs(shifted0[-1]) < abs(shifted1[-1]) else shifted1
    ax3.set_ylim(min_traj.min() - 180, min_traj.max() + 180)
    ax3.set_xlabel(r'$t$ [yr]')
    ax0.set_ylabel(r'$i_0$')
    ax1.set_ylabel(r'Obliquity')
    ax2.set_ylabel(r'$\phi$')
    ax3.set_ylabel(r'$\phi - g_it$')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig('2_3p_mode1_2', dpi=200)
    plt.close()

def test_3p_cs_mode2():
    '''
    3-planet case, 2 precessional modes. Try to find resonance using smaller
    mode, case 1
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [85, 0, 0], [0, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1/3, 0, 0], # Rpl
            0] # dWs_n/dt
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_3p_mode2.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 3e3], y0, method='BDF', atol=1e-10, rtol=1e-10,
                        args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)

    s_vecs = np.reshape(ret.y[ :3 * n_pl], (n_pl, 3, len(ret.t)))
    l_vecs = np.reshape(ret.y[3 * n_pl:6 * n_pl], (n_pl, 3, len(ret.t)))
    obliquities, phis, jhat = get_CS_angles(s_vecs, l_vecs, *args[ :2])
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(
        4, 1,
        figsize=(8, 10),
        sharex=True)
    ax0.plot(ret.t, np.degrees(np.arccos(ts_dot_hat(l_vecs[0], jhat))))
    ax1.plot(ret.t, obliquities[0])
    ax2.plot(ret.t, np.degrees(np.unwrap(phis[0])))
    sign = np.sign(np.unwrap(phis[0])[-1] - phis[0][0])
    shifted0 = np.degrees(np.unwrap(phis[0]) - sign * eigs[0] * ret.t)
    ax3.plot(ret.t, shifted0, label=r'$g = %f$' % eigs[0])
    shifted1 = np.degrees(np.unwrap(phis[0]) - sign * eigs[1] * ret.t)
    ax3.plot(ret.t, shifted1, label=r'$g = %f$' % eigs[1])
    ax3.legend()
    min_traj = shifted0 if abs(shifted0[-1]) < abs(shifted1[-1]) else shifted1
    ax3.set_ylim(min_traj.min() - 180, min_traj.max() + 180)
    ax3.set_xlabel(r'$t$ [yr]')
    ax0.set_ylabel(r'$i_0$')
    ax1.set_ylabel(r'Obliquity')
    ax2.set_ylabel(r'$\phi$')
    ax3.set_ylabel(r'$\phi - g_it$')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig('2_3p_mode2', dpi=200)
    plt.close()

def test_3p_cs_mode3():
    '''
    3-planet case, 2 precessional modes. Try to find resonance using smaller
    mode, case 2
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [85, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1/3, 0, 0], # Rpl
            0] # dWs_n/dt
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_3p_mode3.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 3e3], y0, method='BDF', atol=1e-10, rtol=1e-10,
                        args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)

    s_vecs = np.reshape(ret.y[ :3 * n_pl], (n_pl, 3, len(ret.t)))
    l_vecs = np.reshape(ret.y[3 * n_pl:6 * n_pl], (n_pl, 3, len(ret.t)))
    obliquities, phis, jhat = get_CS_angles(s_vecs, l_vecs, *args[ :2])
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(
        4, 1,
        figsize=(8, 10),
        sharex=True)
    ax0.plot(ret.t, np.degrees(np.arccos(ts_dot_hat(l_vecs[0], jhat))))
    ax1.plot(ret.t, obliquities[0])
    ax2.plot(ret.t, np.degrees(np.unwrap(phis[0])))
    sign = np.sign(np.unwrap(phis[0])[-1] - phis[0][0])
    shifted0 = np.degrees(np.unwrap(phis[0]) - sign * eigs[0] * ret.t)
    ax3.plot(ret.t, shifted0, label=r'$g = %f$' % eigs[0])
    shifted1 = np.degrees(np.unwrap(phis[0]) - sign * eigs[1] * ret.t)
    ax3.plot(ret.t, shifted1, label=r'$g = %f$' % eigs[1])
    ax3.legend()
    min_traj = shifted0 if abs(shifted0[-1]) < abs(shifted1[-1]) else shifted1
    ax3.set_ylim(min_traj.min() - 180, min_traj.max() + 180)
    ax3.set_xlabel(r'$t$ [yr]')
    ax0.set_ylabel(r'$i_0$')
    ax1.set_ylabel(r'Obliquity')
    ax2.set_ylabel(r'$\phi$')
    ax3.set_ylabel(r'$\phi - g_it$')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig('2_3p_mode3', dpi=200)
    plt.close()

def example_plotter(args, fn, ret):
    n_pl = 3
    etas1 = []
    etas2 = []
    etaidxs = np.linspace(0, len(ret.t) - 1, 100, dtype=np.int64)
    for idx in etaidxs:
        eigs, alphas = get_eigs(ret.y[:, idx], args, toprint=False)
        etas1.append(eigs[0] / alphas[0])
        etas2.append(eigs[1] / alphas[0])

    s_vecs = np.reshape(ret.y[ :3 * n_pl], (n_pl, 3, len(ret.t)))
    l_vecs = np.reshape(ret.y[3 * n_pl:6 * n_pl], (n_pl, 3, len(ret.t)))
    obliquities, phis, jhat = get_CS_angles(s_vecs, l_vecs, *args[ :2])
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(
        5, 1,
        figsize=(8, 10),
        sharex=True)
    ax0.plot(ret.t, np.degrees(np.arccos(ts_dot_hat(l_vecs[0], jhat))))
    ax1.plot(ret.t, obliquities[0])

    ylim = ax1.get_ylim()
    cs2_qs1 = []
    cs2_qs2 = []
    for etav1, etav2 in zip(etas1, etas2):
        css = roots(np.radians(10), etav1)
        cs2_qs1.append(
            css[0] if len(css) == 2 else css[1])
        css = roots(np.radians(10), etav2)
        cs2_qs2.append(
            css[0] if len(css) == 2 else css[1])
    ax1.plot(ret.t[etaidxs], np.degrees(cs2_qs1), c='tab:blue', ls='--')
    ax1.plot(ret.t[etaidxs], np.degrees(cs2_qs2), c='tab:orange', ls='--')
    # etas2 = np.array(etas2)
    # ax1.plot(ret.t[etaidxs], np.degrees(np.arccos(
    #     np.cos(cs2_qs2) + (etas2 * np.cos(np.radians(10)) +
    #                        np.sqrt(2 * etas2 * np.sin(np.radians(10))))
    # )), c='tab:orange', ls=':')
    ax1.set_ylim(ylim)
    ax2.plot(ret.t, np.degrees(np.unwrap(phis[0])))
    # hard code the sign since hard to determine apriori
    shifted0 = np.degrees(np.unwrap(phis[0]) + eigs[0] * ret.t)
    ax3.plot(ret.t, shifted0, label=r'$g = %f$' % eigs[0])
    shifted1 = np.degrees(np.unwrap(phis[0]) + eigs[1] * ret.t)
    ax3.plot(ret.t, shifted1, label=r'$g = %f$' % eigs[1])
    ax3.legend()
    min_traj = shifted0 if abs(shifted0[-1]) < abs(shifted1[-1]) else shifted1
    ax3.set_ylim(min_traj.min() - 180, min_traj.max() + 180)
    ax0.set_ylabel(r'$i_0$')
    ax1.set_ylabel(r'Obliquity')
    ax2.set_ylabel(r'$\phi_{\rm rot}$')
    ax3.set_ylabel(r'$\phi_{\rm rot} - g_it$')

    ax4.plot(ret.t[etaidxs], etas1)
    ax4.plot(ret.t[etaidxs], etas2)
    ax4.set_yscale('log')
    ax4.axhline(get_etac(np.radians(10)), c='k', ls='--')
    ax4.set_ylabel(r'$\eta$')
    ax4.set_xlabel(r'$t$ [yr]')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig(fn, dpi=200)
    plt.close()

def dynamical_example1():
    '''
    Start with the test_mode1 params (seems to be the overlapping resonance
    case), and spin the planet down (increase eta), see which resonance it
    follows. Result: phi_rot - g_slow * t is librating??
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [85, 0, 0], [0, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1, 0, 0], # Rpl
            -1e-3, # dWs_n/dt
            0] # tide
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_dynamical1.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 5e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_dynamical1', ret)

def dynamical_example1_2():
    '''
    Start with the test_mode1_2 params (seems to be completely circulating), and
    spin the planet down (increase eta), see which resonance it follows.
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [85, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1, 0, 0], # Rpl
            -3e-3, # dWs_n/dt
            0] # tide
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_dynamical1_2.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 5e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_dynamical1_2', ret)

def dynamical_example2():
    '''
    Start with the test_mode1_2 params (seems to be completely circulating), and
    spin the planet down (increase eta), see which resonance it follows.
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [65, 0, 0], [0, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/2, # k_q / k
            [1/3, 0, 0], # Rpl
            -1e-3, # dWs_n/dt
            0] # tide
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_dynamical2.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 3e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_dynamical2', ret)

def dynamical_example3():
    '''
    Start with the test_mode1_2 params (seems to be completely circulating), and
    spin the planet down (increase eta), see which resonance it follows.
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [65, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1/2, 0, 0], # Rpl
            -1e-3, # dWs_n/dt
            0] # tide
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_dynamical3.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 3e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_dynamical3', ret)

def spinup_example3():
    '''
    Start with the test_mode1_2 params (seems to be completely circulating), and
    spin the planet down (increase eta), see which resonance it follows.
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [35, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1/3, 0, 0], # Rpl
            1e-3, # dWs_n/dt
            0] # tide
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_spinup3.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 5e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_spinup3', ret)

def spinup_example4():
    '''
    Start with the test_mode1_2 params (seems to be completely circulating), and
    spin the planet down (increase eta), see which resonance it follows.
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [35, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1/4, 0, 0], # Rpl
            1e-3, # dWs_n/dt
            0] # tide
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_spinup4.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 3e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_spinup4', ret)

def dissipative_2p_cs_test():
    '''
    Verify that 2-planet CS shows up, with both circulating and librating cases
    for high obliquities (inside/outside separatrix)
    '''
    n_pl = 2
    y0 = get_y0([1, 5], [0, 0], [70, 0], [0, 0], [1, 1])
    args = [np.array([0.4, 5]), [4, 317], 1, [1, 1], [2, 0], 0, 3e-7]
    start = time.time()
    ret = solve_ivp(dydt, [0, 3e7], y0, method='BDF', atol=1e-10, rtol=1e-10,
                    args=args)
    print('Integration took %f s' % (time.time() - start))

    s_vecs = np.reshape(ret.y[ :3 * n_pl], (n_pl, 3, len(ret.t)))
    l_vecs = np.reshape(ret.y[3 * n_pl:6 * n_pl], (n_pl, 3, len(ret.t)))
    obliquities, phis, jhat = get_CS_angles(s_vecs, l_vecs, *args[ :2])
    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1,
        figsize=(8, 10),
        sharex=True)
    ax0.plot(ret.t, np.degrees(np.arccos(ts_dot_hat(l_vecs[0], jhat))))
    ax1.plot(ret.t, obliquities[0])
    ax2.plot(ret.t, np.degrees(np.unwrap(phis[0])))
    ax2.set_xlabel(r'$t$ [yr]')
    ax0.set_ylabel(r'$i_0$')
    ax1.set_ylabel(r'Obliquity')
    ax2.set_ylabel(r'$\phi$')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig('2_2p_lib_tide', dpi=200)
    plt.close()

    y0 = get_y0([1, 5], [0, 0], [30, 0], [0, 0], [1, 1])
    args = [np.array([0.4, 5]), [4, 317], 1, [1, 1], [2, 0], 0, 3e-7]
    start = time.time()
    ret = solve_ivp(dydt, [0, 1e7], y0, method='BDF', atol=1e-10, rtol=1e-10,
                    args=args)
    print('Integration took %f s' % (time.time() - start))

    s_vecs = np.reshape(ret.y[ :3 * n_pl], (n_pl, 3, len(ret.t)))
    l_vecs = np.reshape(ret.y[3 * n_pl:6 * n_pl], (n_pl, 3, len(ret.t)))
    obliquities, phis, jhat = get_CS_angles(s_vecs, l_vecs, *args[ :2])
    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1,
        figsize=(8, 10),
        sharex=True)
    ax0.plot(ret.t, np.degrees(np.arccos(ts_dot_hat(l_vecs[0], jhat))))
    ax1.plot(ret.t, obliquities[0])
    ax2.plot(ret.t, np.degrees(np.unwrap(phis[0])))
    ax2.set_xlabel(r'$t$ [yr]')
    ax0.set_ylabel(r'$i_0$')
    ax1.set_ylabel(r'Obliquity')
    ax2.set_ylabel(r'$\phi$')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig('2_2p_circ_tide', dpi=200)
    plt.close()

def dissipative_mode1():
    '''
    mode1 params + disp
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [85, 0, 0], [0, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1, 0, 0], # Rpl
            0, # dWs_n/dt
            1e-3] # tidal term
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_dissipative1.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 5e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_dissipative1', ret)

def dissipative_mode1_2():
    '''
    mode1_2 params + disp
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [85, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1, 0, 0], # Rpl
            0, # dWs_n/dt
            1e-3] # tidal term
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_dissipative1_2.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 5e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_dissipative1_2', ret)

def dissipative_mode1_22():
    '''
    mode1_2 params + disp
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [89, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1, 0, 0], # Rpl
            0, # dWs_n/dt
            1e-3] # tidal term
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_dissipative1_22.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 5e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_dissipative1_22', ret)

def dissipative_mode1_23():
    '''
    mode1_2 params + disp
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [90, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1, 0, 0], # Rpl
            0, # dWs_n/dt
            1e-3] # tidal term
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_dissipative1_23.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 5e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_dissipative1_23', ret)

def spinup_disp_example3():
    '''
    Start with the test_mode1_2 params (seems to be completely circulating), and
    spin the planet down (increase eta), see which resonance it follows.
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [35, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1/3, 0, 0], # Rpl
            1e-3,
            1e-3] # dWs_n/dt
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_spinupdisp3.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 5e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_spinupdisp3', ret)

def spinup_disp_example4():
    '''
    Start with the test_mode1_2 params (seems to be completely circulating), and
    spin the planet down (increase eta), see which resonance it follows.
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [35, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1/4, 0, 0], # Rpl
            1e-3,
            1e-3] # dWs_n/dt
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_spinupdisp4.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 5e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_spinupdisp4', ret)

def dynamicaltide_example1():
    '''
    Start with the test_mode1 params (seems to be the overlapping resonance
    case), and spin the planet down (increase eta), see which resonance it
    follows. Result: phi_rot - g_slow * t is librating??
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [85, 0, 0], [0, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1, 0, 0], # Rpl
            -1e-3, # dWs_n/dt
            1e-3] # tide
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_dynamicaltide1.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 5e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_dynamicaltide1', ret)

def dynamicaltide_example1_2():
    '''
    Start with the test_mode1_2 params (seems to be completely circulating), and
    spin the planet down (increase eta), see which resonance it follows.
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [85, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1, 0, 0], # Rpl
            -3e-3, # dWs_n/dt
            1e-3] # tide
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_dynamicaltide1_2.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 5e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_dynamicaltide1_2', ret)

def dynamicaltide_example2():
    '''
    Start with the test_mode1_2 params (seems to be completely circulating), and
    spin the planet down (increase eta), see which resonance it follows.
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [65, 0, 0], [0, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/2, # k_q / k
            [1/3, 0, 0], # Rpl
            -1e-3, # dWs_n/dt
            1e-3] # tide
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_dynamicaltide2.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 3e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_dynamicaltide2', ret)

def dynamicaltide_example3():
    '''
    Start with the test_mode1_2 params (seems to be completely circulating), and
    spin the planet down (increase eta), see which resonance it follows.
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [65, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1/2, 0, 0], # Rpl
            -1e-3, # dWs_n/dt
            1e-3] # tide
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_dynamicaltide3.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 3e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_dynamicaltide3', ret)

def no_spinup_example3():
    '''
    Start with the test_mode1_2 params (seems to be completely circulating), and
    spin the planet down (increase eta), see which resonance it follows.
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [35, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1/3, 0, 0], # Rpl
            0, # dWs_n/dt
            0] # tide
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_no_spinup3.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 5e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_no_spinup3', ret)

def no_spinup_example4():
    '''
    Start with the test_mode1_2 params (seems to be completely circulating), and
    spin the planet down (increase eta), see which resonance it follows.
    '''
    n_pl = 3
    y0 = get_y0([1, 5, 5], [0, 180, 0], [35, 0, 0], [180, 0, 0], [1, 1, 1])
    args = [0.035 * np.array([1, 1.3, 5.0]), # apl
            [1, 10, 317], # mpl
            1, # mstar
            1/3, # k_q / k
            [1/4, 0, 0], # Rpl
            0, # dWs_n/dt
            0] # tide
    eigs, _ = get_eigs(y0, args)
    pkl_fn = '2_no_spinup4.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        ret = solve_ivp(dydt, [0, 3e3], y0, method='DOP853', atol=1e-9,
                        rtol=1e-9, args=args)
        print('Integration took %f s' % (time.time() - start))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    example_plotter(args, '2_no_spinup4', ret)

if __name__ == '__main__':
    # test_dydt_jupsat()
    # test_2p_cassini_state()
    # test_3p_cs_mode1()
    # test_3p_cs_mode1_2()
    # test_3p_cs_mode2()
    # test_3p_cs_mode3()
    # dissipative_2p_cs_test()

    dynamical_example1()
    dynamical_example1_2()
    dynamical_example2()
    dynamical_example3()
    spinup_example3()
    spinup_example4()
    no_spinup_example3()
    no_spinup_example4()

    dissipative_mode1()
    dissipative_mode1_2()
    dissipative_mode1_22()
    dissipative_mode1_23()

    spinup_disp_example3()
    spinup_disp_example4()
    dynamicaltide_example1()
    dynamicaltide_example1_2()
    dynamicaltide_example2()
    dynamicaltide_example3()
    pass
