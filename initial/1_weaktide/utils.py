import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy import integrate
import scipy.optimize as opt

###################################
### copied from 0_eta/utils.py
###################################

def to_cart(q, phi):
    return [
        -np.sin(q) * np.cos(phi),
        -np.sin(q) * np.sin(phi),
        np.cos(q),
    ]

def to_ang(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    q = np.arccos(z / r)
    phi = (np.arctan2(-y / np.sin(q), -x / np.sin(q)) + 2 * np.pi)\
        % (2 * np.pi)
    return q, phi

def get_etac(I):
    return (np.sin(I)**(2/3) + np.cos(I)**(2/3))**(-3/2)

###################################
### end copy
###################################

def stringify(*args, strf='%.1f'):
    return 'x'.join([strf % arg for arg in args]).replace('.', '_')

def H(I, s_c, s, mu, phi):
    eta = s_c / s
    return -0.5 * (s / s_c) * mu**2 + (
        mu * np.cos(I) -
        np.sqrt(1 - mu**2) * np.sin(I) * np.cos(phi))

def H_max(I, s_reviewers_c, s):
    ''' H_max is always at phi=pi '''
    def minus_H(mu):
        return -H(I, s_c, s, mu, np.pi)
    res = opt.minimize_scalar(minus_H, bounds=(-1, 1), method='bounded')
    return res.x, -res.fun

def roots(I, s_c, s):
    '''
    returns theta roots from EOM, not phi (otherwise same as 0_eta func)
    '''
    eta_c = get_etac(I)
    eta = s_c / s

    # function to minimize and derivatives
    f = lambda q: -eta * np.sin(q - I) + np.sin(q) * np.cos(q)
    fp = lambda q: -eta * np.cos(q - I) + np.cos(2 * q)

    if eta < eta_c:
        roots = []
        inits = [0, np.pi / 2, -np.pi, -np.pi / 2]
        for qi in inits:
            roots.append(opt.newton(f, qi, fprime=fp))
        return np.array(roots)

    else:
        roots = []
        inits = [np.pi / 2 - I, -np.pi + I]
        for qi in inits:
            # newton doesn't seem to work very well here @ large eta
            # roots.append(opt.newton(f, qi, fprime=fp))
            dq = np.pi / 2
            roots.append(opt.bisect(f, qi - dq, qi + dq))
        return np.array(roots)

def solve_ic(I, s_c, eps, y0, tf, method='RK45', rtol=1e-6, **kwargs):
    '''
    wraps solve_ivp and returns sim time
    '''
    dydt = get_dydt_0(I, s_c, eps)
    ret = solve_ivp(dydt, [0, tf], y0, rtol=rtol, method=method, **kwargs)
    return ret.t, ret.y[0:3, :], ret.y[3, :], ret

def find_cs(I, eta, q0):
    f = lambda q: -eta * np.sin(q - I) + np.sin(q) * np.cos(q)
    fp = lambda q: -eta * np.cos(q - I) + np.cos(2 * q)
    return np.cos(opt.newton(f, q0, fprime=fp))

def get_crit(I):
    eta_c = get_etac(I)

    eta = eta_c - 1e-3 # small displacement since no CS4 at eta_c

    # search for CS4
    mu_4 = find_cs(I, eta, -np.pi / 2)
    return eta_c, mu_4

def get_mu4(I, s_c, s):
    '''
    gets mu4 for a list of spins s, returning -1 if no CS4 for given s
    '''
    eta = s_c / s
    eta_c = get_etac(I)
    mu4 = np.full(np.shape(eta), -1.0)

    valid_idxs = np.where(eta < eta_c)

    for idx in zip(*valid_idxs):
        mu4[idx] = find_cs(I, eta[idx], -np.pi/2)

    return mu4

def get_mu2(I, s_c, s):
    '''
    gets mu2 for a list of spins s
    '''
    eta = s_c / s
    mu2 = np.zeros_like(eta)

    for idx, eta_val in enumerate(eta):
        mu2[idx] = find_cs(I, eta_val, +np.pi/2)

    return mu2

def get_H4(I, s_c, s):
    [mu4] = get_mu4(I, s_c, np.array([s]))
    return H(I, s_c, s, mu4, 0)

def get_inf_avg_sol(smax=10):
    '''
    solve averaged equations for IC mu = 0, s >> 1
    '''
    ret = solve_ivp(dmu_ds_nocs, [smax, 1.001], [0],
                    max_step=0.1, dense_output=True)
    s, [mu], interp_sol = ret.t, ret.y, ret.sol
    return mu, s, interp_sol

def get_dydt_0(I, s_c, eps):
    '''
    Full CS equations, no phi treatment. in units where Omega_1 = 1
    '''
    def dydt(t, v):
        '''
        ds/dt, dmu/dt

        try two tricks to keep s close to unit norm: renormalize at start and
        only take normal component at end
        '''
        x, y, z, s = v
        # renormalize, else numerical error accumulates
        # x, y, z = v[ :3] / np.sqrt(np.sum(v[ :3]**2))
        tide = eps * 2 / s * (1 - s * z / 2)
        eta_inv = s / s_c
        ret = [
            eta_inv * y * z - y * np.cos(I) - tide * z * x,
            -eta_inv * x * z + (x * np.cos(I) - z * np.sin(I)) - tide * z * y,
            y * np.sin(I) + tide * (1 - z**2),
            2 * eps * (z - s * (1 + z**2) / 2),
        ]
        # take normal component to s
        # s_hat = np.array([x, y, z])
        # ret[ :3] -= np.dot(ret[ :3], s_hat) * s_hat
        return ret
    return dydt

def dydt_nocs(s, mu):
    '''
    ds/dt, dmu/dt in precession-ignored limit
    '''
    return 2 * mu - s * (1 + mu**2), (1 - mu**2) * (2 / s - mu)

def dmu_ds_nocs(s, y):
    '''
    dmu/ds
    '''
    mu = y[0]
    ds, dmu = dydt_nocs(s, mu)
    return dmu/ds

def get_dydt_num_avg(I, s_c, eps):
    '''
    return ds/dt, dmu/dt getter, but instead do it via a 2pi phi integral of
    dydt_0

    returned function can be used for solve_ivp or for plotting!
    '''
    dydt_0 = get_dydt_0(I, s_c, eps)
    def dydt(s, y):
        # coerce some args into np.arrays
        s = np.reshape(s, np.array(s).size)
        z = np.reshape(y[0], np.array(y[0]).size) # = mu
        mu4 = get_mu4(I, s_c, s)
        x = -np.sqrt(1 - z**2)
        sign = np.sign(mu4 - z) # mu < mu4, dphi > 0

        # a non-terminal event for a negative->positive y-crossing (2pi)
        # (can't exclude the initial condition w/ a terminal event)
        event = lambda t, y: y[1] # set sign in loop

        ds = np.zeros_like(s, dtype=np.float)
        dmu = np.zeros_like(s, dtype=np.float)
        for idx in zip(*np.where(s)): # iterate through s
            event.direction = -sign[idx]
            t_f = 1
            t_events = []
            while len(t_events) == 0:
                # t_f start/end are arbitrary, keep doubling until find event
                ret = solve_ivp(dydt_0, [0, 2 * t_f],
                                # start y != 0 so first event is important
                                [x[idx], -1e-5 * sign[idx], z[idx], s[idx]],
                                events=event, dense_output=True)
                t_events = ret.t_events[0]
                t_f *= 2
            # fetch from first occurrence
            _, _, z_f, s_f = ret.sol(t_events[0])

            # dydt is in epsilon * tau time
            ds[idx] = (s_f - s[idx]) / (eps * t_events[0])
            dmu[idx] = (z_f - z[idx]) / (eps * t_events[0])
        return ds, dmu

    return dydt

def get_dydt_piecewise(I, s_c):
    '''
    returns ds/dt, dmu/dt under piecewise approximation definition
    '''
    def dydt(s, y):
        s = np.array(s)
        mu = np.array(y[0])
        mu4 = get_mu4(I, s_c, s)
        eta = s_c / s
        dist = 2 * np.sqrt(eta * np.sin(I))

        ds = 2 * mu - s * (1 + mu**2)

        # compute piecewise dmu/dt as multiplier of dydt_nocs
        _, dmu = dydt_nocs(s, mu)
        close_above_idx = np.where(np.logical_and(
            mu - mu4 < dist,
            mu - mu4 > 0))
        close_below_idx = np.where(np.logical_and(
            mu4 - mu < dist,
            mu4 - mu > 0))

        _, [dmu_eff_above] = dydt_nocs(s, np.array([mu4 + dist]))
        _, [dmu_eff_below] = dydt_nocs(s, np.array([mu4 - dist]))

        dmu[close_above_idx] = (dmu_eff_above * (dist / (mu - mu4))
                                )[close_above_idx]
        dmu[close_below_idx] = (dmu_eff_below * (dist / (mu4 - mu))
                                )[close_below_idx]
        return ds, dmu
    return dydt

def s_c_str(s_c):
    return ('%.2f' % s_c).replace('.', '_')

def get_mu_equil(s):
    '''
    solve quadratic mu / (1 + mu^2) = s / 2
    assumes s <= 1 everywhere, else will fail

    1 + mu^2 - 2mu / s = 0
    '''
    if s > 1:
        raise ValueError('Cannot get equil mu for s > 1')
    return (2/s - np.sqrt(4 / s**2 - 4)) / 2

def get_crit_mus(I, s_c):
    s_c_crit = get_etac(I) # etac = s_c_crit / (s = 1)
    def dmu_cs2_equil(s):
        mu_cs = np.cos(roots(I, s_c, s))
        mu_equil = get_mu_equil(s)
        if len(mu_cs) == 2:
            return mu_cs[0] - mu_equil
        else:
            return mu_cs[1] - mu_equil
    def dmu_cs1_equil(s):
        # does not check eta < etac!
        mu_cs = np.cos(roots(I, s_c, s))
        mu_equil = get_mu_equil(s)
        return mu_cs[0] - mu_equil
    cs2_equil_mu = get_mu_equil(opt.bisect(dmu_cs2_equil, 0.1, 1))
    # if CS1 just before disappearing is below mu_equil, we won't have an
    # intersection
    s_etac = s_c / (0.9999 * s_c_crit)
    # don't search if won't satisfy s < 1: mu_equil only defined for s < 1
    if s_etac <= 1:
        cs1_crit_mu = np.cos(roots(I, s_c, s_etac)[0])
        mu_equil_etac = get_mu_equil(s_etac)
        if cs1_crit_mu > mu_equil_etac:
            return (
                get_mu_equil(opt.bisect(dmu_cs1_equil, s_etac, 1)),
                cs2_equil_mu,
            )
    return None, cs2_equil_mu

def get_eta_cutoff(I):
    '''
    if the separatrix isn't double valued for phi (i.e. it touches (np.pi, +1)),
    then the calculations of the probabilities is somewhat different; above this
    eta, finding the separatrix to integrate is somewhat tricky. We will ad-hoc
    this cutoff since we only use two I values...
    '''
    if I > np.radians(10):
        return 0.4
    return 0.65
def get_ps_anal(I, s_c, s, *args):
    ''' analytical capture probabilities '''
    eta = s_c / s
    def get_top():
        return s_c / s**2 * (
            -2 * np.cos(I) * (
                2 * np.pi * eta * np.cos(I)
                + (8 * np.sqrt(eta * np.sin(I)))
            )
            + s * np.cos(I) * 2 * np.pi

            + (eta * np.cos(I)) * (
                -8 * np.sqrt(np.sin(I) / eta)
            )
            + (s / 2) * (8 * np.sqrt(np.sin(I) / eta))

            - 4 * np.pi * np.sin(I)
        ) + 2 / s * ( # second term
            -2 * np.pi * (1 - 2 * eta * np.sin(I))
                - (16 * np.cos(I) * eta) * np.sqrt(eta * np.sin(I))
        ) + (
            8 * np.sqrt(eta * np.sin(I))
            + 2 * np.pi * eta * np.cos(I)
            - 64/3 * (eta * np.sin(I))**(3/2)
        )
    def get_bot():
        return s_c / s**2 * (
            -2 * np.cos(I) * (
                -2 * np.pi * eta * np.cos(I)
                + (8 * np.sqrt(eta * np.sin(I)))
            )
            - s * np.cos(I) * 2 * np.pi

            + (eta * np.cos(I)) * (
                -8 * np.sqrt(np.sin(I) / eta)
            )
            + (s / 2) * (8 * np.sqrt(np.sin(I) / eta))

            + 4 * np.pi * np.sin(I)
        ) + 2 / s * ( # second term
            +2 * np.pi * (1 - 2 * eta * np.sin(I))
                - (16 * np.cos(I) * eta) * np.sqrt(eta * np.sin(I))
        ) + (
            8 * np.sqrt(eta * np.sin(I))
            - 2 * np.pi * eta * np.cos(I)
            - 64/3 * (eta * np.sin(I))**(3/2)
        )
    top = get_top()
    bot = get_bot()
    top[np.where(eta > get_eta_cutoff(I))[0]] = -1
    bot[np.where(eta > get_eta_cutoff(I))[0]] = -1
    return top, bot

def get_ps_numinterp(I, s_c, s_arr):
    ''' numerical capture probabilities '''
    eps = 1e-5
    def getter(s):
        ''' gets probability for a single spin s '''
        eta = s_c / s
        if eta > get_eta_cutoff(I):
            return -1, -1
        [mu4] = get_mu4(I, s_c, np.array([s]))

        def mu_up(phi):
            def dH(mu):
                return H(I, s_c, s, mu, phi) - H(I, s_c, s, mu4, 0)
            return opt.brentq(dH, mu4 - eps, 1)
        def mu_down(phi):
            def dH(mu):
                return H(I, s_c, s, mu, phi) - H(I, s_c, s, mu4, 0)
            return opt.brentq(dH, -1, mu4 + eps)

        def arg_top(phi):
            m = mu_up(phi)
            return (
                2 * (m - s * (1 + m**2) / 2) * s_c / (2 * s**2) * (
                    m / eta + np.cos(I))
                + (1 - m**2) * (2 / s - m)
            )
        def arg_bot(phi):
            m = mu_down(phi)
            return (
                2 * (m - s * (1 + m**2) / 2) * s_c / (2 * s**2) * (
                    m / eta + np.cos(I)) +
                (1 - m**2) * (2 / s - m)
            )

        top = -integrate.quad(arg_top, 0, 2 * np.pi)[0]
        bot = integrate.quad(arg_bot, 0, 2 * np.pi)[0]
        return top, bot
    all_probs = np.array([getter(s) for s in s_arr])
    return interp1d(s_arr, all_probs[:, 0]), interp1d(s_arr, all_probs[:, 1])

def get_ps_num(I, s_c, real_crosses, top_interp, bot_interp):
    return top_interp(real_crosses), bot_interp(real_crosses)

def get_anal_caps(I, s_c, cross_dat, mu_vals,
                  getter=get_ps_anal):
    eta_c = get_etac(I)
    s_crosses = cross_dat[:, :, 0]
    p_caps = np.zeros(np.shape(s_crosses), dtype=np.float64)
    # where s_crosses == -2 (no encounter) is already zero, don't set

    cs1_crit_mu, cs2_crit_mu = get_crit_mus(I, s_c)
    min_s_cross = np.abs(s_crosses).min()
    s_interp = np.linspace(min_s_cross, s_crosses.max(), 100)
    top_interp, bot_interp = get_ps_numinterp(I, s_c, s_interp)
    # compute pc for actual crossing spins
    for idx, row in enumerate(s_crosses):
        if s_c == 0.7 and mu_vals[idx] > 0.465:
            # for some reason, the sep crossing is mis-detected in these values?
            p_caps[idx, :] = 0
            continue
        cross_idxs = np.where(row > 0)[0]
        real_crosses = row[cross_idxs]
        top, bot = getter(I, s_c, real_crosses, top_interp, bot_interp)
        d_mu = cross_dat[idx, cross_idxs, 1]

        p_caps[idx, np.where(row == -1)[0]] = 1

        # all the dmus are shared per row (same mu value)
        if len(np.where(d_mu < 0)[0]) > 0:
            p_caps[idx, cross_idxs] = (top + bot) / bot
        else:
            p_caps[idx, cross_idxs] = (top + bot) / top

        # eta cutoff values above sep are zero
        cutoff_expr = np.logical_and(top == -1, bot == -1)
        cutoff_idxs_above = np.where(np.logical_and(
            cutoff_expr,
            d_mu > 0,
        ))[0]
        p_caps[idx, cutoff_idxs_above] = 0

        # finally, if the crossing spin ~ eta_critical, always CS2
        sep_vanish_cross_idx = np.where(
            np.abs(real_crosses - s_c / eta_c) < 1e-2)[0]
        p_caps[idx, sep_vanish_cross_idx] = 1

    p_caps = np.minimum(np.maximum(p_caps, np.zeros_like(p_caps)),
                        np.ones_like(p_caps))
    return p_caps

def get_num_caps(I, s_c, cross_dat, mu_vals):
    return get_anal_caps(I, s_c, cross_dat, mu_vals, getter=get_ps_num)

def get_areas_ward(I, s_c, s):
    eta = s_c / s
    mu4 = eta * np.cos(I) / (1 - eta * np.sin(I))
    q4 = -np.arccos(mu4)

    # WH2004 eq 11-13
    z0 = eta * np.cos(I)
    chi = np.sqrt(-np.tan(q4)**3 / np.tan(I) - 1)
    rho = chi * np.sin(q4)**2 * np.cos(q4) / (
        chi**2 * np.cos(q4)**2 + 1)
    T = 2 * chi * np.cos(q4) / (
        chi**2 * np.cos(q4)**2 - 1)
    A2 = 8 * rho + 4 * np.arctan(T) - 8 * z0 * np.arctan(1 / chi)
    A1 = 2 * np.pi * (1 - z0) - A2 / 2
    A3 = 2 * np.pi * (1 + z0) - A2 / 2
    return A1, A2, A3
