import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize as opt

###################################
### copied from 0_eta/utils.py
###################################

def to_cart(q, phi):
    return [
        np.sin(q) * np.cos(phi),
        np.sin(q) * np.sin(phi),
        np.cos(q),
    ]

def to_ang(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    q = np.arccos(z / r)
    phi = (np.arctan2(y / np.sin(q), x / np.sin(q)) + np.pi)\
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
    return -0.5 * mu**2 + eta * (
        mu * np.cos(I) -
        np.sqrt(1 - mu**2) * np.sin(I) * np.cos(phi))

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
            roots.append(opt.newton(f, qi, fprime=fp))
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
        '''
        x, y, z, s = v
        tide = eps * 2 / s * (1 - s * z / 2)
        rat = s / s_c
        return [
            rat * y * z - y * np.cos(I) - tide * z * x,
            -rat * x * z + (x * np.cos(I) - z * np.sin(I)) - tide * z * y,
            y * np.sin(I) + tide * (1 - z**2),
            2 * eps * (z - s * (1 + z**2) / 2),
        ]
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
