'''
utils functions for toy problem 2

since we're only concerned w/ separatrix hopping cases, we will ignore ICs
inside the separatrix or that start too close to CS1/CS3
'''
import numpy as np
import scipy.optimize as opt
from scipy.integrate import solve_ivp

########
# COPIED
########

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

def roots(I, eta):
    '''
    returns theta roots from EOM, not phi (otherwise same as 0_eta func)
    '''
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

    else:
        roots = []
        inits = [np.pi / 2 - I, -np.pi + I]
        for qi in inits:
            # newton doesn't seem to work very well here...
            # roots.append(opt.newton(f, qi, fprime=fp))
            roots.append(opt.bisect(f, 0, 2 * (np.pi / 2 - I)))
        return np.array(roots)

def H(I, eta, q, phi):
    mu = np.cos(q)
    return -0.5 * mu**2 + eta * (
        mu * np.cos(I) -
        np.sqrt(1 - mu**2) * np.sin(I) * np.cos(phi))

def get_mu4(I, eta):
    '''
    gets mu4 for a list of spins s, returning -1 if no CS4 for given s
    '''
    eta_c = get_etac(I)
    mu4 = np.full(np.shape(eta), -1.0)

    valid_idxs = np.where(eta < eta_c)

    for idx in zip(*valid_idxs):
        mu4[idx] = np.cos(roots(I, eta[idx])[3])

    return mu4

############
# END COPIED
############

def get_areas(ret):
    ''''
    compute enclosed phase space area at time intervals. passed sol
    interpolates (x, y, z, eta)

    assume solutions only librate about CS2, and never begin @ libration
    (assumptions detailed at top of file). then at all times, either t_0/t_pi
    interlocking or only t_pi exists
    - circulating: area is sum(-mu * dphi) between successive t_0 points
    - librating: area is sum(-mu * dphi) between over *three* t_pi points
    '''
    num_pts = 100

    [t_events] = ret.t_events
    x_events, _, z_events, eta_events = ret.sol(t_events)
    # phi = 0 means x < 0
    idx_0 = np.where(x_events < 0)[0]
    idx_pi = np.where(x_events > 0)[0]
    t_0 = t_events[idx_0]
    t_pi = t_events[idx_pi]

    t_cross = np.inf
    ends_circ = False

    t_areas = []
    areas = []

    # NB: t_pi is always longer than t_0
    for t_0_i, t_0_f, t_pi_i, t_pi_f in zip(t_0[ :-1], t_0[1: ],
                                            t_pi, t_pi[1: ]):
        # all t_0 points correspond to circulation
        t_vals = np.linspace(t_0_i, t_0_f, num_pts)
        sol_vals = np.array(ret.sol(t_vals)[ : 3]).T # index by time then var

        t_areas.append((t_0_i + t_0_f) / 2)
        # TODO maybe better integral?
        area = 0
        angs = [to_ang(*sol_val) for sol_val in sol_vals]
        for (q0, phi0), (q1, phi1) in zip(angs[ :-1], angs[1: ]):
            dphi = phi1 - phi0
            if abs(dphi) > np.pi:
                # can get wraparound effects, do this dumb way
                dphi += -np.sign(dphi) * 2 * np.pi
            area += -(np.cos(q1) + np.cos(q0)) / 2 * dphi

        areas.append(area)

        # detect escape
        sol_at_i_f = ret.sol(np.array([t_0_i, t_0_f, t_pi_i, t_pi_f]))
        mu_0_i, mu_0_f, mu_pi_i, mu_pi_f = sol_at_i_f[2]
        if np.sign(mu_pi_f - mu_0_f) != np.sign(mu_pi_i - mu_0_i):
            ends_circ = True
            t_cross = (t_pi_f + t_pi_i) / 2
            # debug how much adiabaticity is violated during crossing
            # print('Etas during escape', sol_at_i_f[3])

    _lib_ts = t_pi[np.where(t_pi > t_0[-1])[0]]
    if len(_lib_ts) > 1:
        t_cross = t_0[-1]

        lib_ts = _lib_ts[ ::2] # only every other time
        for t_0_i, t_0_f in zip(lib_ts[ :-1], lib_ts[1: ]):
            t_vals = np.linspace(t_0_i, t_0_f, num_pts)
            sol_vals = np.array(ret.sol(t_vals)[ : 3]).T

            t_areas.append((t_0_i + t_0_f) / 2)
            # TODO also better integral
            area = 0
            angs = [to_ang(*sol_val) for sol_val in sol_vals]
            for (q0, phi0), (q1, phi1) in zip(angs[ :-1], angs[1: ]):
                dphi = phi1 - phi0
                if abs(dphi) > np.pi:
                    # can get wraparound effects, do this dumb way
                    dphi += -np.sign(dphi) * 2 * np.pi
                area += -(np.cos(q1) + np.cos(q0)) / 2 * dphi

            areas.append(area)

    return t_areas, np.array(areas), t_cross, ends_circ

def solve_ic(I, eps, y0, tf, rtol=1e-6, **kwargs):
    '''
    get dy/dt for (x, y, z, eta)

    returns (t, y, sep areas, ret)
    '''
    def dydt(t, s):
        x, y, z, eta = s
        return [
            y * z - eta * y * np.cos(I),
            -x * z + eta * (x * np.cos(I) - z * np.sin(I)),
            eta * y * np.sin(I),
            eps * eta * z,
        ]
    event = lambda t, y: y[1]
    ret = solve_ivp(dydt, [0, tf], y0,
                    rtol=rtol, dense_output=True, events=[event],
                    **kwargs)

    return ret

def solve_ic_base(I, eps, y0, tf, rtol=1e-6, **kwargs):
    '''
    get dy/dt for (x, y, z, eta)

    returns (t, y, sep areas, ret)
    '''
    def dydt(t, s):
        x, y, z, eta = s
        return [
            y * z - eta * y * np.cos(I),
            -x * z + eta * (x * np.cos(I) - z * np.sin(I)),
            eta * y * np.sin(I),
            eps * eta,
        ]
    event = lambda t, y: y[1]
    ret = solve_ivp(dydt, [0, tf], y0,
                    rtol=rtol, dense_output=True, events=[event],
                    **kwargs)

    return ret

def get_plot_coords(q, phi):
    return np.sin(q / 2) * np.cos(phi), np.sin(q / 2) * np.sin(phi)
