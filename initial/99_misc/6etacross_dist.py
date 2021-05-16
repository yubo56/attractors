'''
try to semi-analytically estimate the distribution of eta @ crossing as a
function of initial conditions, assuming sep crossing occurs when theta = 90
degrees (which is where theta_dot, tide = 0 if spin is large)

Conclusion: due to flatness of theta_dot,tide as theta grows close to 90
degrees, most systems will spin down significantly before separatrix encounter.
Thus, approximate crossing probability by evaluation at eta_sync
'''
from scipy.interpolate import interp1d
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

def integrand(q):
    return -(1 + np.cos(q)**2) / (np.cos(q) * np.sin(q))

def get_integrals(q_vals):
    '''
    get the value of int(integrand, {q, 90, q_val in q_vals}) by interpolating a
    densely-evaluated integral
    '''
    q_vals_dense = np.linspace(np.min(q_vals), np.max(q_vals), 10000)
    int_vals = np.cumsum(integrand(q_vals_dense)) * (
        np.max(q_vals) - np.min(q_vals)) / len(q_vals_dense)
    int_orig = interp1d(q_vals_dense, int_vals)(q_vals)
    return int_orig

def plot_Ws_cross(Wsi=10):
    '''
    \Omega_s normalized to units of n
    '''
    q_vals = np.linspace(np.radians(100), np.radians(175), 100)
    integrals = get_integrals(q_vals)
    # solve 1/Ws,cross - 1 / Wsi = integrals
    plt.plot(np.degrees(q_vals), integrals, 'r')
    plt.xlabel(r'$\theta_{\rm i}$')
    plt.ylabel(r'Integral')
    plt.tight_layout()
    plt.savefig('6etacross_dist', dpi=300)
    plt.close()

if __name__ == '__main__':
    plot_Ws_cross()
