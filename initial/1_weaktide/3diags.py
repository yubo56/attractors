import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from utils import dydt_nocs, get_dydt_num_avg, get_dydt_piecewise, get_mu4
EPS = 1e-3
I = np.radians(20)

def dmu_mu_plot(s_c, s_val):
    '''
    Make plot of \Delta mu(mu0) using the three dydts that we have
    '''
    [mu4] = get_mu4(I, s_c, np.array([s_val]))
    mu = np.linspace(-0.95, 0.95, 200)
    s = np.full_like(mu, fill_value=s_val)

    # get global pictur
    dydt_num_avg = get_dydt_num_avg(I, s_c, EPS)
    dydt_pw = get_dydt_piecewise(I, s_c)

    _, dmu_nocs = dydt_nocs(s, mu)
    _, dmu_num_avg = dydt_num_avg(s, [mu])
    _, dmu_pw = dydt_pw(s, [mu])

    plt.plot(mu, dmu_nocs, label='nocs')
    plt.plot(mu, dmu_num_avg, label='num-avg')
    plt.plot(mu, dmu_pw, label='pw')
    plt.legend()
    plt.xlabel(r'$\mu_0$')
    plt.ylabel(r'$\Delta \mu$')
    plt.xlim([-1, 1])
    plt.ylim([-3, 3])
    plt.savefig('3diags.png')
    plt.clf()

    # loglog above mu4
    # exponential spacing above mu4
    _mu_above = np.linspace(np.log(0.01), np.log(0.98 - mu4), 200)
    mu_above = mu4 + np.exp(_mu_above)
    _, above_num_avg = dydt_num_avg(s, [mu_above])
    _, above_pw = dydt_pw(s, [mu_above])
    plt.loglog(mu_above - mu4, -above_num_avg, label='num-avg')
    plt.loglog(mu_above - mu4, -above_pw, label='pw')
    plt.legend()
    plt.xlabel(r'$\mu_0 - \mu_4$')
    plt.ylabel(r'$-\Delta \mu$')
    plt.savefig('3diags_loglog.png')
    plt.clf()

if __name__ == '__main__':
    s_c = 0.7
    s_val = 3.0
    dmu_mu_plot(s_c, s_val)
