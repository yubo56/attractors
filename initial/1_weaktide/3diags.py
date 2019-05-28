import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from utils import dydt_nocs, get_dydt_num_avg, get_dydt_piecewise, get_mu4,\
    stringify
EPS = 1e-3
I = np.radians(20)

def dmu_mu_plot(s_c, s_val):
    '''
    Make plot of \Delta mu(mu0) using the three dydts that we have
    '''
    [mu4] = get_mu4(I, s_c, np.array([s_val]))
    mu = np.linspace(-0.95, 0.95, 200)
    s = np.full_like(mu, fill_value=s_val)

    # get global picture
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
    plt.ylabel(r'$\Delta \mu / \Delta t$')
    plt.xlim([-1, 1])
    plt.ylim([-3, 3])
    plt.title(r'$(s, s_c) = (%.1f, %.1f)$' % (s_val, s_c))
    plt.savefig('3diags%s.png' % stringify(s_c))
    plt.close()

    # loglog above mu4
    # exponential spacing above mu4
    _mu_above = np.linspace(np.log(0.01), np.log(0.98 - mu4), 200)
    mu_above = mu4 + np.exp(_mu_above)
    _, above_num_avg = dydt_num_avg(s, [mu_above])
    _, above_pw = dydt_pw(s, [mu_above])
    plt.loglog(mu_above - mu4, -above_num_avg, 'ro',
               label='num-avg', markersize=2)
    plt.loglog(mu_above - mu4, -above_pw, 'bo',
               label='pw', markersize=2)
    plt.legend()
    plt.xlabel(r'$\mu_0 - \mu_4$')
    plt.ylabel(r'$-\Delta \mu/\Delta t$')
    plt.savefig('3diags_loglog%s.png' % stringify(s_c))
    plt.close()

def plot_crits():
    '''
    find critical s_c as function of s where dmu/dt above CS4 changes from
    negative to positive
    '''
    s = np.linspace(1.0, 10.0, 91)
    crits = []
    for s_val in s:
        print(s_val)
        def val_above(s_c):
            [mu4] = get_mu4(I, s_c, np.array([s_val]))
            dydt_num_avg = get_dydt_num_avg(I, s_c, EPS)
            _, dmu_above = dydt_num_avg(s_val, [mu4 + 0.03])
            return dmu_above
        s_c_interval = [0.1, s_val]
        root = brentq(val_above, *s_c_interval)
        crits.append(root)
    plt.plot(s, crits)
    plt.xlabel(r'$s$')
    plt.ylabel(r'Critical $s_c$')
    plt.savefig('3crits.png')
    plt.close()

if __name__ == '__main__':
    # generate dmu/dt
    # s_val = 3.0
    # for s_c in [0.1, 0.3, 0.5, 0.7, 1.0]:
    #     dmu_mu_plot(s_c, s_val)

    plot_crits()
