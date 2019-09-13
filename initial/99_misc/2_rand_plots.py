'''
more random plots
'''
import numpy as np
import scipy.optimize as opt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

def get_etac(I):
    return (np.sin(I)**(2/3) + np.cos(I)**(2/3))**(-3/2)

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
        return np.cos(np.array(roots))

    else:
        roots = []
        inits = [np.pi / 2 - I, -np.pi + I]
        for qi in inits:
            roots.append(opt.newton(f, qi, fprime=fp))
        return np.cos(np.array(roots))

def plot_cs(I=np.radians(5)):
    etac = get_etac(I)
    eps = 1e-5
    min_eta, max_eta = etac / 30, etac * 30
    etas = np.concatenate((
        np.log(np.linspace(np.exp(min_eta), np.exp(etac - eps), 100)),
        np.exp(np.linspace(np.log(etac + eps), np.log(max_eta), 100)),
    ))
    cs_vals = [[], [], [], []]

    for eta_val in etas:
        root_vals = roots(I, eta_val)
        if eta_val > etac:
            cs_vals[1].append(root_vals[0])
            cs_vals[2].append(root_vals[1])
        else:
            for cs_lst, root_val in zip(cs_vals, root_vals):
                cs_lst.append(root_val)

    etas_four = etas[np.where(etas < etac)[0]]
    plt.semilogx(etas_four, cs_vals[0], 'g', label='1')
    plt.semilogx(etas, cs_vals[1], 'k', label='2')
    plt.semilogx(etas, cs_vals[2], 'b', label='3')
    plt.semilogx(etas_four, cs_vals[3], 'r', label='4')
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\cos\theta$')
    plt.legend()
    plt.xlim([max_eta, min_eta])
    plt.title(r'$I = %d^\circ$' % np.degrees(I))
    plt.savefig('2_cs_locs.png', dpi=400)

if __name__ == '__main__':
    plot_cs(np.radians(20))