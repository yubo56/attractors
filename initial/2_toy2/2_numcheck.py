'''
quick script, just need to store somewhere

Numerically validate the integrals for Phop in toy problem 2, using the known
eta_star values
'''
import numpy as np
from scipy import integrate

I = np.radians(20)
eta = 0.029

def mu_up(phi):
    return eta * np.cos(I) + np.sqrt(2 * eta * np.sin(I) * (1 - np.cos(phi)))

def mu_down(phi):
    return eta * np.cos(I) - np.sqrt(2 * eta * np.sin(I) * (1 - np.cos(phi)))

def arg(phi):
    m = mu_up(phi)
    return m * (m / eta + np.cos(I))

def arg_bot(phi):
    m = mu_down(phi)
    return m * (m / eta + np.cos(I))

if __name__ == '__main__':
    top = integrate.quad(arg, 0, 2 * np.pi)[0]
    bot = integrate.quad(arg_bot, 0, 2 * np.pi)[0]
    tot = top - bot
    top_anal = (4 * np.pi * np.sin(I) +
        4 * np.pi * eta * np.cos(I)**2 +
          24 * np.cos(I) * np.sqrt(eta * np.sin(I)))
    tot_anal = (48 * np.cos(I) * np.sqrt(eta * np.sin(I)))
    print(top_anal, top)
    print(tot_anal, tot)
