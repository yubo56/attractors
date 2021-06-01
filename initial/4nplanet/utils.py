import numpy as np
import scipy.optimize as opt

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
