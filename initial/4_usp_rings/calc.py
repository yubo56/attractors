import os, pickle, lzma
from scipy.fft import fft, fftfreq
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

MEARTH = 3e-6
REARTH = 4.7e-5

def generate_txt(mps=[1, 10], p_rat=1.5, Is=[1, 5], prefix='', ain=0.035,
                 **kwargs):
    with open('rings_usp/in%s.txt' % prefix, 'w') as f:
        sun_line = [1e9, 0, 1, 0, 0, 0, 0] # tV k I R Spin0 Spin1 Spin2
        f.write(' '.join(['%e' % l for l in sun_line]))
        f.write('\n')
        for idx, (mp, I) in enumerate(zip(mps, Is)):
            planet_dat = [
                mp * MEARTH, # m
                ain * (p_rat)**(idx * 2/3), # a
                1e-3, # e
                I, # I
                0, 0, 1e9, 0, 0, 0, # Omega omega tV k I R
                0, 0, 0 # spin
            ]
            f.write(' '.join(['%e' % l for l in planet_dat]))
            f.write('\n')

def get_laplace(a, j=1):
    '''
    b_{3/2}^1(a) / 3 * a, determined numerically
    '''
    psi = np.linspace(0, 2 * np.pi, 10000)
    integrand = np.cos(j * psi) / (
        1 - 2 * a * np.cos(psi) + a**2)**(3/2)
    return np.mean(integrand) * 2 / (3 * a)

def run_txt(t='1e6', prefix='', tol='1e-10', **kwargs):
    cmd = (
        'rings --input rings_usp/in{p}.txt --output rings_usp/out{p}.txt --time {t} '
        '--epsint {tol}'
        .format(p=prefix, t=t, tol=tol)
    )
    print(cmd)
    os.system(cmd)

def get_peaks(powers, npeaks=1):
    p_idxs = np.argsort(powers)[::-1]
    peaks = [] # actual peaks
    idx = 0
    while len(peaks) < npeaks:
        for other_ps in p_idxs[ :idx]:
            idx_rat = p_idxs[idx] / other_ps
            if np.abs(idx_rat - round(idx_rat)) < 0.02:
                break
        else:
            peaks.append(p_idxs[idx])
        idx += 1
    return np.array(peaks)

def run_for_prefix(plot=True, prefix='', **kwargs):
    if not os.path.exists('rings_usp/out%s.txt' % prefix):
        print('Running for out%s' % prefix)
        generate_txt(prefix=prefix, **kwargs)
        run_txt(prefix=prefix, **kwargs)
    else:
        print('Found out%s' % prefix)
    nbody = 3 if 'mps' not in kwargs else len(kwargs['mps']) + 1
    out = np.loadtxt('rings_usp/out%s.txt' % prefix)
    orig_t = out[1::nbody, 0] / (2 * np.pi)
    orig_I = out[1::nbody, 4]
    orig_I2 = out[2::nbody, 4]
    n = 1 / out[1, 2]**(3/2)

    t = np.linspace(orig_t[0], orig_t[-1], len(orig_t) * 10)
    I = interp1d(orig_t, orig_I)(t)
    I2 = interp1d(orig_t, orig_I2)(t)

    Ifft = fft(I - np.mean(I))
    N = len(I)
    dt = np.median(np.diff(t))
    Ifft_freqs = fftfreq(N, dt)[1:N//2]

    ifft_part = np.abs(Ifft[1:N//2])
    max_freq = Ifft_freqs[np.argmax(ifft_part) + 1]
    xlim = np.max(Ifft_freqs)
    basic_freqs = get_prec_basic(**kwargs) / (2 * np.pi)
    LL_freqs = get_prec_LL(**kwargs) / (2 * np.pi)
    peak_freqs = Ifft_freqs[get_peaks(ifft_part, nbody - 2)]
    print('Simple/LL/alg period',
          1 / basic_freqs, 1 / LL_freqs, 1 / peak_freqs)
    if plot:
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(8, 8))
        ax1.plot(orig_t, orig_I, 'b', alpha=0.5)
        ax1.plot(orig_t, orig_I2, 'g', alpha=0.5)
        ax1.set_xlabel(r'$T$ (yr)')
        ax1.set_ylabel(r'$I$')

        ax2.semilogy(Ifft_freqs, 2 / N * ifft_part, 'ko')
        ax2.set_xlabel(r'Freq [1 / period] (yr$^{-1}$)')
        ax2.set_ylabel(r'Amplitude')
        ax2.set_xlim(left=0, right=xlim / 50)
        for f in basic_freqs:
            ax2.axvline(f, c='r', alpha=0.5)
        for f in LL_freqs:
            ax2.axvline(f, c='b', alpha=0.5)
        for f in peak_freqs:
            ax2.axvline(f, c='k', alpha=0.5)


        plt.tight_layout()
        plt.savefig('rings_usp/out%s' % prefix, dpi=300)
        plt.close()
    return basic_freqs, LL_freqs, peak_freqs

def get_prec_basic(p_rat=1.5, ain=0.035, mps=[1, 10], Is=[1, 5], **kwargs):
    '''
    just first two precession freqs for now...
    '''
    n = 2 * np.pi / ain**(3/2)
    w_pl = 3/4 * mps[1] * MEARTH * (p_rat)**(-2) * n
    f_alpha = get_laplace(p_rat**(-2/3))
    # assumes initial alignment, close enough
    a_rats = p_rat**(2/3 * np.arange(len(mps)))
    l_rats = mps * np.sqrt(a_rats)
    angmom_rat = (l_rats[0] + l_rats[1]) / l_rats[1]
    prec_in = angmom_rat * w_pl * np.cos(np.radians(Is[1] - Is[0])) * f_alpha
    if len(mps) == 2:
        return np.array([prec_in])

    w_pl31 = 3/4 * mps[2] * MEARTH * (p_rat)**(-4) * n
    f_alpha31 = get_laplace(p_rat**(-4/3))
    # assumes initial alignment, close enough
    angmom_rat31 = (l_rats[0] + l_rats[2]) / l_rats[2]
    prec_in31 = angmom_rat31 * w_pl31 * np.cos(np.radians(Is[2] - Is[0])) * f_alpha31

    n2 = 2 * np.pi / ain**(3/2) / p_rat
    w_pl32 = 3/4 * mps[2] * MEARTH * (p_rat)**(-2) * n2
    # assumes initial alignment, close enough
    angmom_rat32 = (l_rats[1] + l_rats[2]) / l_rats[2]
    prec_out32 = angmom_rat32 * w_pl32 * np.cos(np.radians(Is[2] - Is[1])) * f_alpha
    return np.array([prec_in + prec_in31, prec_out32])
    # return np.array([prec_in, prec_out32])

def get_prec_LL(p_rat=1.5, ain=0.035, mps=[1, 10], Is=[1, 5], **kwargs):
    '''
    ignore J2 term for now
    '''
    N = len(mps)
    B_mat = np.zeros((N, N))
    for j in range(N):
        n_j = 2 * np.pi / ain**(3/2) / p_rat**j
        def get_val(j, k):
            alpha = p_rat**(-2 * np.abs(k - j) / 3)
            alpha_bar = min(p_rat**(-2 * (k - j) / 3), 1)
            return (
                3/4 * n_j * mps[k] * MEARTH * alpha**2 * alpha_bar
                    * get_laplace(alpha)
            )

        # off diagonal terms
        for k in range(N):
            if j != k:
                B_mat[j, k] = -get_val(j, k)
        # diagonal terms
        for k in range(N):
            if k != j:
                B_mat[j, j] += get_val(j, k)
    eigs = np.abs(np.linalg.eigvals(B_mat))
    return eigs[np.where(eigs > 1e-10)[0]]

def scan_alpha(I_in=1, I_out=5, m_in=1, m_out=10, fn='scan_alpha', name='scan'):
    sma_ratios = np.linspace(1.2, 2, 20)
    pkl_fn = '%s.pkl' % fn
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        basics2 = []
        LLs2 = []
        numericals2 = []
        basics3 = []
        LLs3 = []
        numericals3 = []
        for idx, alpha in enumerate(sma_ratios):
            basic_freqs, LL_freqs, peak_freqs = run_for_prefix(
                prefix='_%s2p_%d' % (name, idx),
                p_rat=alpha**(3/2),
                Is=[I_in, I_out],
                mps=[m_in, m_out],
                ain=0.035,
                t='7e5')
            basics2.append(basic_freqs)
            LLs2.append(LL_freqs)
            numericals2.append(peak_freqs)

            basic_freqs, LL_freqs, peak_freqs = run_for_prefix(
                prefix='_%s3p_%d' % (name, idx),
                p_rat=alpha**(3/2),
                Is=[I_in, I_out, I_out],
                mps=[m_in, m_out, m_out],
                ain=0.035,
                t='7e5')
            basics3.append(basic_freqs)
            LLs3.append(LL_freqs)
            numericals3.append(peak_freqs)
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump((basics2, LLs2, numericals2, basics3, LLs3, numericals3), f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            basics2, LLs2, numericals2, basics3, LLs3, numericals3 = pickle.load(f)
    basics2 = np.array(basics2)
    basics3 = np.array(basics3)
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(8, 10),
        sharex=True)
    ax1.semilogy(sma_ratios, 2 * np.pi * np.array(numericals2), 'g', alpha=0.5, label='Num')
    ax1.semilogy(sma_ratios, 2 * np.pi * np.array(LLs2), 'b', alpha=0.5, label='LL')
    ax1.semilogy(sma_ratios, 2 * np.pi * basics2, 'r', alpha=0.5, label='Basic')
    ax1.legend(loc='upper right')

    for idx, (op, ls, label) in enumerate(zip(
        [np.maximum, np.minimum],
        ['', '--'],
        [r'$\Omega_{32, 1}$', r'$\Omega_{32}$'],
        # [r'$\Omega_{21}$', r'$\Omega_{32}$'],
    )):
        ax2.semilogy(sma_ratios, 2 * np.pi * np.array(basics3).T[idx], 'r%s' % ls,
                     alpha=0.5,
                     label=label)
        ax2.semilogy(sma_ratios, 2 * np.pi * op(*np.array(numericals3).T), 'g%s' % ls, alpha=0.5)
        ax2.semilogy(sma_ratios, 2 * np.pi * op(*np.array(LLs3).T), 'b%s' % ls, alpha=0.5)
    ax2.legend(loc='upper right')
    ax1.set_ylabel(r'Precession Freq [yr$^{-1}$; 2p]')
    ax2.set_ylabel(r'Precession Freq [yr$^{-1}$; 3p]')

    ax3.plot(sma_ratios,
             (np.array(basics3).T[0] /
                np.maximum(*np.array(numericals3).T)),
             'k%s' % ls)
    ax3.set_ylabel(r'$(\Omega_{32, 1}) / \mathrm{Num}$')
    ax3.set_xlabel(r'$a_{\rm 2} / a_{\rm 1}$')

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig(fn, dpi=300)
    plt.close()

if __name__ == '__main__':
    # print(get_laplace(0.544493))
    # print(get_laplace(0.035 / 0.0459))
    # print(get_laplace(0.01))
    # run_for_prefix(prefix='_jup', t='7e7',
    #                p_rat=(0.544)**(-3/2), ain=5.202,
    #                Is=[1.3, 2.48],
    #                mps=[317, 95])
    # run_for_prefix(prefix='_SESE', t='7e5')
    # run_for_prefix(prefix='_3p', t='7e5',
    #                Is=[1, 5, 5],
    #                mps=[1, 10, 10])
    # run_for_prefix(prefix='_SESEwide', t='7e6', p_rat=3)
    # run_for_prefix(prefix='_3pwide', t='7e6',
    #                Is=[1, 5, 5],
    #                mps=[1, 10, 10], p_rat=3)
    scan_alpha()
    scan_alpha(m_in=5, m_out=5, fn='scan_alpha5', name='scan5')
