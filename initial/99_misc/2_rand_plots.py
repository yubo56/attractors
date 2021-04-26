'''
more random plots
'''
# convert 2_nonad_rot.png -crop 650x700+100+75 2_nonad_rot_cropped.png
import numpy as np
import scipy.optimize as opt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
LW=3.5

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

def get_cs(I, num_pts=100):
    ''' gets list of cassini states at an uneven grid of etas '''
    etac = get_etac(I)
    eps = 1e-5
    min_eta, max_eta = etac / 30, etac * 30
    etas = np.concatenate((
        np.log(np.linspace(np.exp(min_eta), np.exp(etac - eps), num_pts)),
        np.exp(np.linspace(np.log(etac + eps), np.log(max_eta), num_pts)),
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
    return etas, cs_vals, etas_four, etac

def plot_cs(I=np.radians(5)):
    etas, cs_vals, etas_four, etac = get_cs(I)

    plt.semilogx(etas_four, np.degrees(cs_vals[0]), 'y', lw=LW, label='1')
    plt.semilogx(etas, np.degrees(cs_vals[1]), 'r', lw=LW, label='2')
    plt.semilogx(etas, np.degrees(cs_vals[2]), 'm', lw=LW, label='3')
    plt.semilogx(etas_four, np.degrees(cs_vals[3]), 'c', lw=LW, label='4')
    plt.xlabel(r'$\eta$', fontsize=22)
    plt.ylabel(r'$\theta$ (deg)', fontsize=22)
    plt.xlim([min(etas), max(etas)])
    plt.yticks([-180 + np.degrees(I),
                -90,
                np.degrees(I),
                90],
               [r'$%d$' % (np.degrees(I) - 180),
                r'$-90$',
                r'$%d$' % np.degrees(I),
                r'$90$'])
    # place upper right corner of legend flush against eta -> infinity asymptote
    # (np.degrees(I))
    ymin, ymax = plt.ylim()
    y_perc = (np.degrees(I) - ymin) / (ymax - ymin)
    legend = plt.legend(fontsize=16, loc='upper right',
                        bbox_to_anchor=(1.0, y_perc))
    plt.axhline(np.degrees(I), lw=0.8, c='k', ls='dashed')
    plt.axhline(-180 + np.degrees(I), lw=0.8, c='k', ls='dashed')
    plt.axvline(etac, c='k', lw='0.8', ls='dashed')
    # plt.title(r'$I = %d^\circ$' % np.degrees(I))
    plt.tight_layout()
    plt.savefig('2_cs_locs', dpi=400)

    legend.remove()
    # place upper right corner of legend flush against eta -> infinity asymptote
    # (np.degrees(I))
    legend = plt.legend(fontsize=16, loc='upper left',
                        bbox_to_anchor=(0.0, y_perc))
    xlims = plt.xlim()
    plt.xlim(xlims[1], xlims[0]) # flip
    plt.savefig('2_cs_locs_flip', dpi=400)
    plt.clf()

def plot_eigens(I=np.radians(5)):
    etas, cs_vals, etas_four, etac = get_cs(I, num_pts=200)
    def lambda2(eta, q, sign):
        # note that the 4 CSs are 0, pi/2, -pi, -pi/2 by convention, which
        # correspond to sign choices of -1 for all of them.
        return (
            (np.sin(q) - sign * eta * np.sin(I) / (np.sin(q)**2)) *
            (sign * eta * np.sin(I))) / (1 + eta**2)
    plt.semilogx(etas_four,
                 [lambda2(e, q, -1)
                  for e, q in zip(etas_four, cs_vals[0])],
                 'y', label='1', lw=LW)
    plt.semilogx(etas,
                 [lambda2(e, q, -1)
                  for e, q in zip(etas, cs_vals[1])],
                  'r', label='2', lw=LW)
    plt.semilogx(etas,
                 [lambda2(e, q, -1)
                  for e, q in zip(etas, cs_vals[2])],
                  'm', label='3', lw=LW)
    plt.semilogx(etas_four,
                 [lambda2(e, q, -1)
                  for e, q in zip(etas_four, cs_vals[3])],
                  'c', label='4', lw=LW)
    # plt.yscale('symlog', linthreshy=1e-5)
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\lambda^2/ (1 + \eta^2)$')
    legend = plt.legend(loc='upper left', fontsize=14,
               bbox_to_anchor=(0.5, 0.7))
    plt.xlim([min(etas), max(etas)])
    plt.axhline(0, lw=0.8, c='k', ls='dashed')
    plt.axvline(etac, c='k', lw='0.8', ls='dashed')
    # plt.title(r'$I = %d^\circ$' % np.degrees(I))
    plt.tight_layout()
    plt.savefig('2_lambdas.png', dpi=400)

    # add plots for other values of phi
    legend.remove()
    plt.legend(loc='upper right', fontsize=12, ncol=2,
               bbox_to_anchor=(plt.xlim()[1], 0),
               bbox_transform=plt.gca().transData)
    lw = 0.4 * LW
    plt.semilogx(etas_four,
                 [lambda2(e, q, -1 / 2)
                  for e, q in zip(etas_four, cs_vals[0])],
                 'y--', lw=lw, alpha=0.5)
    plt.semilogx(etas,
                 [lambda2(e, q, -1 / 2)
                  for e, q in zip(etas, cs_vals[1])],
                  'r--', lw=lw, alpha=0.5)
    plt.semilogx(etas,
                 [lambda2(e, q, -1 / 2)
                  for e, q in zip(etas, cs_vals[2])],
                  'm--', lw=lw, alpha=0.5)
    plt.semilogx(etas_four,
                 [lambda2(e, q, -1 / 2)
                  for e, q in zip(etas_four, cs_vals[3])],
                  'c--', lw=lw, alpha=0.5)
    # plt.semilogx(etas_four,
    #              [lambda2(e, q, -np.sqrt(3) / 2)
    #               for e, q in zip(etas_four, cs_vals[0])],
    #              'y:', lw=lw, alpha=0.5)
    # plt.semilogx(etas,
    #              [lambda2(e, q, -np.sqrt(3) / 2)
    #               for e, q in zip(etas, cs_vals[1])],
    #              'r:', lw=lw, alpha=0.5)
    # plt.semilogx(etas,
    #              [lambda2(e, q, -np.sqrt(3) / 2)
    #               for e, q in zip(etas, cs_vals[2])],
    #              'm:', lw=lw, alpha=0.5)
    # plt.semilogx(etas_four,
    #              [lambda2(e, q, -np.sqrt(3) / 2)
    #               for e, q in zip(etas_four, cs_vals[3])],
    #              'c:', lw=lw, alpha=0.5)
    plt.yscale('symlog', linthresh=0.1)
    plt.tight_layout()
    plt.savefig('2_lambdas_full.png', dpi=300)
    plt.clf()

def get_xy(angle, mag=1):
    return 1 - mag * np.sin(np.radians(angle)), mag * np.cos(np.radians(angle))

def plot_3vec():
    ''' plots the relative orientations of the three vectors '''
    offset = 0.02 # offset for text from arrow tip
    alpha = 0.8

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.axis('off')
    ax.set_xlim(0.2, 1.1)
    ax.set_ylim(-0.1, 1.1)

    # central dot
    ax.plot(1, 0, 'ko', ms=8, zorder=np.inf)
    arrowprops = lambda c: {'fc': c, 'alpha': alpha, 'lw': 0,
                            'width': 3, 'headwidth': 12}

    # draw three arrows
    l_xy = get_xy(0)
    l_c = 'k'
    ax.annotate('', xy=l_xy, xytext=(1, 0),
                 arrowprops=arrowprops(l_c))
    ax.text(l_xy[0] - offset / 3, l_xy[1] + offset, r'$\hat{\mathbf{l}}$',
             fontdict={'c': l_c})

    ld_q = 20
    ld_xy = get_xy(ld_q)
    ld_c = 'b'
    ax.annotate('', xy=ld_xy, xytext=(1, 0),
                 arrowprops=arrowprops(ld_c))
    ax.text(ld_xy[0] - offset / 2, ld_xy[1] + offset,
            r'$\hat{\mathbf{l}}_{\rm d}$', fontdict={'c': ld_c})

    s_q = 50
    s_xy = get_xy(s_q)
    s_c = 'r'
    ax.annotate('', xy=s_xy, xytext=(1, 0),
                 arrowprops=arrowprops(s_c))
    ax.text(s_xy[0] - offset, s_xy[1] + offset, r'$\hat{\mathbf{s}}$',
             fontdict={'c': s_c})

    # draw arcs
    # center, (dx, dy), rotation, start angle, end angle (degrees)
    arc_lw = 3
    ld_arc = patches.Arc((1, 0), 1.0, 1.0, 90, 0, ld_q,
                         color=l_c, lw=arc_lw, alpha=alpha,
                         label='1')
    ax.add_patch(ld_arc)
    s_arc = patches.Arc((1, 0), 0.6, 0.6, 90, 0, s_q,
                        color=s_c, lw=arc_lw, alpha=alpha)
    ax.add_patch(s_arc)
    # label arcs
    ax.text(1 - np.sin(np.radians(ld_q * 0.6)) * 0.5,
            np.cos(np.radians(ld_q * 0.6)) * 0.5 + offset,
            r'$I$',
            fontdict={'c': l_c})
    ax.text(1 - np.sin(np.radians(0.8 * s_q)) * 0.3 - 4 * offset,
            np.cos(np.radians(0.8 * s_q)) * 0.3 + 2 * offset,
            r'$+\theta$',
            fontdict={'c': s_c})
    xy_s_tip = (
        1 - 0.3 * np.sin(np.radians(s_q)),
        0.3 * np.cos(np.radians(s_q)))
    ax.annotate('', xy=xy_s_tip - np.array([0.003, 0.003]), xytext=xy_s_tip,
                arrowprops=arrowprops(s_c))

    ax.set_aspect('equal')
    plt.savefig('2_3vec', dpi=400)
    plt.clf()

def plot_nonad_diagram():
    ''' plots the relative orientations of the three vectors '''
    offset = 0.02 # offset for text from arrow tip
    alpha = 0.8

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.axis('off')
    ax.set_xlim(0, 1.2)
    ax.set_ylim(-0.02, 1.1)
    fs = 14

    # central dot
    ax.plot(1, 0, 'ko', ms=8, zorder=np.inf)
    arrowprops = lambda c: {'fc': c, 'ec': c, 'lw': 1}

    # draw three arrows
    l_xy = get_xy(0)
    l_c = 'b'
    ax.annotate('', xy=l_xy, xytext=(1, 0),
                 arrowprops=arrowprops(l_c), zorder=2)
    ax.text(l_xy[0] - offset / 3, l_xy[1] + offset, r'$\hat{\mathbf{l}}$',
            fontdict={'c': l_c, 'size': fs}, zorder=2)

    ld_q = 20
    ld_xy = get_xy(ld_q)
    ld_c = 'k'
    ax.annotate('', xy=ld_xy, xytext=(1, 0),
                arrowprops=arrowprops(ld_c), zorder=2)
    ax.text(ld_xy[0] - offset / 2, ld_xy[1] + offset,
            r'$\hat{\mathbf{l}}_{\rm d}$', fontdict={'c': ld_c, 'size': fs}, zorder=2)

    ld_qf = 50
    ldqf_xy = get_xy(ld_qf)
    ldqf_c = (0.5, 0.5, 0.5)
    ax.annotate('', xy=ldqf_xy, xytext=(1, 0),
                arrowprops=arrowprops(ldqf_c), zorder=2)
    # ax.text(ldqf_xy[0] - 5 * offset, ldqf_xy[1] + offset, r'$\theta_{\rm 0f}$',
    #         fontdict={'c': ldqf_c, 'size': fs}, zorder=2)

    # draw locations of obliquities
    def draw(q_cent, c, c_fill, q_sdi=10, mag1=0.9, mag2=0.45, hatch=None):
        vrts_orig = np.array([get_xy(q_cent + q_sdi, mag=mag1),
                              get_xy(q_cent - q_sdi, mag=mag1)])
        vrts = np.array([get_xy(q_cent + q_sdi, mag=mag1 * 0.9),
                         get_xy(q_cent - q_sdi, mag=mag1 * 0.9),
                         get_xy(q_cent - q_sdi, mag=mag2),
                         get_xy(q_cent + q_sdi, mag=mag2)])
        # ax.plot([1, vrts_orig[0, 0]], [0, vrts_orig[0, 1]], c=c, zorder=1)
        # ax.plot([1, vrts_orig[1, 0]], [0, vrts_orig[1, 1]], c=c, zorder=1)
        ax.annotate('', xy=vrts_orig[0], xytext=(1, 0),
                    arrowprops={**arrowprops(c), 'width': 1}, zorder=2)
        ax.annotate('', xy=vrts_orig[1], xytext=(1, 0),
                    arrowprops={**arrowprops(c), 'width': 1}, zorder=2)
        ax.fill(vrts[:, 0], vrts[:, 1], c=c_fill, zorder=0)
        if hatch:
            fill_polygon = patches.Polygon(vrts, fill=False, hatch=hatch)
            ax.add_patch(fill_polygon)
        return vrts_orig
    verts_i = draw(ld_q, (1, 0.3, 0.3), (0.7, 0.2, 0.2), mag2=0)
    # verts_f = draw(ld_qf, (1, 0.5, 0.5), (0.7, 0.4, 0.4), hatch='/', mag2=0)
    verts_f = draw(ld_qf, (1, 0.5, 0.5), (0.7, 0.4, 0.4), mag2=0)

    # label angular coordinates
    ax.text(verts_i[0, 0] - fs / 205, verts_i[0, 1],
            r'$I + \theta_{\rm sd, i}$', c=(1, 0.3, 0.3),
            size=fs - 2)
    ax.text(verts_i[1, 0] - fs / 205, verts_i[1, 1],
            r'$I - \theta_{\rm sd, i}$', c=(1, 0.3, 0.3),
            size=fs - 2)

    # label arcs
    arc_lw = 2
    ld_arc = patches.Arc((1, 0), 1.4, 1.4, 90, 0, ld_q,
                         color=l_c, lw=arc_lw, alpha=alpha,
                         zorder=10)
    ax.add_patch(ld_arc)
    ax.text(1 - np.sin(np.radians(ld_q * 0.4)) * 0.7,
            np.cos(np.radians(ld_q * 0.4)) * 0.7 + offset,
            r'$I$',
            fontdict={'c': 'k', 'fontsize': fs})
    q0f_arc = patches.Arc((1, 0), 1.1, 1.1, 90, 0, ld_qf,
                         color=(0.5, 0.5, 0.5), lw=arc_lw, alpha=alpha,
                         zorder=10)
    ax.add_patch(q0f_arc)
    ax.text(1 - np.sin(np.radians(ld_qf * 0.78)) * 0.6,
            np.cos(np.radians(ld_qf * 0.78)) * 0.6,
            r'$\theta_{\rm 0f}$',
            fontdict={'c': (0.5, 0.5, 0.5), 'fontsize': fs})

    # label the range of final obliquities
    # ax.plot([verts_f[0, 0], 1], [verts_f[0, 1], verts_f[0, 1]], 'g',
    #         zorder=3, lw=2.5)
    # ax.plot([verts_f[1, 0], 1], [verts_f[1, 1], verts_f[1, 1]], 'g',
    #         zorder=3, lw=2.5)
    # obl_final_label_ycoord = (2 * verts_f[0, 1] + verts_f[1, 1]) / 3
    # ax.text(1 + 2 * offset, obl_final_label_ycoord,
    #         r'$\cos \theta_{\rm f}$', c='g', size=fs)

    # try drawing case where q_sdi > q_0f? Too cluttered
    # draw(ld_q, (0.3, 1, 0.3), (0.2, 0.7, 0.2),
    #      q_sdi=60, mag1=0.45, mag2=0)
    # draw(ld_qf, (0.5, 1, 0.5), (0.4, 0.7, 0.4),
    #      hatch='/', q_sdi=60, mag1=0.45, mag2=0)

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('2_nonad_rot', dpi=200)
    plt.clf()

if __name__ == '__main__':
    # plot_cs(np.radians(5))
    plot_eigens(np.radians(5))
    # plot_3vec()
    # plot_nonad_diagram()
