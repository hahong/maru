#!/usr/bin/env python

import sys
import numpy as np
import cPickle as pk
from matplotlib import rc, use
use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
import pylab as pl
from common_fn import parse_opts2

XLIM = [-100, 400]
DEF_NBINS = 500


# ------------------------------------------------------------
# assume interval is (-100, 250)
def load_db(files, db, n_bins=DEF_NBINS, t_shift=100000, verbose=1):
    n = len(files)
    #db['n_bins'] = n_bins
    for i_f, f in enumerate(files):
        if verbose > 0: print 'At (%d/%d): %s' % (i_f+1, n, f)
        dat = pk.load(open(f))

        for el in dat['all_spike']:
            n_elec = len(dat['all_spike'])
            if verbose > 1: print 'At channel: %d/%d' % (el, n_elec)
            i_el = el - 1

            for img in dat['all_spike'][el]:
                trials = dat['all_spike'][el][img]
                if img not in db:
                    db[img] = {}
                    db[img]['cnt'] = 0
                    db[img]['mat'] = np.zeros((n_elec, n_bins)).astype('int')

                # kinda wrong... cause it'll add the same thing `n_elec` times
                db[img]['cnt'] += len(trials)
                mat = db[img]['mat']

                for tr in trials:
                    for t0 in tr:
                        t = int(np.round((t0 + t_shift) / 1000.))
                        if t < 0 or t >= n_bins: continue

                        mat[i_el][t] += 1


# ------------------------------------------------------------
def init():
    rc('font', **{
        'family'          : 'serif',
        'serif'           : ['Times New Roman']})
    rc('figure',**{
        'figsize'           : '8,8.5',
        'subplot.left'      : '0.03',
        'subplot.right'     : '0.97',
        'subplot.bottom'    : '0.05',
        'subplot.top'       : '0.95',
        'subplot.wspace'    : '0.25',
        'subplot.hspace'    : '0.3',
    })



def SNR(wav0, chunk=15):
    wav = wav0[1:-1]

    ibs = range(0, len(wav), chunk)
    ies = ibs[1:] + [len(wav)]

    S = np.mean(wav)
    #D = np.max(wav) - np.min(wav)
    D = np.std(wav, ddof=1)
    Ns = []
    for ib, ie in zip(ibs, ies):
        if ie - ib < 2: continue
        Ns.append(np.std(wav[ib:ie], ddof=1))
    N = np.mean(Ns)

    return S/N, D/N


# ------------------------------------------------------------
def plot_all(db, pp, n_bins=None, t_shift=100000, log='log.tmp', log2=None, verbose=0):
    keys = db.keys()
    k0 = keys[0]
    n_elec, n_bins0 = db[k0]['mat'].shape
    if n_bins is None: n_bins = n_bins0

    n = len(keys)
    M = np.zeros((n_elec, n_bins))   # averaged across the images
    B = np.zeros((n_elec, n_bins))   # best image
    C = np.zeros(n_elec)
    L = {}
    t = np.array(range(-t_shift/1000, -t_shift/1000 + n_bins)) + 0.5

    for img in keys:
        if verbose > 1:
            print img
            print db[img]['cnt']
        # converting to Spkies/s
        # DBG print db[img]['mat']
        # DBG print db[img]['cnt']
        # DBG print img
        T = db[img]['mat'] / (float(db[img]['cnt']) / n_elec) * 1000.
        M += T

        c = T.mean(axis=1)
        ii = np.nonzero(c > C)[0]   # all channels that have bigger responses
        C[ii] = c[ii]
        B[ii,:] = T[ii,:]
        for ch in ii: L[int(ch)] = img

    M /= float(n)



    # -- for each chaneel
    snrs = []
    dnrs = []
    if log2 is None: log2 = log + '.ch.log'
    fp = open(log2, 'wt')
    print >>fp, '# 1-based ch, SNR, DNR'
    for ch in xrange(n_elec):
        if verbose > 0: print '-->', ch
        pl.figure(figsize=(18,10))
        N = M[ch]
        snr, dnr = SNR(N)

        pl.plot(t, N)
        pl.xlim(XLIM)
        pl.title('Firing rate averaged across all images (Ch %d, SNR = %f, DNR = %f)' % (ch + 1, snr, dnr))
        print >>fp, (ch+1), snr, dnr

        pp.savefig()
        pl.close()
        snrs.append(snr)
        dnrs.append(dnr)
    fp.close()
    snrs = np.array(snrs)
    dnrs = np.array(dnrs)
    snrm = np.mean(snrs[np.isfinite(snrs)])
    dnrm = np.mean(dnrs[np.isfinite(dnrs)])

    # -- summary
    pl.figure(figsize=(18,10))
    # averaged PSTH for all images and channels
    pl.subplot(221)
    pl.plot(t, M.mean(axis=0))
    pl.xlim(XLIM)
    pl.title('Firing rate averaged across all images and channels')

    # averaged PSTH for all images (different color for each ch)
    pl.subplot(222)
    pl.plot(t, M.T)
    pl.xlim(XLIM)
    pl.title('Firing rate averaged across all images (mean SNR = %f, mean DNR = %f)' % (snrm, dnrm))

    # PSTH for best images averaged for all chs
    pl.subplot(223)
    pl.plot(t, B.mean(axis=0))
    pl.xlim(XLIM)
    pl.title('Firing rate of best images averaged across all channels')

    # PSTH for best images for each ch
    d = 10
    t2 = np.zeros(n_bins/d)
    B2 = np.zeros((n_elec, n_bins/d))
    for i, s in enumerate(range(0, n_bins, d)):
        e = s + d
        t2[i] = np.mean(t[s:e])
        B2[:,i] = np.mean(B[:,s:e], axis=1)

    pl.subplot(224)
    # ignore the last time bin
    pl.plot(t2[:-1], B2[:,:-1].T)
    pl.xlim(XLIM)
    pl.title('Firing rate of best images')

    fp = open(log, 'wt')
    print >>fp, '# 1-based ch, preferred stim'
    for ch in sorted(L.keys()):
        print >>fp, (ch+1), L[ch]
    fp.close()

    pp.savefig()
    pl.close()


# ------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print 'plot_PSTH.py <out prefix> <in1.psf.pk> [in2.psf.pk] ...'
        print
        print 'Options:'
        print '  --n_bins=#       Number of 1-ms bins'
        return

    # -- parse options and arguments
    args, opts = parse_opts2(sys.argv[1:])

    of = args[0]
    files = args[1:]
    print 'OF =', of
    print 'FILES =', files

    n_bins = DEF_NBINS
    if 'n_bins' in opts:
        n_bins = int(opts['n_bins'])
        print '  * n_bins =', n_bins

    db = {}
    load_db(files, db, n_bins=n_bins)

    # -- plot
    init()
    fpic = of + '.pdf'
    flog = of + '.bstim.log'
    flog2 = of + '.snr.log'

    pp = PdfPages(fpic)
    plot_all(db, pp, log=flog, log2=flog2)
    pp.close()

    print 'Done.'


if __name__ == '__main__':
    main()
