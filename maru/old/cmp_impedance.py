#!/usr/bin/env python

import sys
import numpy as np
from common_fn import parse_opts2

ELECS = range(1, 97)


def load_impedance(fi, elecs=ELECS):
    im = np.zeros(len(elecs))    # XXX: kludge...

    for line in open(fi, 'rt').readlines():
        tokens = line.strip().split()
        if len(tokens) != 3 or tokens[2] != 'kOhm':
            continue

        ch = int(tokens[0].split('elec')[-1])
        if ch not in elecs:
            continue

        im[ch - 1] = int(tokens[1])

    return im


def cmp_impedance(fi1, fi2):
    im1 = load_impedance(fi1)
    im2 = load_impedance(fi2)

    denom = np.max([im1, im2], axis=0)
    nom = np.abs(im1 - im2)
    ratio = nom / denom

    return ratio


def cmp_impedance_all(files, plot=None):
    fi1s = files[0::2]
    fi2s = files[1::2]
    res = []

    for fi1, fi2 in zip(fi1s, fi2s):
        ratio = cmp_impedance(fi1, fi2)
        si = np.argsort(ratio)    # sorted index
        res.append([(r, ch + 1) for r, ch in zip(ratio[si], si)])
    n_ch = len(res[0])

    for i_rank in range(n_ch):
        row = []
        for col in range(len(res)):
            row.append(res[col][i_rank])

        fmt = '%3d' + '\t%1.3f (Ch %2d)' * len(row)
        print fmt % ((i_rank + 1,) + tuple(np.ravel(row)))

    if plot != None:
        from matplotlib import use
        use('pdf')
        import pylab as pl

        pl.figure()
        for i, dat0 in enumerate(res):
            dat = [d for d, _ in dat0]
            y, x0 = np.histogram(dat, range=(0, 1), bins=20)
            x = (x0[1:] + x0[:-1]) / 2.
            lbl = 'Pair %d' % (i + 1)
            pl.plot(x, y, label=lbl)
        pl.legend()
        pl.xlabel('Ratio')
        pl.ylabel('Count')
        pl.title('Comparison of impedance values visualized in histogram')
        pl.savefig(plot)


def main():
    if len(sys.argv) < 3:
        print 'cmp_impedance.py [opts] <impedance log 1a.txt> <impedance log' \
                '1b.txt> [<log 2a.txt> <log 2b.txt>] ...'
        print 'cmp_impedance.py: compare pairs of impedance log files.'
        print 'Options:'
        print '   --plot=<filename.pdf>'
        return

    # -- parse options and arguments
    args, opts = parse_opts2(sys.argv[1:])

    plot = None
    if 'plot' in opts:
        plot = opts['plot']
        print '* Plotting into', plot

    # -- do the work
    files = args
    cmp_impedance_all(files, plot=plot)


if __name__ == '__main__':
    main()
