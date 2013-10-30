#!/usr/bin/env python

import sys
import cPickle as pk
import numpy as np
import time
from joblib import Parallel, delayed
from common_fn import parse_opts2

BLANKS = 'blanks'
RMAX = 0.1     # calculate d'-CV-max10%
NITER = 16     # from on 16 iterations
RSPLIT = 0.5   # with 0.5/0.5 splitting
RSEED = 0
ABS = True     # compare absolute values
NJOBS = -1     # maximum worker threads

TOPN = 128
NCOL = 10
NROW = 10
NPLOTS = 96
PREF='cvdp'
# ---------------------------------------------------------------------------------------
def dprime(m_y, s_y, m_n, s_n):
    return (m_y - m_n)/np.sqrt(((s_y * s_y) + (s_n * s_n)) / 2.)

def dp(img, blank):
    m_y = np.mean(img)
    s_y = np.std(img, ddof=1)
    m_n = np.mean(blank)
    s_n = np.std(blank, ddof=1)
    return dprime(m_y, s_y, m_n, s_n)

def dp_thread(dat_ch, nblank, babs=ABS, rsplit=RSPLIT, rmax=RMAX):
    blanks = list(dat_ch[nblank])
    np.random.shuffle(blanks)
    lb = len(blanks)
    blank1 = blanks[:int(lb*rsplit)]
    blank2 = blanks[int(lb*rsplit):]
    dps = []

    for n in dat_ch:
        imgs = list(dat_ch[n])
        np.random.shuffle(imgs)
        li = len(imgs)
        img1 = imgs[:int(li*rsplit)]
        img2 = imgs[int(li*rsplit):]

        # compute cross-validated d's
        dp1 = dp(img1, blank1)
        dp2 = dp(img2, blank2)

        # take absolute if needed
        if babs:
            dp1 = abs(dp1)
            dp2 = abs(dp2)

        dps.append((dp1, dp2, n)) 

    # take rmax portion only
    sdps = sorted(dps, reverse=True)
    ls = len(sdps)
    return sdps[:int(ls*rmax)]


def dp_cv(db, niter=NITER, verbose=1):
    dat = db['dat']
    iid2num = db['iid2num']
    num2iid = db['num2iid']
    nblank = iid2num[BLANKS]
    res = {}   # res[ch] = {sig_iids:sig_iids, m_dp:m_cv_rmax_dp, s_dp:s_cv_rmax_dp}

    for ch in dat:
        t0 = time.time()
        max_dps = Parallel(n_jobs=NJOBS, verbose=0)(delayed(dp_thread)(dat[ch], nblank) for _ in range(niter))

        cv_rmax_dp = [d[1] for sublst in max_dps for d in sublst]
        sig_iids = list(set([num2iid[d[2]] for sublst in max_dps for d in sublst]))
        m_cv_rmax_dp = np.mean(cv_rmax_dp)
        s_cv_rmax_dp = np.std(cv_rmax_dp, ddof=1)
        
        # save
        res[ch] = {'sig_iids':sig_iids, 'm_dp':m_cv_rmax_dp, 's_dp':s_cv_rmax_dp, 'blanks':dat[ch][nblank]}
        t1 = time.time()

        if verbose:
            print '  Ch %d: m_cv_rmax_dp = %1.4f   (std = %1.4f; dt=%f)' % (ch, m_cv_rmax_dp, s_cv_rmax_dp, t1-t0)

    return res


# ========================----------------------------------------------------------
def init():
    from matplotlib import rc
    rc('font', **{
        'family'          : 'serif',
        'serif'           : ['Times New Roman'],
        'size'            : 7})
    rc('figure',**{
        'figsize'           : '8,8.5',
        'subplot.left'      : '0.04',
        'subplot.right'     : '0.97',
        'subplot.bottom'    : '0.065',
        'subplot.top'       : '0.96',
        'subplot.wspace'    : '0.25',
        'subplot.hspace'    : '0.44',
    })


# ---
def plot_all(db, ncol=NCOL, nrow=NCOL, nplots=NPLOTS, px=PREF):
    import pylab as pl
    from matplotlib import rc

    npage = len(db) / nplots
    ipage = 0
    for i, ch in enumerate(db):
        i_segment = i % nplots
        if i_segment == 0:
            if ipage != 0: pl.savefig(px + '_p%d.pdf' % ipage)
            pl.figure(figsize=(18,10))
            ipage += 1
        pl.subplot(nrow, ncol, i_segment+1)
        pl.hist(db[ch]['blanks'], bins=20)
        pl.title('Ch' + str(ch), fontsize='medium')
    return ipage


def pick_top(db, topn, bad=[], ng=[], prndp=False, px=PREF):
    # convert to 0-based
    dps = [(db[ch]['m_dp'], ch-1) for ch in db if ch not in bad]
    top = sorted(dps, reverse=True)[:topn]

    # XXX: kludge
    topch = np.array([t[1] for t in top])
    topchng = []
    A = topch[topch < 96]
    M = topch[(topch >= 96) & (topch < 96*2)]
    P = topch[topch >= 96*2]

    print '0-Based selected %d (n_A=%d, n_M=%d, n_P=%d):' % (topn, len(A), len(M), len(P))
    print A
    print M
    print P
    print

    for ch in ng:
        if ch - 1 in topch: topchng.append(ch - 1)
    print 'No good channels (0-based):', topchng

    if prndp:
        rdp = []; rch0 = []; rarr = []
        print
        print 'cv-dprime values:'
        print 'dv-dpime \t ch(0-based) \t Arr'
        for t in top:
            ch = t[1]
            if ch < 96: arr = 'A'
            elif (ch >= 96) and (ch < 96*2): arr = 'M'
            else: arr = 'P'

            print t[0],'\t',ch,'\t',arr
            rdp.append(t[0])
            rch0.append(ch)
            rarr.append(arr)
        pk.dump({'dp': rdp, 'ch0': rch0, 'arr': rarr}, open(px+'_sorted128.pk', 'wb'))

# ----
def load_all(files, skipchk=False):
    db = {}
    for f in files:
        offset = len(db)
        db0 = pk.load(open(f))
        for ch in db0:
            if not skipchk:
                assert all([all(np.isfinite(db0[ch][k])) for k in db0[ch]])
            db[ch + offset] = db0[ch]

    return db



# ---------------------------------------------------------------------------------------
def main(rseed=RSEED):
    if len(sys.argv) < 2:
        print 'Prep Mode:'
        print 'summarizedp_cv.py prep <d\'-CV output.pk> <combined frate.pk>'
        print 'Note: input pickle files are made by combine_firrate_data.py'
        print
        print 'Result Formatting Mode:'
        print 'summarizedp_cv.py res [options] <d\'-CV output A.pk> <d\'-CV output M.pk> <d\'-CV output P.pk>'
        print 'Options:'
        print '   --prndp            Print d\' values'
        print '   --doplot           Plot PSTHs of blanks for each channel'
        print '   --px=prefix        Prefix of output files'
        print '   --bad=ch1,ch2,..   Comma separated list of 1-based bad channels'
        print '   --ng=ch1,ch2,...   Comma separated list of 1-based not-so-good channels'
        print '   --topn=#           Pick top # channels' 
        return

    # parse options and arguments
    args, opts = parse_opts2(sys.argv[1:])

    mode = args[0]
    np.random.seed(rseed)    

    # prepare d'-cv computation
    if mode == 'prep':
        fo = args[1]
        fi = args[2]
        print '* Output d\'-CV result:', fo

        db = pk.load(open(fi))
        res = dp_cv(db)
        pk.dump(res, open(fo, 'wb'))
    # or give the results...
    elif mode == 'res':
        doplot, prndp, topn, px = False, False, TOPN, PREF
        bad = []
        ng = []

        if 'doplot' in opts:
            from matplotlib import rc, use
            use('pdf')
            import pylab as pl
            init()
            doplot = True
            print '* DOPLOT'
        if 'prndp' in opts:
            prndp = True
            print '* PRNDP'
        if 'topn' in opts:
            topn = int(opts['topn'])
            print '* TOPN =', topn
        if 'bad' in opts:
            bad = [int(b) for b in opts['bad'].split(',')]
            print '* BAD =', bad
        if 'ng' in opts:
            ng = [int(b) for b in opts['ng'].split(',')]
            print '* NG =', ng
        if 'px' in opts:
            px = opts['px']
            print '* PX =', px

        files = args[1:]
        db = load_all(files, skipchk=True)
        pick_top(db, topn, bad=bad, ng=ng, prndp=prndp, px=px)
        
        if doplot:
            ipage = plot_all(db, px=px)
            pl.savefig(px +'_p%d.pdf' % (ipage))
            print 'Done.'
            #pl.show()
    else:
        print 'Unrecognized mode. Aborting.'


if __name__ == '__main__':
    main()
