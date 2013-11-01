#!/usr/bin/env python

import sys
import numpy as np
import cPickle as pk
from matplotlib import rc, use
use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
from common_fn import parse_opts
import pylab as pl
import csv

DEF_ON_RANGE_US = [75000, 175000]    # in microsec
DEF_OFF_RANGE_US = [-50000, 50000]
DEF_PSTH_RANGE_MS = [-50, 200]       # in ms
DEF_RSEED = 0
PROC_CLUSTER = False

NCOLS = 16
NROWS = 8


# ---------------------------------------------------------------------------
def get_one_ch(elec_id, all_spike, include=None, exclude=[], \
        trial_by_trial=False, prange=None):
    arr = []
    n_trial = 0
    if include is None:
        include = all_spike[elec_id]
    for iid in include:
        if iid in exclude:
            continue
        if prange != None:
            n = len(all_spike[elec_id][iid])
            i_begin = int(np.round(n * prange[0]))
            i_end = int(np.round(n * prange[1]))
            trials = all_spike[elec_id][iid][i_begin:i_end]
        else:
            trials = all_spike[elec_id][iid]
        for trial in trials:
            n_trial += 1
            if trial_by_trial:
                arr.append(trial)
            else:
                arr.extend(trial)
    return arr, n_trial


def get_one_img(iid, all_spike, include=None, exclude=[], \
        trial_by_trial=False):
    arr = []
    n_trial = 0
    if include is None:
        include = all_spike  # this gives all electrodes
    for elec_id in include:
        if elec_id in exclude:
            continue
        for trial in all_spike[elec_id][iid]:
            n_trial += 1
            if trial_by_trial: arr.append(trial)
            else: arr.extend(trial)
    return arr, n_trial


def dprime(m_y, s_y, m_n, s_n):
    return (m_y - m_n)/np.sqrt(((s_y * s_y) + (s_n * s_n)) / 2.)

def dprime_bootstrap(c_y, c_n, iter=200):
    if len(c_y) == 0 or len(c_n) == 0: return float('nan')
    dps = []
    n_y = len(c_y)
    n_n = len(c_n)
    c_y = np.array(c_y)
    c_n = np.array(c_n)
    for i in range(iter):
        # picking samples with replacement
        i_y = np.random.randint(0, n_y, n_y)
        i_n = np.random.randint(0, n_n, n_n)
        y = c_y[i_y]
        n = c_n[i_n]
        #np.random.shuffle(c_y)
        #np.random.shuffle(c_n)
        #y = c_y[:n_y/2]
        #n = c_n[:n_n/2]
        # calculate bootstrapped d'
        dp = dprime(y.mean(), y.std(ddof=1), n.mean(), n.std(ddof=1))
        dps.append(dp)
    return np.median(dps), dps


def get_stats(trials, range):
    cnt = []
    # get all indices in the "ON" window
    for trial in trials:
        arr = np.array(trial)
        n_spikes = np.all([arr >= range[0],
                           arr <= range[1]], axis = 0).sum()
        cnt.append(n_spikes)

    # TODO: maybe raising an exception would be better.
    if len(cnt) == 0: m = 0
    else: m = np.mean(cnt)
    if len(cnt) > 1: s = np.std(cnt, ddof=1)
    else: s = 0

    return m, s, cnt


def write_all_to_csv(iterable, fname='out.csv', csvw=None, header=None, append=False):
    # write down to the csv file
    if csvw is None:
        if append:
            csvw = csv.writer(open(fname, 'ab'))
        else:
            csvw = csv.writer(open(fname, 'wb'))
        if header != None:
            csvw.writerow(header)
    csvw.writerows(iterable)
    return csvw


def init_rc(opts=None):
    # init plot fonts
    rc('font', **{
        'family'          : 'serif',
        'serif'           : ['Times New Roman']})
    if opts == 'subplot':
        rc('figure', **{
            'figsize'         : [11, 8.5],
            'subplot.left'    : '0.02',
            'subplot.right'   : '0.98',
            'subplot.bottom'  : '0.02',
            'subplot.top'     : '0.98',
            'subplot.wspace'  : '0.1',
            'subplot.hspace'  : '0.1',
        })
        pl.rcParams.update({
            'axes.labelsize'  : 6,
            'text.fontsize'   : 6,
            'legend.fontsize' : 6,
            'xtick.labelsize' : 6,
            'ytick.labelsize' : 6,
        })
    if opts == 'squeeze':
        rc('figure', **{
            'figsize'         : [11, 8.5],
            'subplot.left'    : '0.02',
            'subplot.right'   : '0.98',
            'subplot.bottom'  : '0.02',
            'subplot.top'     : '0.98',
            'subplot.wspace'  : '0.005',
            'subplot.hspace'  : '0.005',
        })
        pl.rcParams.update({
            'axes.labelsize'  : 6,
            'axes.linewidth'  : 0.5,
            'text.fontsize'   : 6,
            'legend.fontsize' : 6,
            'xtick.labelsize' : 6,
            'ytick.labelsize' : 6,
            'lines.linewidth' : 0.5,
        })
    else:
        rc('figure', **{
            'figsize'         : [7, 5]})


# ---------------------------------------------------------------------------
def get_all_iids(all_spike, as_string=False):
    iids = []
    for elec_id in all_spike:
        iids.extend(all_spike[elec_id].keys())
    iids = list(set(iids) - get_all_iids.exclude)    # return unique image_ids

    if type(iids[0]) is tuple:
        if get_all_iids.num_sort:
            t_iids = []
            for iid in iids:
                s = iid[0].split('_')
                if len(s) == 1:
                    s = [iid[0][:get_all_iids.l_pref], iid[0][get_all_iids.l_pref:]]
                k = (int(s[-1]),) + iid[1:]
                t_iids.append((k, iid))
            t_iids = sorted(t_iids)
            iids = [t_iid[1] for t_iid in t_iids]
        else:
            iids = sorted(iids)
        if as_string:
            import string
            s_iid = []
            for iid in iids:
                s = [str(e) for e in iid]
                s_iid.append(string.join(s, '_'))
            return s_iid
        else:
            return iids
    else:
        if not get_all_iids.num_sort:
            return sorted(iids)
        t_iids = []
        for iid in iids:
            s = iid.split('_')
            if len(s) == 1:
                s = [iid[:get_all_iids.l_pref], iid[get_all_iids.l_pref:]]
            try:
                k = int(s[-1])
                t_iids.append((k, iid))
            except:
                t_iids.append((iid,iid))
        t_iids = sorted(t_iids)
        return [t_iid[1] for t_iid in t_iids]
    return iids
get_all_iids.l_pref = 4
get_all_iids.num_sort = True
get_all_iids.exclude = set([])


# ----------------------------------------------------------------------------
# calc_drpime_ch: the super function -- calculates d' and firing rate.
def calc_dprime_ch(data, verbose=0, opts={}, bootstrap=False, extbsinfo=False,
        on_range=DEF_ON_RANGE_US, off_range=DEF_OFF_RANGE_US):
    proc_cluster = PROC_CLUSTER
    if 'proc_cluster' in opts or 'proc_spksort' in opts:
        print '* collect spike sorting information'
        proc_cluster = True

    all_spike = data['all_spike']
    # if proc_cluster: actvelecs = sorted(data['clus_info'].keys())
    if proc_cluster: actvelecs = sorted(all_spike.keys())
    else: actvelecs = sorted(data['actvelecs'])
    all_iids = get_all_iids(all_spike)
    dp_all = []; fir_all = []; fir_byimg = []; fir_byimg_trials = {}
    dp_off_byimg = []
    dp_blank_byimg = []
    nfac_on = 1000000. / (on_range[1] - on_range[0])
    nfac_off = 1000000. / (off_range[1] - off_range[0])
    prange = None

    blanks = [all_iids[-1]]
    if 'bootstrap' in opts:   # DEPRECATED!!!!!
        bsiter = int(opts['bootstrap'])
        bootstrap = True
        print '* bootstrap:', bsiter
    if 'blanks' in opts:
        excl_start = int(opts['blanks'].split(',')[0])
        excl_end = int(opts['blanks'].split(',')[1])
        blanks = all_iids[excl_start:excl_end + 1]
    if 'blanks_patt' in opts:
        patt = opts['blanks_patt']
        blanks = [iid for iid in all_iids if patt in iid]
    if 'prange' in opts:
        p_begin = float(opts['prange'].split(',')[0])
        p_end = float(opts['prange'].split(',')[1])
        prange = (p_begin, p_end)
    if 'extbsinfo' in opts:
        fn_pk = opts['extbsinfo']
        # Too big: outpk = open(fn_pk, 'wb')
        extbsinfo = True
        csvw = write_all_to_csv([['elec_id', 'img_id', 'dp_type'] + ['bstrp_%d' % (iv + 1) for iv in range(bsiter)]], fname=fn_pk)
        print '* extbsinfo:', fn_pk
    if verbose != 0:
        print '* blanks:', blanks
        if prange != None: print '* %%range: [%1.2f, %1.2f)' % prange


    # calculate statistics for blanks
    m_blank = {}
    s_blank = {}
    c_blank = {}
    for elec_id in actvelecs:
        trials, _ = get_one_ch(elec_id, all_spike,
                               include=blanks,
                               trial_by_trial=True,
                               prange=prange)
        # get stats for the "on" window
        m_blank[elec_id], s_blank[elec_id], c_blank[elec_id] = \
                get_stats(trials, on_range)
        # DEBUG: print '**', elec_id, c_blank[elec_id]

    for elec_id in actvelecs:
        trials, _ = get_one_ch(elec_id, all_spike,
                               exclude=blanks,
                               trial_by_trial=True,
                               prange=prange)
        # get stats for the "on" window
        m_on, s_on, c_on = get_stats(trials, on_range)
        # get stats for the "off" window
        m_off, s_off, c_off = get_stats(trials, off_range)

        # calculation of d'
        if bootstrap:
            dp_vs_off, dps_off = dprime_bootstrap(c_on, c_off, iter=bsiter)
            dp_vs_blank, dps_blank = dprime_bootstrap(c_on, c_blank[elec_id], iter=bsiter)
            if extbsinfo:
                #pk.dump([(elec_id, 'all', 'off'), dps_off], outpk)
                #pk.dump([(elec_id, 'all', 'off'), dps_off], outpk)
                write_all_to_csv([[elec_id, 'all', 'off'] + dps_off], csvw=csvw)
                write_all_to_csv([[elec_id, 'all', 'blank'] + dps_blank], csvw=csvw)
        else:
            dp_vs_off = dprime(m_on, s_on, m_off, s_off)
            dp_vs_blank = dprime(m_on, s_on, m_blank[elec_id], s_blank[elec_id])

        # calculation of firing rate
        fir_on = m_on * nfac_on
        fir_off = m_off * nfac_off
        fir_blank = m_blank[elec_id] * nfac_on

        # save all
        dp_all.append((elec_id, dp_vs_off, dp_vs_blank))
        fir_all.append((elec_id, fir_on, fir_off, fir_blank))

        # calculate d' for each image ------------------
        dps_vs_off = [elec_id]
        dps_vs_blank = [elec_id]
        firs = [elec_id]
        fir_byimg_trials[elec_id] = {}
        for iid in all_iids:
            trials, _ = get_one_ch(elec_id, all_spike,
                                   include=[iid],
                                   trial_by_trial=True,
                                   prange=prange)

            # get stats for the "on" window
            m_on, s_on, c_on = get_stats(trials, on_range)
            # get stats for the "off" window
            m_off, s_off, c_off = get_stats(trials, off_range)

            # calculation of d'
            if bootstrap:
                dp_vs_off, dps_off = dprime_bootstrap(c_on, c_off, iter=bsiter)
                dp_vs_blank, dps_blank = dprime_bootstrap(c_on, c_blank[elec_id], iter=bsiter)
                if extbsinfo:
                    #pk.dump([(elec_id, iid, 'off'), dps_off], outpk)
                    #pk.dump([(elec_id, iid, 'blank'), dps_blank], outpk)
                    write_all_to_csv([[elec_id, iid, 'off'] + dps_off], csvw=csvw)
                    write_all_to_csv([[elec_id, iid, 'blank'] + dps_blank], csvw=csvw)
            else:
                dp_vs_off = dprime(m_on, s_on, m_off, s_off)
                dp_vs_blank = dprime(m_on, s_on, m_blank[elec_id], s_blank[elec_id])

            # calculation of firing rate
            fir_on = m_on * nfac_on

            # save all
            dps_vs_off.append(dp_vs_off)
            dps_vs_blank.append(dp_vs_blank)
            firs.append(fir_on)
            fir_byimg_trials[elec_id][iid] = {}
            fir_byimg_trials[elec_id][iid]['on'] = np.array(c_on) * nfac_on
            fir_byimg_trials[elec_id][iid]['off'] = np.array(c_off) * nfac_off

        fir_blank = m_blank[elec_id] * nfac_on
        fir_byimg_trials[elec_id]['blanks'] = np.array(c_blank[elec_id]) * nfac_on
        firs.append(fir_blank)

        dp_off_byimg.append(dps_vs_off)
        dp_blank_byimg.append(dps_vs_blank)
        fir_byimg.append(firs)

    # Too big: outpk.close()
    return dp_all, dp_off_byimg, dp_blank_byimg, fir_all, fir_byimg, fir_byimg_trials

# ----------------------------------------------------------------------------
# plot_dprime_corr: plot d'(on-vs.-off) vs. d'(on-vs-blanks) correlation plot
#                   using the result of calc_dprime_ch
# note: dp_on and dp_blank should have same size
def plot_dprime_corr(dp_on, dp_blank, fn_out):
    init_rc()
    pl.figure()
    for r in range(len(dp_on)):
        for c in range(1, len(dp_on[r])):
            pl.scatter(dp_on[r][c], dp_blank[r][c], s=1)
    pl.xlabel(r"$d^{'}_\mathrm{off}$")
    pl.ylabel(r"$d^{'}_\mathrm{blanks}$")
    pl.savefig(fn_out)


# ----------------------------------------------------------------------------
def plot_one_PSTH(arr, n_trial, ax, rng=DEF_PSTH_RANGE_MS, xrng=DEF_PSTH_RANGE_MS, \
        aggregate=10, ymax=250, color='#999999', nticks=5, visible=False, txt=None, nondraw=False):
    arr = np.array(arr)
    # plot PSTH
    if n_trial > 0:
        arr = arr/1000.                       # to ms
        interval = rng[1] - rng[0]
        bins = int(interval / aggregate)
        weight = 1000. / aggregate / n_trial  # to convert into Spiks/s.
        weights = np.array([weight] * len(arr))
        if nondraw:
            rtn = np.histogram(arr, range=rng, bins=bins, weights=weights)
            pl.axhline(y=0, color='#333333')
        else:
            rtn = pl.hist(arr, range=rng, bins=bins, weights=weights, fc=color, ec=color)
    # beutify axes
    pl.xlim(xrng)
    pl.ylim([0,ymax])
    ax.xaxis.set_major_locator(pl.MaxNLocator(nticks))
    ax.yaxis.set_major_locator(pl.MaxNLocator(nticks))
    if not visible:
        ax.set_xticklabels([''] * nticks)
        ax.set_yticklabels([''] * nticks)
    pl.axvline(x=0, ymin=0, ymax=1, lw=0.5, color='r')
    if txt != None:
        if nondraw:
            pl.text(xrng[1] - 20, ymax - 70, txt, size=6, va='top', ha='right')
        else:
            pl.text(xrng[1] - 20, ymax - 20, txt, size=6, va='top', ha='right')
    return rtn

# ----------------------------------------------------------------------------
def plot_PSTH(data, outfn, opts={}, verbose=0, ymax=250, evoked=False, sel=None):
    all_spike = data['all_spike']
    actvelecs = sorted(data['actvelecs'])
    all_iids = get_all_iids(all_spike)
    pp = PdfPages(outfn + '.PSTH.pdf')

    init_rc(opts='squeeze')
    blanks = [all_iids[-1]]

    if opts.has_key('blanks'):
        excl_start = int(opts['blanks'].split(',')[0])
        excl_end = int(opts['blanks'].split(',')[1])
        blanks = all_iids[excl_start:excl_end + 1]
    if opts.has_key('ymax'):
        ymax = float(opts['ymax'])
        print '* ymax:', ymax
    if opts.has_key('evoked'):
        evoked = True
        print '* evoked PSTH'
    if opts.has_key('sel'):
        if opts['sel'] == 'none': sel = []
        else:
            sel = [int(el) for el in opts['sel'].split(',')]
        print '* selected:', sel
    if verbose != 0: print '* blanks:', blanks
    if sel is None: sel = actvelecs

    avghist_x = {}
    avghist_y = {}
    blankhist_x = {}
    blankhist_y = {}
    n_elecs = len(actvelecs)
    i_lbl_plot = (int(n_elecs / NCOLS) - 1) * NCOLS

    # plot overall per-channel averaged PSTH first.
    print '* Pass 1: overall plot'
    pl.figure()
    for i_plot, elec_id in enumerate(actvelecs):
        arr, n_trial = get_one_ch(elec_id, all_spike, exclude=blanks)   # get spikes.
        if verbose > 1: print '* len =', len(arr)
        txt = 'E%d (avg)' % elec_id

        ax = pl.subplot(NROWS, NCOLS, i_plot + 1)      # XXX: 10x10 should be changed later!
        if i_plot == i_lbl_plot: visible = True
        else: visible = False

        hinfo = plot_one_PSTH(arr, n_trial, ax, ymax=ymax, visible=visible, txt=txt)
        xarr = []; yarr = []
        for i, y in enumerate(hinfo[0]):
            yarr.append(y); yarr.append(y)
            xarr.append(hinfo[1][i]); xarr.append(hinfo[1][i + 1])
        avghist_x[elec_id] = np.array(xarr)
        avghist_y[elec_id] = np.array(yarr)
    pp.savefig()
    pl.close()

    # plot overall per-channel blank PSTH
    print '* Pass 2: overall evoked plot'
    pl.figure()
    for i_plot, elec_id in enumerate(actvelecs):
        arr, n_trial = get_one_ch(elec_id, all_spike, include=blanks)   # get spikes.
        txt = 'E%d (blanks)' % elec_id

        ax = pl.subplot(NROWS, NCOLS, i_plot + 1)      # XXX: 10x10 should be changed later!
        if i_plot == i_lbl_plot: visible = True
        else: visible = False

        hinfo = plot_one_PSTH(arr, n_trial, ax, ymax=ymax, visible=visible, txt=txt)
        xarr = []; yarr = []
        for i, y in enumerate(hinfo[0]):
            yarr.append(y); yarr.append(y)
            xarr.append(hinfo[1][i]); xarr.append(hinfo[1][i + 1])
        blankhist_x[elec_id] = np.array(xarr)
        blankhist_y[elec_id] = np.array(yarr)
    pp.savefig()
    pl.close()

    # per-channel, per-image PSTH
    print '* Pass 3: per-chanel plot'
    for elec_id in sel:
        pl.figure()
        n_row = int(np.ceil(np.sqrt(len(all_iids))))
        n_col = int(np.ceil(len(all_iids) / float(n_row)))
        for i_plot, iid in enumerate(all_iids):
            arr, n_trial = get_one_ch(elec_id, all_spike, include=[iid])
            txt = 'E%d (%s)' % (elec_id, iid)
            ax = pl.subplot(n_row, n_col, i_plot + 1)
            if i_plot == (n_row - 1) * n_col: visible = True
            else: visible = False
            if evoked:
                # XXX: FIXME: quick-and-dirty..
                hinfo = plot_one_PSTH(arr, n_trial, ax, ymax=ymax, visible=visible, txt=txt, nondraw=True)
                xarr = []; yarr = []
                for i, y in enumerate(hinfo[0]):
                    yarr.append(y); yarr.append(y)
                    xarr.append(hinfo[1][i]); xarr.append(hinfo[1][i + 1])
                xarr = np.array(xarr); yarr = np.array(yarr)
                pl.fill_between(xarr, yarr - blankhist_y[elec_id], color='#999999', facecolor='#999999', lw=0, zorder=0)
                pl.plot(avghist_x[elec_id], avghist_y[elec_id] - blankhist_y[elec_id], color='g', lw=0.5, alpha=0.3)
                pl.xlim(DEF_PSTH_RANGE_MS)
                pl.ylim([-50, ymax - 50])
            else:
                plot_one_PSTH(arr, n_trial, ax, ymax=ymax, visible=visible, txt=txt)
                pl.plot(avghist_x[elec_id], avghist_y[elec_id], color='b', lw=0.5, alpha=0.3)
                pl.plot(blankhist_x[elec_id], blankhist_y[elec_id], color='g', lw=0.5, alpha=0.45)
        pp.savefig()
        pl.close()

    pp.close()

# ----------------------------------------------------------------------------
def main():
    if len(sys.argv) < 4:
        print 'expr_anal.py <ps | dp> <PS firing data pk> <output file prefix> [options]'
        print 'Mode:'
        print '   ps  - plot PSTH, save as .pdf'
        print '         (opts: blanks)'
        print "   dp  - calculate d', save as .csv"
        print '         (opts: blanks, bootstrap, corr, prange, suffix)'
        return

    mode = sys.argv[1]
    fn_pk = sys.argv[2]
    fn_out = sys.argv[3]
    fn_out_suffix = ''
    np.random.seed(DEF_RSEED)

    # parse options
    opts = parse_opts(sys.argv[4:])

    if 'suffix' in opts:
        fn_out_suffix = opts['suffix']

    if 'on_range' in opts:
        rng = opts['on_range'].split(',')
        on_range = [float(rng[0]), float(rng[1])]
        print '* on_range =', on_range
    else:
        on_range = DEF_ON_RANGE_US

    if 'off_range' in opts:
        rng = opts['off_range'].split(',')
        off_range = [float(rng[0]), float(rng[1])]
        print '* off_range =', off_range
    else:
        off_range = DEF_OFF_RANGE_US

    if 'off_num_sort' in opts:
        get_all_iids.num_sort = False
        print '* No num_sort'

    if 'exclude_img' in opts:
        get_all_iids.exclude = set(opts['exclude_img'].split(','))
        print '* Exclude =', get_all_iids.exclude

    # load input pickle file
    data = pk.load(open(fn_pk))
    all_spike = data['all_spike']
    all_iids = get_all_iids(all_spike, as_string=True)

    # PSTH for each channel -----------------
    if mode == 'ps':
        plot_PSTH(data, fn_out, opts=opts, verbose=1)

    # d' for each channel -------------------
    elif mode == 'dp':
        # calculate d' for each channel
        dp_all, dp_off_byimg, dp_blank_byimg, fir_all, fir_byimg, fir_byimg_trials = calc_dprime_ch(data, opts=opts, verbose=1, on_range=on_range, off_range=off_range)

        # write down to the csv file
        write_all_to_csv(dp_all, fname=fn_out + '.dp_all' + fn_out_suffix + '.csv', header=['elec_id', "d'(on-vs-off)", "d'(on-vs-blank)"])
        write_all_to_csv(dp_off_byimg, fname=fn_out + '.dp_off' + fn_out_suffix + '.csv', header=['elec_id'] + all_iids)
        write_all_to_csv(dp_blank_byimg, fname=fn_out + '.dp_blank' + fn_out_suffix + '.csv', header=['elec_id'] + all_iids)
        write_all_to_csv(fir_all, fname=fn_out + '.fir_all' + fn_out_suffix + '.csv', header=['elec_id', "firrate_on", "firrate_off", "firrate_blanks"])
        write_all_to_csv(fir_byimg, fname=fn_out + '.fir_byimg' + fn_out_suffix + '.csv', header=['elec_id'] + all_iids + ['blanks'])

        # if want to save correlation plot
        if opts.has_key('corr'):
            print '* save correlation plot'
            plot_dprime_corr(dp_off_byimg, dp_blank_byimg, fn_out + '.dp' + fn_out_suffix + '.pdf')

        if opts.has_key('extfirinfo'):
            outpk = open(fn_out + '.fir_byimg_trials' + fn_out_suffix + '.pk', 'wb')
            pk.dump(fir_byimg_trials, outpk)
            pk.dump({'on_rage':on_range, 'off_range':off_range}, outpk)
            outpk.close()

    # -----------------
    else:
        print 'Unsupported mode.'


if __name__ == '__main__':
    main()
