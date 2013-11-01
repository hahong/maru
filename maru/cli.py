#!/usr/bin/env python
import warnings
import os
from .utils import parse_opts_adapter, parse_opts2

MERGE_USAGE = \
"""
${PNAME} [options] <mwk> <nev/plx>
Aligns and merges mwk and nev/plx files.

Options:
   --procwav                include waveform data (not recommended)
   --code_sent=<code>       specify the code name for sending out timestamps
   --code_recv=<code>       specify the code name for returned timestamp bits
   --force_timetransf=<slope,intercept,delay>
                            use the supplied parameters for merging
"""

PSINFO_USAGE = \
"""
${PNAME} [options] <mwk> <output>
Collects spike timing/snippets around visual stimuli

Options:
   --c_success=<code name>   code name for "success" signal
   --delay=<# in us>         override delay in us
   --nelec=#                 set the number of electrodes
   --t_start=<# in us>       set the beginning of the peristimulus analysis
   --t_stop=<# in us>        set the end time of the peristimulus analysis
   --reject_sloppy           reject sloppy image presentations
   --movie_begin_fname       set the prefix of the movie stimuli
   --ign_unregistered        ignore unregistered channels
   --ch_shift=<rule name>    shift channels based on the given rule
   --exclude_img=<img1,...>  exclude unwanted images.  Should be a comma-
                             separated list
   --extinfo                 collects extra stimuli information in addition to
                             the names
   --wav=<.nev file name>    collects waveform for **spike sorting** with the
                             .nev file.  This produces a specialized outputs.
   --pkl                     save pickle format as well (HIGHLY DISCOURAGED,
                             and "--wav" does not support this)
"""

SPKSORT_USAGE = \
"""${PNAME}: a spike sorting swiss army knife

Feature Computing Mode
======================
Computes feautures of spikes for later clustering after the
re-thresholding (Quiroga et al., 2004) and the spike alignment.

${PNAME} feature [options] <input.psf.h5> <output.c1.feat.h5>

Options:
   --with=<reference.h5> Use the parameters used in the reference file
                         (another feature output.c1.feat.h5 file).
   --rethreshold_mult=#  The multiplier for Quiroga's thresholding method
   --align_subsmp=#
   --align_maxdt=#
   --align_peakloc=#
   --align_findbwd=#
   --align_findfwd=#
   --align_cutat=#
   --align_outdim=#
   --feat_metd=<str>     The method used to extract features.  Available:
                         wavelet
   --njobs=#             The number of worker processes


Clustering Mode
===============
Cluster spikes using pre-computed features.

${PNAME} cluster [options] <input.c1.feat.h5> <output.c2.clu.h5>

Options:
   --cluster_alg=<str>   The clustering algorithm to use.  Avaliable:
                         affinity_prop
   --feat_kssort=<bool>
   --feat_outdim=#
   --feat_wavelet_lev=#
   --skimspk_tb=#        Beginning relative time for collecting examplar spikes
   --skimspk_te=#        End time for examplar spikes
   --extract_nperimg=#   The max number of spikes collected for each stimulus
   --qc_minsnr=#         Minimum SNR to be qualified as a cluster
   --qc_ks_plevel=#      Desired significance level for the KS-test
   --njobs=#             The number of worker processes
   --ref=<reference.h5>  Use the specified file for clustering


Collation Mode
==============
Find out common clusters in multiple .c2.clu.h5 files and save the final
sorted spikes as .psf.h5 format.

***NOTE: multiple files are not supported yet!!!***

${PNAME} collate [options] <input1.c2.clu.h5> <output.psf.h5>
"""

PSINFO2FEAT_USAGE = \
"""
Converts peristimulus firing information (in .psf.h5) into a feature matrix
by counting the total number of spikes in time bins.

${PNAME} tbte <tinfo.h5> <output_repr.h5> <t_begins> <t_ends> [opts]
  t_begins: beginnings of reading window (0=stim onset), comma-separated list
  t_ends: ends of reading window, comma-separated, length must be same
          as t_begins

${PNAME} twmp <tinfo.h5> <output_repr.h5> <delta_t> <midpt> [opts]
  delta_t: reading window width, comma-separated list
  midpt: mid-point of window, comma-separated

Notes:
All time values are in ms.

Options:
  --preservefmt    do not compact file
"""


def merge_main(args):
    from .merge import (merge, DEF_CODE_SENT, DEF_CODE_RECV,
        DEF_PROC_WAV, DEF_ADJ_REJECT, DEF_AMP_REJECT, DEF_TIMETRANSF)
    from .utils import set_new_threshold_rng

    pname = get_entry_name(args[0])
    args = args[1:]

    if len(args) < 2:
        print MERGE_USAGE.replace('${PNAME}', pname)
        return 1

    warnings.simplefilter('once')
    args, opts = parse_opts_adapter(args, 2)
    fn_mwk, fn_neu = args[:2]

    proc_wav = DEF_PROC_WAV
    # -- parse options
    if 'nowav' in opts:
        # for backward compatibility
        print '* No waveform processing'
        proc_wav = False
    if 'procwav' in opts:
        print '* Including waveform data'
        proc_wav = True

    code_sent = DEF_CODE_SENT
    if 'code_sent' in opts:
        code_sent = opts['code_sent']
        print '* code_sent =', code_sent

    code_recv = DEF_CODE_RECV
    if 'code_recv' in opts:
        code_recv = opts['code_recv']
        print '* code_recv =', code_recv

    adj_reject = DEF_ADJ_REJECT
    amp_reject = DEF_AMP_REJECT
    if 'adj_reject' in opts:
        adj_reject = float(opts['adj_reject'])
        amp_reject = set_new_threshold_rng
        print '* Amplitude rejection multiplier =', adj_reject

    _force_timetransf = DEF_TIMETRANSF
    if 'force_timetransf' in opts:
        a, b, delay = [float(e) for e in opts['force_timetransf'].split(',')]
        _force_timetransf = {}
        _force_timetransf['C'] = [a, b]
        _force_timetransf['delay'] = delay
        print '* Time transformation =', _force_timetransf

    if merge(fn_mwk, fn_neu, proc_wav=proc_wav,
        code_sent=code_sent, code_recv=code_recv,
        adj_reject=adj_reject, amp_reject=amp_reject,
        _force_timetransf=_force_timetransf):
        return 0
    return 1


def psinfo_main(args):
    from .utils import CH_SHIFT
    from .peristiminfo import (DEFAULT_ELECS, C_SUCCESS, T_SUCCESS, T_START,
            T_STOP, REJECT_SLOPPY, get_PS_firrate, get_PS_waveform)
    pname = get_entry_name(args[0])
    args, opts = parse_opts_adapter(args[1:], 4)

    if len(args) < 2:
        print PSINFO_USAGE.replace('${PNAME}', pname)
        return 1

    fn_mwk = args[0]
    fn_out = args[1]
    fn_nev = None

    # -- parsing extra arguments (mainly for backward compatibility)
    if len(args) >= 3:
        override_delay_us = long(args[2])
        if override_delay_us < 0:
            override_delay_us = None
        print '* Delay override:', override_delay_us
    else:
        override_delay_us = None

    if len(args) >= 4:
        override_elecs = range(1, int(args[3]) + 1)
        print '* Active electrodes override: [%d..%d]' % \
                (min(override_elecs), max(override_elecs))
    else:
        override_elecs = DEFAULT_ELECS

    # -- handle "opts"
    mode = 'firrate'

    if 'wav' in opts:
        mode = 'wav'
        fn_nev = opts['wav']
        print '* Collecting waveforms for later spike sorting with', fn_nev

    if 'delay' in opts:
        override_delay_us = long(opts['delay'])
        if override_delay_us < 0:
            override_delay_us = None
        print '* Delay override:', override_delay_us

    if 'nelec' in opts:
        override_elecs = range(1, int(opts['nelec']) + 1)
        print '* Active electrodes override: [%d..%d]' % \
                (min(override_elecs), max(override_elecs))

    extinfo = False
    if 'extinfo' in opts:
        extinfo = True
        print '* Collecting extra information of the stimuli'

    if 'c_success' in opts:
        c_success = opts['c_success']
        print '* c_success:', c_success
    else:
        c_success = C_SUCCESS

    if 't_success' in opts:
        t_success = int(opts['t_success'])
        print '* t_success:', t_success
    else:
        t_success = T_SUCCESS

    t_start0 = T_START
    if 't_start' in opts:
        t_start0 = int(opts['t_start'])
        print '* t_start =', t_start0

    t_stop0 = T_STOP
    if 't_stop' in opts:
        t_stop0 = int(opts['t_stop'])
        print '* t_stop =', t_stop0

    reject_sloppy = REJECT_SLOPPY
    if 'reject_sloppy' in opts:
        reject_sloppy = True
        print '* Rejecting sloppy stimuli'

    exclude_img = None
    if 'exclude_img' in opts:
        exclude_img = opts['exclude_img'].split(',')
        print '* Exclude unwanted images:', exclude_img

    movie_begin_fname = None
    if 'movie_begin_fname' in opts:
        movie_begin_fname = opts['movie_begin_fname']
        print '* movie_begin_fname:', movie_begin_fname

    ign_unregistered = False
    if 'ign_unregistered' in opts:
        ign_unregistered = True
        print '* Ignore unregistered keys'

    ch_shift = None
    if 'ch_shift' in opts:
        ch_shift = opts['ch_shift']
        print '* Shifting based on this rule:', ch_shift

    save_pkl = False
    if 'save_pkl' in opts or 'pkl' in opts:
        save_pkl = True
        print '* Save pickle format'

    # -- go go go
    kwargs = {'override_delay_us': override_delay_us,
            'override_elecs': override_elecs,
            'extinfo': extinfo, 'c_success': c_success,
            't_success_lim': t_success,
            't_start0': t_start0, 't_stop0': t_stop0,
            'reject_sloppy': reject_sloppy,
            'exclude_img': exclude_img,
            'movie_begin_fname': movie_begin_fname,
            'ign_unregistered': ign_unregistered,
            'ch_shift': CH_SHIFT[ch_shift]
            }
    if mode == 'firrate':
        get_PS_firrate(fn_mwk, fn_out, save_pkl=save_pkl, **kwargs)
    elif mode == 'wav':
        get_PS_waveform(fn_mwk, fn_nev, fn_out, **kwargs)
    else:
        raise ValueError('Invalid mode')

    print 'Done.                                '
    return 0


def spksort_main(args):
    from .spksort import get_features, cluster, collate

    print
    print '*********************************************'
    print '*** Many parts are not yet implemented!!  ***'
    print '*********************************************'
    print
    pname = get_entry_name(args[0])
    args, opts = parse_opts2(args[1:])
    if len(args) < 3:
        print SPKSORT_USAGE.replace('${PNAME}', pname)
        return 1
    mode = args[0]

    # -- parsing extra arguments (mainly for backward compatibility)
    if mode == 'feature':
        get_features(args[1], args[2], opts)
    elif mode == 'cluster':
        cluster(args[1], args[2], opts)
    elif mode == 'collate':
        collate(args[1], args[2], opts)
    else:
        raise ValueError('Invalid mode')

    print 'Done.                                '
    return 0


def psinfo2feat_main(argv):
    from .io import tinfo
    pname = get_entry_name(argv[0])
    args, opts = parse_opts2(argv[1:])

    if len(args) != 5:
        print PSINFO2FEAT_USAGE.replace('${PNAME}', pname)
        return 1

    mode, ifn, ofn, t1, t2 = args
    t1 = [int(t) for t in t1.split(',')]
    t2 = [int(t) for t in t2.split(',')]

    if mode == 'tbte':
        tbs = t1
        tes = t2
        nooversized = False
    elif mode == 'twmp':
        tbs = []
        tes = []
        for dt in t1:
            dt2 = dt / 2
            for mp in t2:
                tbs.append(mp - dt2)
                tes.append(mp + dt2)
        nooversized = True
    else:
        print '* unrecognized mode "%s"' % mode
        return

    preservefmt = False
    if 'preservefmt' in opts:
        print '* Preserve format'
        preservefmt = True

    assert len(tbs) == len(tes)
    tinfo.tinfo2feat(ifn, ofn, tbs, tes, nooversized=nooversized,
            preservefmt=preservefmt)
    return 0


def get_entry_name(arg0):
    return os.path.basename(arg0)
