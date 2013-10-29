#!/usr/bin/env python

import numpy as np
import cPickle as pk
import tables as tbl
import sys
import warnings
from .utils import (getspk, C_SUCCESS, T_SUCCESS, T_START, T_STOP,
        REJECT_SLOPPY, makeavail, parse_opts_adapter, DEFAULT_ELECS,
        DEFAULT_SAMPLES_PER_SPK, DEFAULT_MAX_SPKS, CH_SHIFT)


# ----------------------------------------------------------------------------
def get_PS_firrate(fn_mwk, fn_out,
        movie_begin_fname=None, **kwargs):
    """Get spiking data around stimuli presented."""

    # some housekeeping things...
    kwargs['verbose'] = 2
    t_start0 = kwargs['t_start0']
    t_stop0 = kwargs['t_stop0']

    # all_spike[chn_id][img_id]: when the neurons spiked?
    all_spike = {}
    all_foffset = {}

    frame_onset = {}
    movie_iid = None
    movie_onsets = []
    movie_onset0 = 0

    for info in getspk(fn_mwk, fn_nev=None, **kwargs):
        # -- get the metadata. this must be called before other clauses
        if info['type'] == 'preamble':
            actvelecs = info['actvelecs']
            t_adjust = info['t_adjust']

        # -- do some housekeeping things once per each img
        elif info['type'] == 'begin':
            t0 = info['t_imgonset']
            iid = info['imgid']

            # process movie if requested -- tested weakly. expect bugs!
            if movie_begin_fname is not None:
                # begin new clip?
                if movie_begin_fname in iid:
                    # was there previous clip?
                    if movie_iid is not None:
                        if movie_iid not in frame_onset:
                            frame_onset[movie_iid] = []
                        frame_onset[movie_iid].append(movie_onsets)
                    # init for new clip
                    movie_onsets = []
                    iid = movie_iid = iid.replace(movie_begin_fname, '')
                    movie_onset0 = t0
                    movie_onsets.append(0)
                elif movie_iid is not None:
                    movie_onsets.append(t0 - movie_onset0)
                    continue

            # prepare the t_rel & foffset
            t_rel = {}
            foffset = {}

            for ch in actvelecs:
                t_rel[ch] = []
                foffset[ch] = []

        # -- put actual spiking info
        elif info['type'] == 'spike':
            ch = info['ch']
            key = ch

            t_rel[key].append(int(info['t_rel']))
            foffset[key].append(int(info['pos']))

        # -- finalize info for the image
        elif info['type'] == 'end':
            for el in actvelecs:
                key = el
                if key not in all_spike:
                    # not using defaultdict here:
                    # all_spike[key] = defaultdict(list)
                    all_spike[key] = {}
                    all_foffset[key] = {}
                if iid not in all_spike[key]:
                    all_spike[key][iid] = []
                    all_foffset[key][iid] = []
                all_spike[key][iid].append(t_rel[key])
                all_foffset[key][iid].append(foffset[key])

    # -- done!
    # flush movie data
    if movie_iid is not None:
        if movie_iid not in frame_onset:
            frame_onset[movie_iid] = []
        frame_onset[movie_iid].append(movie_onsets)

    # finished calculation....
    out = {'all_spike': all_spike,
           't_start': t_start0,
           't_stop': t_stop0,
           't_adjust': t_adjust,
           'actvelecs': actvelecs,
           'all_foffset': all_foffset,
           'frame_onset': frame_onset}

    return out


# ----------------------------------------------------------------------------
def get_PS_waveform(fn_mwk, fn_nev, fn_out, movie_begin_fname=None,
        n_samples=DEFAULT_SAMPLES_PER_SPK, n_max_spks=DEFAULT_MAX_SPKS,
        **kwargs):
    """Get waveform data around stimuli presented for later spike sorting.
    This will give completely different output file format.
    NOTE: this function is memory intensive!  Will require approximately
    as much memory as the size of the files."""

    # -- some housekeeping things...
    kwargs['verbose'] = 2
    kwargs['only_new_t'] = True
    t_start0 = kwargs['t_start0']
    t_stop0 = kwargs['t_stop0']

    iid2idx = {}
    idx2iid = []
    ch2idx = {}
    idx2ch = []
    n_spks = 0

    # does "n_spks_lim" reach "n_max_spks"?
    b_warn_max_spks_lim = False
    # list of image presentations without spikes
    l_empty_spks = []

    for info in getspk(fn_mwk, fn_nev=fn_nev, **kwargs):
        # -- get the metadata. this must be called before other clauses
        if info['type'] == 'preamble':
            actvelecs = info['actvelecs']
            t_adjust = info['t_adjust']
            chn_info = info['chn_info']
            n_spks_lim = min(info['n_packets'], n_max_spks)
            print '* n_spks_lim =', n_spks_lim

            for ch in sorted(actvelecs):
                makeavail(ch, ch2idx, idx2ch)

            # Data for snippets ===
            # Msnp: snippet data
            # Msnp_tabs: when it spiked (absolute time)
            # Msnp_ch: which channel ID spiked?
            # Msnp_pos: corresponding file position
            Msnp = np.empty((n_spks_lim, n_samples), dtype='int16')
            Msnp_tabs = np.empty(n_spks_lim, dtype='uint64')
            Msnp_ch = np.empty(n_spks_lim, dtype='uint32')
            Msnp_pos = np.empty(n_spks_lim, dtype='uint64')

            # Data for images ===
            # Mimg: image indices in the order of presentations
            # Mimg_tabs: image onset time (absolute)
            Mimg = []
            Mimg_tabs = []

        # -- do some housekeeping things once per each img
        if info['type'] == 'begin':
            t_abs = info['t_imgonset']
            iid = info['imgid']
            i_img = info['i_img']

            makeavail(iid, iid2idx, idx2iid)
            Mimg.append(iid2idx[iid])
            Mimg_tabs.append(t_abs)
            b_no_spks = True

            # process movie if requested
            if movie_begin_fname is not None:
                raise NotImplementedError('Movies are not supported yet.')

        # -- put actual spiking info
        elif info['type'] == 'spike':
            wav = info['wavinfo']['waveform']
            t_abs = info['t_abs']
            i_ch = ch2idx[info['ch']]
            pos = info['pos']

            Msnp[n_spks] = wav
            Msnp_tabs[n_spks] = t_abs
            Msnp_ch[n_spks] = i_ch
            Msnp_pos[n_spks] = pos
            b_no_spks = False

            n_spks += 1
            if n_spks >= n_spks_lim:
                warnings.warn('n_spks exceedes n_spks_lim! '
                    'Aborting further additions.')
                b_warn_max_spks_lim = True
                break

        elif info['type'] == 'end':
            if not b_no_spks:
                continue
            # if there's no spike at all, list the stim
            warnings.warn('No spikes are there!       ')
            l_empty_spks.append(i_img)

    # -- done!
    # finished calculation....
    Msnp = Msnp[:n_spks]
    Msnp_tabs = Msnp_tabs[:n_spks]
    Msnp_ch = Msnp_ch[:n_spks]
    Msnp_pos = Msnp_pos[:n_spks]

    Mimg = np.array(Mimg, dtype='uint32')
    Mimg_tabs = np.array(Mimg_tabs, dtype='uint64')

    filters = tbl.Filters(complevel=4, complib='blosc')
    t_int16 = tbl.Int16Atom()
    t_uint32 = tbl.UInt32Atom()
    t_uint64 = tbl.UInt64Atom()

    h5o = tbl.openFile(fn_out, 'w')
    CMsnp = h5o.createCArray(h5o.root, 'Msnp', t_int16,
            Msnp.shape, filters=filters)
    CMsnp_tabs = h5o.createCArray(h5o.root, 'Msnp_tabs', t_uint64,
            Msnp_tabs.shape, filters=filters)
    CMsnp_ch = h5o.createCArray(h5o.root, 'Msnp_ch', t_uint32,
            Msnp_ch.shape, filters=filters)
    CMsnp_pos = h5o.createCArray(h5o.root, 'Msnp_pos', t_uint64,
            Msnp_pos.shape, filters=filters)

    CMsnp[...] = Msnp
    CMsnp_tabs[...] = Msnp_tabs
    CMsnp_ch[...] = Msnp_ch
    CMsnp_pos[...] = Msnp_pos

    h5o.createArray(h5o.root, 'Mimg', Mimg)
    h5o.createArray(h5o.root, 'Mimg_tabs', Mimg_tabs)

    meta = h5o.createGroup('/', 'meta', 'Metadata')
    h5o.createArray(meta, 't_start0', t_start0)
    h5o.createArray(meta, 't_stop0', t_stop0)
    h5o.createArray(meta, 't_adjust', t_adjust)
    h5o.createArray(meta, 'chn_info_pk', pk.dumps(chn_info))
    h5o.createArray(meta, 'kwargs_pk', pk.dumps(kwargs))

    h5o.createArray(meta, 'idx2iid', idx2iid)
    h5o.createArray(meta, 'iid2idx_pk', pk.dumps(iid2idx))
    h5o.createArray(meta, 'idx2ch', idx2ch)
    h5o.createArray(meta, 'ch2idx_pk', pk.dumps(ch2idx))

    # some error signals
    h5o.createArray(meta, 'b_warn_max_spks_lim', b_warn_max_spks_lim)
    if len(l_empty_spks) > 0:
        h5o.createArray(meta, 'l_empty_spks', l_empty_spks)

    h5o.close()


# ----------------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print 'collect_PS_firing.py [options] <mwk> <output> ' \
                '[override delay in us] [number of electrodes]'
        print 'Collects spike timing around visual stimuli'
        print
        print 'Options:'
        print '   --wav=<.nev file name>  - collects waveform for spike' \
                ' sorting with the .nev file.'
        print '   --extinfo               - collects extra stimuli info' \
                'rmation in addition to the names'
        print '   --c_success=<code name> - code name for "success" signal'
        print '   --proc_cluster          - process extra spike sorting info' \
                'rmation'
        print '   --max_cluster=#         - maximum number of clusters per' \
                ' channel'
        return

    args, opts = parse_opts_adapter(sys.argv[1:], 4)
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
        get_PS_firrate(fn_mwk, fn_out, **kwargs)
    elif mode == 'wav':
        get_PS_waveform(fn_mwk, fn_nev, fn_out, **kwargs)
    else:
        raise ValueError('Invalid mode')

    print 'Done.                                '
