#!/usr/bin/env python

import numpy as np
import cPickle as pk
import tables as tbl
import sys
import warnings
sys.path.append('lib')
from common_fn import getspk, C_SUCCESS, T_SUCCESS, \
        T_START, T_STOP, REJECT_SLOPPY, makeavail, parse_opts_adapter

DEFAULT_ELECS = range(1, 97)
DEFAULT_SAMPLES_PER_SPK = 48
DEFAULT_MAX_SPKS = 50000000
PROC_CLUSTER = False
# max number of clusters per channel
MAX_CLUS = 5

# shifting channels based on rules:
#   CH_SHIFT[rule_name] = {src_1_based_ch:new_1_based_ch}
CH_SHIFT = {}
CH_SHIFT[None] = None
# for 1-to-1 cards
CH_SHIFT['1to1'] = {}
for ch1 in xrange(1, 49):
    CH_SHIFT['1to1'][ch1] = ch1
for ch1 in xrange(81, 129):
    CH_SHIFT['1to1'][ch1] = ch1 - 32

# for 20110720A: assign all 40 A channels to 1-40
# and all 70 M channels to 41-110
CH_SHIFT['20110720A'] = {1: 41, 2: 42, 3: 43, 4: 44, 5: 45, 6: 46, \
        7: 47, 8: 48, 9: 49, 10: 50, 11: 51, 12: 52, 13: 53, 14: 54, \
        15: 55, 16: 56, 17: 57, 18: 58, 19: 59, 20: 60, 21: 61, \
        22: 62, 23: 63, 24: 64, 25: 65, 26: 66, 27: 67, 28: 68, \
        29: 69, 30: 70, 31: 71, 32: 72, 33: 73, 34: 74, 35: 75, \
        44: 1, 45: 2, 46: 3, 47: 4, 48: 5, 49: 6, 50: 7, 51: 8, \
        52: 9, 53: 10, 54: 11, 55: 12, 56: 13, 57: 14, 58: 15, \
        59: 16, 60: 17, 61: 18, 62: 19, 63: 20, 64: 21, 65: 22, \
        66: 23, 67: 24, 68: 25, 69: 26, 70: 27, 71: 28, 72: 29, \
        73: 30, 74: 31, 75: 32, 76: 33, 77: 34, 78: 35, 79: 36, \
        80: 37, 81: 38, 82: 39, 83: 40, 94: 76, 95: 77, 96: 78, \
        97: 79, 98: 80, 99: 81, 100: 82, 101: 83, 102: 84, 103: 85, \
        104: 86, 105: 87, 106: 88, 107: 89, 108: 90, 109: 91, \
        110: 92, 111: 93, 112: 94, 113: 95, 114: 96, 115: 97, \
        116: 98, 117: 99, 118: 100, 119: 101, 120: 102, 121: 103, \
        122: 104, 123: 105, 124: 106, 125: 107, 126: 108, \
        127: 109, 128: 110}


# ----------------------------------------------------------------------------
def get_firrate(fn_mwk, fn_out, proc_cluster=PROC_CLUSTER, max_clus=MAX_CLUS, \
        movie_begin_fname=None, **kwargs):
    """Get spiking data around stimuli presented."""

    # some housekeeping things...
    kwargs['verbose'] = 2
    t_start0 = kwargs['t_start0']
    t_stop0 = kwargs['t_stop0']

    # all_spike[chn_id][img_id]: when the neurons spiked?
    all_spike = {}
    all_foffset = {}
    clus_info = {}

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

            # process movie if requested
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
            actvunits = {}
            t_rel = {}
            foffset = {}

            for ch in actvelecs:
                # if no clustering info is used...
                if not proc_cluster:
                    t_rel[ch] = []
                    foffset[ch] = []
                    continue
                # if clustering info is used...
                cids = range(max_clus)
                actvunits[ch] = cids
                for cid in cids:
                    t_rel[(ch, cid)] = []
                    foffset[(ch, cid)] = []

        # -- put actual spiking info
        elif info['type'] == 'spike':
            ch = info['ch']

            if proc_cluster:
                cid = info['cluster_id']
                key = (ch, cid)
            else:
                key = ch

            t_rel[key].append(int(info['t_rel']))
            foffset[key].append(int(info['pos']))
            # update the clus_info and n_cluster
            if proc_cluster and key not in clus_info:
                cvalue = info['cluster_value']
                clus_info[key] = cvalue
                if cvalue['nclusters'] > max_clus:
                    raise ValueError, '*** Unit %s: max_cluster(=%d) is '\
                            'smaller than the actual number of clusters(=%d)!'\
                            % (str(key), max_clus, cvalue['nclusters'])

        # -- finalize info for the image
        elif info['type'] == 'end':
            for el in actvelecs:
                if proc_cluster:
                    cids = actvunits[el]
                else:
                    cids = [-1]
                for cid in cids:
                    if proc_cluster:
                        key = (el, cid)
                    else:
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
    f = open(fn_out, 'w')
    out = {'all_spike': all_spike,
           't_start': t_start0,
           't_stop': t_stop0,
           't_adjust': t_adjust,
           'actvelecs': actvelecs}
    if proc_cluster:
        out['clus_info'] = clus_info
        out['max_clus'] = max_clus
    if movie_begin_fname is not None:
        out['frame_onset'] = frame_onset
    pk.dump(out, f)

    # put all_foffset into the 2nd half to speed up reading
    out2 = {'all_foffset': all_foffset}
    pk.dump(out2, f)

    f.close()


# ----------------------------------------------------------------------------
def get_waveform(fn_mwk, fn_nev, fn_out, movie_begin_fname=None, \
        n_samples=DEFAULT_SAMPLES_PER_SPK, n_max_spks=DEFAULT_MAX_SPKS, \
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
    # these two are unused
    kwargs.pop('proc_cluster', None)
    kwargs.pop('max_clus', None)

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
                warnings.warn('n_spks exceedes n_spks_lim! '\
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
    CMsnp = h5o.createCArray(h5o.root, 'Msnp', t_int16, \
            Msnp.shape, filters=filters)
    CMsnp_tabs = h5o.createCArray(h5o.root, 'Msnp_tabs', t_uint64, \
            Msnp_tabs.shape, filters=filters)
    CMsnp_ch = h5o.createCArray(h5o.root, 'Msnp_ch', t_uint32, \
            Msnp_ch.shape, filters=filters)
    CMsnp_pos = h5o.createCArray(h5o.root, 'Msnp_pos', t_uint64, \
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

    proc_cluster = PROC_CLUSTER
    if 'proc_cluster' in opts or 'proc_spksorting' in opts:
        print '* Collecting spike sorting information'
        proc_cluster = True

    max_clus = MAX_CLUS
    if 'max_cluster' in opts:
        max_clus = int(opts['max_cluster'])
        print '* Maximum number of clusters per channel:', max_clus

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
            'proc_cluster': proc_cluster, 'max_clus': max_clus,
            't_start0': t_start0, 't_stop0': t_stop0,
            'reject_sloppy': reject_sloppy,
            'exclude_img': exclude_img,
            'movie_begin_fname': movie_begin_fname,
            'ign_unregistered': ign_unregistered,
            'ch_shift': CH_SHIFT[ch_shift]
            }
    if mode == 'firrate':
        get_firrate(fn_mwk, fn_out, **kwargs)
    elif mode == 'wav':
        get_waveform(fn_mwk, fn_nev, fn_out, **kwargs)
    else:
        raise ValueError('Invalid mode')

    print 'Done.                                '


if __name__ == '__main__':
    main()
