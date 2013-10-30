#!/usr/bin/env python
import numpy as np
import cPickle as pk
import tables as tbl
import sys
import warnings
import os
from multiprocessing import Process, Queue
from collections import namedtuple
from pymario.brreader import BRReader
from pymario.plxreader import PLXReader
from pymworks.data import MWKFile
from .utils import makeavail, set_new_threshold
from .merge import Merge
from .io.tinfo import save_tinfo

C_STIM = '#announceStimulus'
C_MSG = '#announceMessage'
# by default, do not reject sloppy (time to present > 2 frames) stimuli
REJECT_SLOPPY = False
ERR_UTIME_MSG = 'updating main window display is taking longer than two frames'
# Errors are type 2 in #aanounceStimulus
ERR_UTIME_TYPE = 2

T_START = -100000
T_STOP = 250000
# one visual stimulus should have one "success" event within
C_SUCCESS = 'number_of_stm_shown'
# 250ms time window in order to be considered as valid one.
T_SUCCESS = 250000

# if the animal is not working longer than 3s, then don't do readahead
DIFF_CUTOFF = 3000000
# maximum readahead: 6s
MAX_READAHEAD = 6000000

DEFAULT_ELECS = range(1, 97)
DEFAULT_SAMPLES_PER_SPK = 48
DEFAULT_MAX_SPKS = 50000000


# ----------------------------------------------------------------------------
def get_PS_firrate(fn_mwk, fn_out,
        movie_begin_fname=None, save_pkl=False, **kwargs):
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
    if save_pkl:
        # old-format: slow and inefficient and not supported
        pk.dump(out, open(fn_out, 'wb'))
    else:
        save_tinfo(fn_out, 'test2.h5')


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
# Other support functions
def xget_events(mf, **kwargs):
    """Memory saving trick"""
    def xloader(q, mf, kwargs):
        evs = mf.get_events(**kwargs)
        q.put([(ev.value, ev.time, ev.empty) for ev in evs])

    q = Queue()
    p = Process(target=xloader, args=(q, mf, kwargs))
    p.start()
    evs = q.get()
    p.join()

    Event = namedtuple('Event', 'value time empty')
    return [Event(*ev) for ev in evs]


def xget_events_readahead(mf, code, time_range, readahead=15000000,
        peeklast=False):
    """Buffered MWKFile handler.
    TODO: should be re-implemented as a class!"""
    tstart0, tend0 = time_range
    assert tstart0 < tend0
    _tr = xget_events_readahead.trange
    _ra = xget_events_readahead.readahead
    _rat = xget_events_readahead.readahead_t
    tstart = tstart0
    tend = tend0 + readahead

    if code not in _tr:
        _tr[code] = (-1, -1)

    # if _ra doesn't have the full time_range, read ahead
    if _tr[code][0] > tstart0 or _tr[code][1] < tend0:
        _ra[code] = xget_events(mf, codes=[code], time_range=[tstart, tend])
        _rat[code] = [e.time for e in _ra[code]]
        _tr[code] = (tstart, tend)
        # DBG print '** len =', len(_rat[code])

    # get the corresponding slice
    ib = np.searchsorted(_rat[code], tstart0)                # begin
    ie = np.searchsorted(_rat[code], tend0, side='right')    # end
    # DBG print '#<', np.sum(np.array(_rat[code][ib:ie]) < tstart0)
    # DBG print '#>', np.sum(np.array(_rat[code][ib:ie]) > tend0)

    if peeklast:
        last = _ra[code][-1] if len(_ra[code]) > 1 else None
        return _ra[code][ib:ie], last
    else:
        return _ra[code][ib:ie]
xget_events_readahead.trange = {}
xget_events_readahead.readahead = {}
xget_events_readahead.readahead_t = {}


def load_spike_data(neu_filename):
    ext = os.path.splitext(neu_filename)[1]
    if ext.lower() == '.nev':
        nf = BRReader(neu_filename)
    else:
        nf = PLXReader(neu_filename)
    return nf


def get_stim_info(mf, c_stim=C_STIM, extinfo=False,
        dynstim='ds_9999', rotrmn=180, blank='zb_99999',
        exclude_img=None):
    stims0 = xget_events(mf, codes=[c_stim])
    stims = []
    for x in stims0:
        if type(x.value) == int:
            continue
        if x.value['type'] == 'image':
            if exclude_img is not None:
                ignore = False
                for patt in exclude_img:
                    # if it's in the excluded list, don't put that one!!
                    if patt in x.value['name']:
                        ignore = True
                        break
                if ignore:
                    continue
            stims.append(x)
        elif x.value['type'] == 'dynamic_stimulus' or \
                x.value['type'] == 'blankscreen' or \
             x.value['type'] == 'drifting_grating':
            stims.append(x)
        elif x.value['type'] == 'image_directory_movie':
            if type(x.value['current_stimulus']) == int:
                continue
            stims.append(x)
        # otherwise, ignore

    # when the stimulus was shown? (in us).
    img_onset = [x.time for x in stims]
    # ..and each corresponding image id
    # img_id = [int(x.value['name'].split('_')[1]) for x in stims]
    img_id = []
    for x in stims:
        if x.value['type'] == 'image':
            if not extinfo:
                iid = x.value['name']
            else:
                iid = (x.value['name'],
                    x.value['pos_x'], x.value['pos_y'],
                    x.value['rotation'],
                    x.value['size_x'], x.value['size_y'])
        elif x.value['type'] == 'dynamic_stimulus' or \
                x.value['type'] == 'drifting_grating':
            if not extinfo:
                iid = dynstim
            else:
                iid = (dynstim,
                    x.value['xoffset'], x.value['yoffset'],
                    x.value['rotation'] % rotrmn,
                    x.value['width'], x.value['height'])
        elif x.value['type'] == 'image_directory_movie':
            if not extinfo:
                iid = x.value['current_stimulus']['name']
            else:
                iid = (x.value['current_stimulus']['name'],
                    x.value['current_stimulus']['pos_x'],
                    x.value['current_stimulus']['pos_y'],
                    x.value['current_stimulus']['rotation'],
                    x.value['current_stimulus']['size_x'],
                    x.value['current_stimulus']['size_y'])
        elif x.value['type'] == 'blankscreen':
            if not extinfo:
                iid = blank
            else:
                iid = (blank, 0, 0, 0, 0, 0)
        img_id.append(iid)

    assert (len(img_id) == len(img_onset))
    return img_onset, img_id


# ----------------------------------------------------------------------------
# Peri-stimulus info related
def getspk(fn_mwk, fn_nev=None, override_elecs=None,
        ch_shift=None, ign_unregistered=False,
        override_delay_us=None, verbose=False,
        extinfo=False, exclude_img=None,
        c_success=C_SUCCESS, t_success_lim=T_SUCCESS,
        t_start0=T_START, t_stop0=T_STOP,
        new_thr=None,
        only_new_t=False, reject_sloppy=REJECT_SLOPPY,
        c_stim=C_STIM, c_msg=C_MSG,
        err_utime_msg=ERR_UTIME_MSG, err_utime_type=ERR_UTIME_TYPE):
    """Get all valid spiking info"""

    mf = MWKFile(fn_mwk)
    mf.open()

    if fn_nev is not None:
        br = load_spike_data(fn_nev)
        assert br.open()
    else:
        br = None

    # read TOC info from the "merged" mwk file
    toc = xget_events(mf, codes=[Merge.C_MAGIC])[0].value
    c_spikes = toc[Merge.K_SPIKE]   # get the code name from the toc

    # when the visual stimuli presented is valid?
    t_success = [ev.time for ev in xget_events(mf, codes=[c_success])]
    t_success = np.array(t_success)

    # get the active electrodes
    if override_elecs is None:
        actvelecs = toc[Merge.K_SPIKEWAV].keys()
    else:
        actvelecs = override_elecs    # e.g, range(1, 97)

    # -- preps
    img_onset, img_id = get_stim_info(mf, extinfo=extinfo,
            exclude_img=exclude_img)

    # if requested, remove all sloppy
    # (time spent during update main window > 2 frames)
    if reject_sloppy:
        # since get_stim_info ignores fixation point,
        # all stimuli info must be retrived.
        all_stims = xget_events(mf, codes=[c_stim])
        all_times = np.array([s.time for s in all_stims])
        msgs = xget_events(mf, codes=[c_msg])
        errs = [m for m in msgs if
                m.value['type'] == err_utime_type and
                err_utime_msg in m.value['message']]

        for e in errs:
            t0 = e.time
            rel_t = all_times - t0
            # index to the closest prior stimulus
            ci = int(np.argsort(rel_t[rel_t < 0])[-1])
            # ...and its presented MWK time
            tc = all_stims[ci].time
            # get all affected sloppy stimuli
            ss = list(np.nonzero(np.array(img_onset) == tc)[0])

            new_img_onset = []
            new_img_id = []

            # I know this is kinda O(n^2), but since ss is short,
            # it's essentially O(n)
            for i, (io, ii) in enumerate(zip(img_onset, img_id)):
                if i in ss:
                    if verbose > 1:
                        print '** Removing sloppy:', img_id[i]
                    continue  # if i is sloppy stimuli, remove it.
                new_img_onset.append(io)
                new_img_id.append(ii)

            # trimmed the bad guys..
            img_onset = new_img_onset
            img_id = new_img_id
        assert len(img_onset) == len(img_id)

    n_stim = len(img_onset)

    # MAC-NSP time translation
    if override_delay_us is not None:
        t_delay = toc['align_info']['delay']
        t_adjust = int(np.round(override_delay_us - t_delay))
    else:
        t_adjust = 0
    t_start = t_start0 - t_adjust
    t_stop = t_stop0 - t_adjust

    # yield metadata
    infometa = {'type': 'preamble', 'actvelecs': actvelecs,
            't_adjust': t_adjust}
    if fn_nev is not None:
        infometa['n_packets'] = br._n_packets
        infometa['chn_info'] = br.chn_info
    yield infometa

    # actual calculation -------------------------------
    t0_valid = []
    iid_valid = []
    for i in xrange(n_stim):
        t0 = img_onset[i]
        iid = img_id[i]
        # -- check if this presentation is successful. if it's not ignore this.
        if np.sum((t_success > t0) &
                (t_success < (t0 + t_success_lim))) < 1:
            continue
        t0_valid.append(t0)
        iid_valid.append(iid)
    n_stim_valid = len(t0_valid)

    # -- deal with readaheads
    t_diff = t_stop - t_start + 1  # 1 should be there as t_stop is inclusive.
    readaheads = np.zeros(n_stim_valid, 'int')
    i_cnkbegin = 0            # beginning of the chunk
    for i in xrange(1, n_stim_valid):
        t0 = t0_valid[i]
        t0p = t0_valid[i - 1]
        t0b = t0_valid[i_cnkbegin]

        if (t0 - t0p > DIFF_CUTOFF) or (t0 - t0b > MAX_READAHEAD):
            readaheads[i_cnkbegin:i] = t0p - t0b + t_diff
            i_cnkbegin = i
            continue
    readaheads[i_cnkbegin:] = t0 - t0b + t_diff

    # -- iter over all valid stims
    te_prev = -1
    for i in xrange(n_stim_valid):
        iid = iid_valid[i]
        readahead = int(readaheads[i])

        t0 = t0_valid[i]
        tb = t0 + t_start
        te = t0 + t_stop    # this is inclusive!!
        if only_new_t and tb <= te_prev:
            tb = te_prev + 1
        te_prev = te

        if verbose > 0:
            print 'At', (i + 1), 'out of', n_stim_valid, '         \r',
            sys.stdout.flush()

        # -- yield new image onset info
        infoimg = {'t_imgonset': t0, 'imgid': iid, 'i_img': i, 'type': 'begin'}
        yield infoimg

        spikes, spk_last = xget_events_readahead(mf, c_spikes,
                (tb, te), readahead=readahead, peeklast=True)
        try:
            if br is not None:
                readahead_br = (spk_last.value['foffset'] -
                        spikes[0].value['foffset']) / br._l_packet + 10
                readahead_br = int(readahead_br)
        except:
            # sometimes "spk_last" becomes None
            readahead_br = 1024  # some default value

        # -- yield actual spike info
        for i_spk, spk in enumerate(spikes):
            infospk = infoimg.copy()
            ch = ch0 = spk.value['id']
            pos = spk.value['foffset']
            t_abs = spk.time

            if ch_shift is not None:   # if mapping is requested
                if ch0 not in ch_shift:
                    continue
                ch = ch_shift[ch0]

            if ign_unregistered and ch not in actvelecs:
                continue

            # prepare yield fields
            t_rel = int(t_abs + t_adjust - t0)
            infospk['t_abs'] = t_abs
            infospk['t_rel'] = t_rel
            infospk['ch'] = ch
            infospk['ch_orig'] = ch0
            infospk['i_spk_per_img'] = i_spk
            infospk['pos'] = pos
            infospk['type'] = 'spike'
            # cluster info support for the old merged-mwk format
            infospk['cluster_id'] = None
            infospk['cluster_value'] = None
            if 'cluster_id' in spk.value:
                infospk['cluster_id'] = spk.value['cluster_id']
                infospk['cluster_value'] = spk.value

            # -- if no BRReader is specified, stop here
            if br is None:
                yield infospk
                continue

            # -- or... get the waveform
            try:
                wavinfo = br.read_once(pos=pos, proc_wav=True,
                        readahead=readahead_br)
            except Exception, e:
                print '*** Exception:', e
                continue

            # apply new threshold if requested
            if new_thr is not None:
                if 'mult' in new_thr:
                    lthr = br.chn_info[ch0]['low_thr']
                    hthr = br.chn_info[ch0]['high_thr']
                    if lthr == 0:
                        thr0 = hthr
                    else:
                        thr0 = lthr
                    thr = thr0 * new_thr['mult']
                elif 'abs' in new_thr:
                    thr = new_thr['abs']

                wf0 = wavinfo['waveform']
                wf = set_new_threshold(wf0, thr)

                if wf is None:
                    continue   # `wf0` is smaller than `thr`
                wavinfo['waveform'] = wf

            # done.
            infospk['wavinfo'] = wavinfo
            yield infospk

        # -- finished sweeping all spikes. yield end of image info
        infoimg['type'] = 'end'
        yield infoimg
