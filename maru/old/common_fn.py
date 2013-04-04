#!/usr/bin/env python

import numpy as np
import sys
import os
sys.path.append('lib')
from multiprocessing import Process, Queue
from collections import namedtuple
from mworks.data import MWKFile
from mergeutil import Merge, BRReader, PLXReader

C_STIM = '#announceStimulus'
C_MSG = '#announceMessage'

# by default, do not reject sloppy (time to present > 2 frames) stimuli
REJECT_SLOPPY = False
ERR_UTIME_MSG = 'updating main window display is taking longer than two frames'
# Errors are type 2 in #aanounceStimulus
ERR_UTIME_TYPE = 2

I_STIM_ID = 2
T_START = -100000
T_STOP = 250000
# one visual stimulus should have one "success" event within
C_SUCCESS = 'number_of_stm_shown'
# 250ms time window in order to be considered as valid one.
T_SUCCESS = 250000

OVERRIDE_DELAY_US = 300
T_REJECT = 10
N_REJECT = 50
DEFAULT_N_PCA = 3
N_PRE_PT = 11
SEARCH_RNG = [6, 16]

# if the animal is not working longer than 3s, then don't do readahead
DIFF_CUTOFF = 3000000
# maximum readahead: 6s
MAX_READAHEAD = 6000000


# ----------------------------------------------------------------------------
# Common functions
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


def xget_events_readahead(mf, code, time_range, readahead=15000000, \
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


# ----------------------------------------------------------------------------
def get_stim_info(mf, c_stim=C_STIM, extinfo=False, \
        dynstim='ds_9999', rotrmn=180, blank='zb_99999',\
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
                iid = (x.value['name'], \
                    x.value['pos_x'], x.value['pos_y'], \
                    x.value['rotation'], \
                    x.value['size_x'], x.value['size_y'])
        elif x.value['type'] == 'dynamic_stimulus' or \
                x.value['type'] == 'drifting_grating':
            if not extinfo:
                iid = dynstim
            else:
                iid = (dynstim, \
                    x.value['xoffset'], x.value['yoffset'], \
                    x.value['rotation'] % rotrmn, \
                    x.value['width'], x.value['height'])
        elif x.value['type'] == 'image_directory_movie':
            if not extinfo:
                iid = x.value['current_stimulus']['name']
            else:
                iid = (x.value['current_stimulus']['name'], \
                    x.value['current_stimulus']['pos_x'], \
                    x.value['current_stimulus']['pos_y'], \
                    x.value['current_stimulus']['rotation'], \
                    x.value['current_stimulus']['size_x'], \
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
def load_spike_data(neu_filename):
    ext = os.path.splitext(neu_filename)[1]
    if ext.lower() == '.nev':
        nf = BRReader(neu_filename)
    else:
        nf = PLXReader(neu_filename)
    return nf


# ----------------------------------------------------------------------------
def seq_search(iterable, target):
    """do sequential search"""
    for i, e in enumerate(iterable):
        if e != target:
            continue
        return i
    return None


# ----------------------------------------------------------------------------
def sort_uniq(base, *args):
    """sort and remove duplicates based on `base` and apply on to `args`"""
    if len(args) == 0:
        return None
    res = []
    # sort
    si = np.argsort(base)
    base = np.array(base[si])
    for arg in args:
        res.append(np.array(arg[si]))
    # remove duplicates
    di = np.nonzero(np.diff(base) == 0)[0]
    si = list(set(range(len(base))) - set(list(di)))
    for i in xrange(len(res)):
        res[i] = np.array(res[i][si])
    return res


# ----------------------------------------------------------------------------
def getspk(fn_mwk, fn_nev=None, override_elecs=None, \
        ch_shift=None, ign_unregistered=False, \
        override_delay_us=None, verbose=False, \
        extinfo=False, exclude_img=None, \
        c_success=C_SUCCESS, t_success_lim=T_SUCCESS, \
        t_start0=T_START, t_stop0=T_STOP, \
        new_thr=None, \
        only_new_t=False, reject_sloppy=REJECT_SLOPPY, \
        c_stim=C_STIM, c_msg=C_MSG, \
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
    img_onset, img_id = get_stim_info(mf, extinfo=extinfo, \
            exclude_img=exclude_img)

    # if requested, remove all sloppy
    # (time spent during update main window > 2 frames)
    if reject_sloppy:
        # since get_stim_info ignores fixation point,
        # all stimuli info must be retrived.
        all_stims = xget_events(mf, codes=[c_stim])
        all_times = np.array([s.time for s in all_stims])
        msgs = xget_events(mf, codes=[c_msg])
        errs = [m for m in msgs if \
                m.value['type'] == err_utime_type and \
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
        if np.sum((t_success > t0) & \
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

        spikes, spk_last = xget_events_readahead(mf, c_spikes, \
                (tb, te), readahead=readahead, peeklast=True)
        try:
            if br is not None:
                readahead_br = (spk_last.value['foffset'] - \
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
                wavinfo = br.read_once(pos=pos, proc_wav=True, \
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

                if wf == None:
                    continue   # `wf0` is smaller than `thr`
                wavinfo['waveform'] = wf

            # done.
            infospk['wavinfo'] = wavinfo
            yield infospk

        # -- finished sweeping all spikes. yield end of image info
        infoimg['type'] = 'end'
        yield infoimg


def invalidate_artifacts(buf0, t_reject=T_REJECT, \
        n_reject=N_REJECT, verbose=True):
    """If there are more than `N_REJET` spikes within `T_REJECT`us window,
    invalidate all of them.
    """
    ti_all = [(b['timestamp'], i) for i, b in enumerate(buf0)]
    ti_all = sorted(ti_all)
    t_all = np.array([t[0] for t in ti_all])
    i_all = [t[1] for t in ti_all]

    nb = len(buf0)
    ri = range(nb)
    i = 0
    while i < nb - 1:
        ii = []
        t0 = t_all[i]
        for j in xrange(i + 1, nb):
            if t_all[j] < t0 + t_reject:
                ii.append(j)
            else:
                break
        i = j

        if len(ii) < n_reject:
            continue
        for ix in ii:
            try:
                ri.remove(i_all[ix])
            except ValueError:
                pass

    buf = [buf0[i] for i in ri]
    if verbose and len(buf) != nb:
        print '* Rejecting', nb - len(buf), 'spikes.'
    return buf


def set_new_threshold(wavform, thr, n_pre=N_PRE_PT, rng=SEARCH_RNG, i_chg=20):
    """Set new threshold `thr`.
    If the `waveform` cannot pass `thr` returns None.
    The new waveform is re-aligned based on the steepest point.
    The returned new waveform has `n_pre` points before the alignment point.
    """
    wav = np.array(wavform)
    sgn = np.sign(thr)
    if np.max(wav[rng[0]:rng[1]] * sgn) < np.abs(thr): return None   # reject

    """ NOT USED -- GIVES IMPRECISE RESULT
    # -- align: find the steepest point having the same sign as `sgn`
    df = np.diff(wav)
    si = np.argsort(-sgn * df)   # reverse sorted
    for i in si:
        if np.sign(wav[i]) == sgn: break
    """
    # -- align: find the point where waveform crosses `thr`
    n = len(wav)
    for i in range(n - 1):
        if sgn * wav[i] <= sgn * thr and sgn * thr <= sgn * wav[i + 1]:
            break
    if i == n - 2:
        # although i could be n - 2, it's highly likely an artifact
        return None
    n_shift = n_pre - i - 1    # > 0: right shift, < 0: left shift
    if n_shift == 0:
        return wav

    wavnew = np.empty(wav.shape)
    wavnew[n_shift:] = wav[:-n_shift]   # PBC shifting
    wavnew[:n_shift] = wav[-n_shift:]

    # -- done: but if the spike doesn't change its sign
    #    within `i_chg`, reject.
    if np.max(-sgn * wavnew[n_pre:i_chg]) < 0:
        return None

    """ DEBUG
    if np.abs(n_shift) > 3:
        print '!!!', n_shift, '/', i, '/', n
        print '---',  np.max(-sgn * wavnew[n_pre:i_chg])
        print list(wav)
        print list(wavnew)
    """

    return wavnew


# -----------------------------------------------------------------------------
def parse_opts(opts0):
    """Parse the options in the command line.  This somewhat
    archaic function mainly exists for backward-compatability."""
    opts = {}
    # parse the stuff in "opts"
    for opt in opts0:
        parsed = opt.split('=')
        key = parsed[0].strip()
        if len(parsed) > 1:
            # OLD: cmd = parsed[1].strip()
            cmd = '='.join(parsed[1:]).strip()
        else:
            cmd = ''
        opts[key] = cmd

    return opts


def parse_opts2(tokens, optpx='--', argparam=False):
    """A newer option parser. (from perf102)"""
    opts0 = []
    args = []
    n = len(optpx)

    for token in tokens:
        if token[:2] == optpx:
            opts0.append(token[n:])
        else:
            if argparam:
                token = token.split('=')
            args.append(token)

    opts = parse_opts(opts0)

    return args, opts


def parse_opts_adapter(tokens, delim, optpx='--', argparam=False):
    """Adapter to support both old- and new-style options"""
    if any([t.startswith(optpx) for t in tokens]):
        # new style
        args, opts = parse_opts2(tokens, optpx=optpx, argparam=argparam)
    else:
        # old style
        args = tokens[:delim]
        opts = parse_opts(tokens[delim:])
    return args, opts


def makeavail(sth, sth2idx, idx2sth, query=None):
    if sth not in sth2idx:
        if query is not None and not query(sth):
            return
        sth2idx[sth] = len(idx2sth)
        idx2sth.append(sth)


def prep_files(flist, sep=',', extchk=True):
    flist = flist.split(sep)
    if flist[0][0] == '+':
        flist = [f.strip() for f in open(flist[0][1:]).readlines()]
    if extchk:
        assert all([os.path.exists(f) for f in flist])

    return flist


def prepare_save_dir(sav_dir):
    if sav_dir != '' and not os.path.exists(sav_dir):
        try:
            os.makedirs(sav_dir)
        # in massively-parallel env, it is possible that
        # the sav_dir is created after os.path.exists() check.
        # We just ignore if makedirs fails.
        except Exception:
            pass


def detect_cpus():
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if 'SC_NPROCESSORS_ONLN' in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
        if isinstance(ncpus, int) and ncpus > 0:
            return ncpus
    else:   # OSX:
        return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if 'NUMBER_OF_PROCESSORS' in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1


# -----------------------------------------------------------------------------
# fastnorm: from Nicolas' code
def fastnorm(x):
    xv = x.ravel()
    return np.dot(xv, xv) ** 0.5


# fastsvd: from Nicolas' code
def fastsvd(M):
    h, w = M.shape
    # -- thin matrix
    if h >= w:
        # subspace of M'M
        U, S, V = np.linalg.svd(np.dot(M.T, M))
        U = np.dot(M, V.T)
        # normalize
        for i in xrange(w):
            S[i] = fastnorm(U[:, i])
            U[:, i] = U[:, i] / S[i]
    # -- fat matrix
    else:
        # subspace of MM'
        U, S, V = np.linalg.svd(np.dot(M, M.T))
        V = np.dot(U.T, M)
        # normalize
        for i in xrange(h):
            S[i] = fastnorm(V[i])
            V[i, :] = V[i] / S[i]
    return U, S, V


def pca_eigvec(M, pca_threshold=DEFAULT_N_PCA):
    U, S, V = fastsvd(M)
    eigvectors = V.T
    eigvectors = eigvectors[:, :pca_threshold]
    # this gives PCA:
    # M = np.dot(M, eigvectors)
    return eigvectors

# ----------------------------------------------------------------------------
if __name__ == '__main__':
    print 'This script is not supposed to be excuted directly.'
