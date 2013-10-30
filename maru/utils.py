import numpy as np
import os

# ----------------------------------------------------------------------------
# Common variables

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
CH_SHIFT['20110720A'] = {1: 41, 2: 42, 3: 43, 4: 44, 5: 45, 6: 46,
        7: 47, 8: 48, 9: 49, 10: 50, 11: 51, 12: 52, 13: 53, 14: 54,
        15: 55, 16: 56, 17: 57, 18: 58, 19: 59, 20: 60, 21: 61,
        22: 62, 23: 63, 24: 64, 25: 65, 26: 66, 27: 67, 28: 68,
        29: 69, 30: 70, 31: 71, 32: 72, 33: 73, 34: 74, 35: 75,
        44: 1, 45: 2, 46: 3, 47: 4, 48: 5, 49: 6, 50: 7, 51: 8,
        52: 9, 53: 10, 54: 11, 55: 12, 56: 13, 57: 14, 58: 15,
        59: 16, 60: 17, 61: 18, 62: 19, 63: 20, 64: 21, 65: 22,
        66: 23, 67: 24, 68: 25, 69: 26, 70: 27, 71: 28, 72: 29,
        73: 30, 74: 31, 75: 32, 76: 33, 77: 34, 78: 35, 79: 36,
        80: 37, 81: 38, 82: 39, 83: 40, 94: 76, 95: 77, 96: 78,
        97: 79, 98: 80, 99: 81, 100: 82, 101: 83, 102: 84, 103: 85,
        104: 86, 105: 87, 106: 88, 107: 89, 108: 90, 109: 91,
        110: 92, 111: 93, 112: 94, 113: 95, 114: 96, 115: 97,
        116: 98, 117: 99, 118: 100, 119: 101, 120: 102, 121: 103,
        122: 104, 123: 105, 124: 106, 125: 107, 126: 108,
        127: 109, 128: 110}


# ----------------------------------------------------------------------------
# Common functions
def seq_search(iterable, target):
    """do sequential search"""
    for i, e in enumerate(iterable):
        if e != target:
            continue
        return i
    return None


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
# Peri-stimulus data extraction related
N_PRE_PT = 11
SEARCH_RNG = [6, 16]
T_REJECT = 10
N_REJECT = 50


def invalidate_artifacts(buf0, t_reject=T_REJECT,
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


def set_new_threshold_rng(wav, thr, rng=(11, 13), i_chg=32):
    return set_new_threshold(wav, thr, rng=rng, i_chg=i_chg)
    # return set_new_threshold(wav, thr)


# -----------------------------------------------------------------------------
# Math codes
DEFAULT_N_PCA = 3


def fastnorm(x):
    # fastnorm: from Nicolas' code
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
