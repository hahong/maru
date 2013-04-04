#!/usr/bin/env python

import numpy as np
import cPickle as pk
import tables as tbl
import pywt as wt
import sys
from scipy import interpolate as ipl
from scipy import stats as st
from joblib import Parallel, delayed
from sklearn.cluster import AffinityPropagation
from sklearn.neighbors import NearestNeighbors
from common_fn import parse_opts2, detect_cpus

# -- defaults for feature computation
RETHRESHOLD_MULT = -2.5
ALIGN_SUBSMP = 256
ALIGN_MAXDT = 2.5
ALIGN_PEAKLOC = 12
ALIGN_FINDBWD = 3
ALIGN_FINDFWD = 6
ALIGN_CUTAT = 3
ALIGN_OUTDIM = 32
ALIGN_PEAKFUNC = 'argmin'

FEAT_METHOD = 'wavelet'
FEAT_WAVL_LEV = 4

# -- defaults for clustering
CLUSTERING_ALG = 'affinity_prop'
AFFINITYPRP_COMMONP = 'min'

SKIMSPK_TB = 100000   # beginning relative time for collecting examplar spikes
SKIMSPK_TE = 250000
EXTRACT_NPERIMG = 4
EXTRACT_NMAX = 2200

FEAT_KSSORT = True
FEAT_OUTDIM = 10

QC = False
QC_MINSNR = 3.5       # minimum SNR to be qualified as a cluster
QC_KS_PLEVEL = .01    # a cluster should have "KS-test p" > QC_KS_PLEVEL
QC_MINSIZE = 60

NN_NNEIGH = 5         # the number of neighbors
NN_RADIUS = 1.

# -- other defaults
NCPU = detect_cpus()  # all CPUs
NCPU_LOWLOAD = 4      # for memory intensive tasks
UNSORTED = 0
BADIDX = -1
ATOL = 1e-4

USAGE = \
"""spksort.py: a spike sorting swiss army knife

Feature Computing Mode
======================
Computes feautures of spikes for later clustering after the
re-thresholding (Quiroga et al., 2004) and the spike alignment.

spksort.py feature [options] <input.psf.h5> <output.c1.feat.h5>

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

spksort.py cluster [options] <input.c1.feat.h5> <output.c2.clu.h5>

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
(Optional) Find out common clusters in multiple .c2.clu.h5 files.

spksort.py [options] collate <input1.c2.clu.h5> [input2.c2.clu.h5] ...
"""


# -- House-keeping stuffs -----------------------------------------------------
def par_comp(func, spks, n_jobs=NCPU, **kwargs):
    n = spks.shape[0]
    n0 = max(n / n_jobs, 1)
    ibs = range(0, n, n0)
    ies = range(n0, n + n0, n0)
    r = Parallel(n_jobs=n_jobs, verbose=0)(delayed(func)(spks[ib: ie],
        **kwargs) for ib, ie in zip(ibs, ies))
    return np.concatenate(r)


def KS_all(sig, full=False):
    if sig.shape[0] <= 1:
        if full:
            return np.ones(sig.shape[1]) / ATOL, np.zeros(sig.shape[1]), sig
        return np.ones(sig.shape[1]) / ATOL, sig

    dev = []
    ps = []
    sig = sig - sig.mean(axis=0)
    s = sig.std(ddof=1, axis=0)
    s[np.abs(s) < ATOL] = 1. / ATOL
    sig /= s

    for i in xrange(sig.shape[1]):
        w = sig[:, i]
        #w = (w - w.mean()) / w.std(ddof=1)
        D, p = st.kstest(w, 'norm')
        dev.append(D)
        ps.append(p)

    if full:
        return np.array(dev), np.array(ps), sig
    return np.array(dev), sig


# -- feature extraction related -----------------------------------------------
def rethreshold_by_multiplier_core(Msnp, Msnp_ch, ch,
        mult=RETHRESHOLD_MULT, selected=None):

    return_full = False
    if selected is None:
        selected = np.zeros(Msnp_ch.shape, dtype='bool')
        return_full = True

    i_ch = np.nonzero(Msnp_ch == ch)[0]
    if len(i_ch) == 0:
        thr = 0
    else:
        M = Msnp[i_ch]
        thr = mult * np.median(np.abs(M) / 0.6745)

        if thr < 0:
            i_pass = np.min(M, axis=1) < thr
        else:
            i_pass = np.max(M, axis=1) > thr

        selected[i_ch[i_pass]] = True

    if return_full:
        return selected, thr
    return thr


def rethreshold_by_multiplier_par(Msnp, Msnp_ch, target_chs,
        mult=RETHRESHOLD_MULT, n_jobs=NCPU_LOWLOAD):
    """Experimental parralelized version of rethreshold_by_multiplier()
    TODO: The performance is terrible.  Needs optimization."""
    core = rethreshold_by_multiplier_core
    r = Parallel(n_jobs=n_jobs, verbose=0)(delayed(core)(Msnp,
            Msnp_ch, ch, mult=mult) for ch in target_chs)
    thrs = [e[1] for e in r]
    selected = np.array([e[0] for e in r]).any(axis=0)

    return selected, thrs


def rethreshold_by_multiplier(Msnp, Msnp_ch, target_chs,
        mult=RETHRESHOLD_MULT):
    thrs = []
    selected = np.zeros(Msnp_ch.shape, dtype='bool')
    for ch in target_chs:
        thr = rethreshold_by_multiplier_core(Msnp, Msnp_ch,
                ch, mult=mult, selected=selected)
        thrs.append(thr)

    return selected, thrs


def align_core(spks, subsmp=ALIGN_SUBSMP, maxdt=ALIGN_MAXDT,
        peakloc=ALIGN_PEAKLOC, findbwd=ALIGN_FINDBWD,
        findfwd=ALIGN_FINDFWD, cutat=ALIGN_CUTAT, outdim=ALIGN_OUTDIM,
        peakfunc=ALIGN_PEAKFUNC):
    """Alignment algorithm based on (Quiroga et al., 2004)"""

    if peakfunc == 'argmin':
        peakfunc = np.argmin
    else:
        raise ValueError('Not recognized "peakfunc"')

    R = np.empty((spks.shape[0], outdim), dtype='int16')
    n = spks.shape[1]
    x0 = np.arange(n)

    for i_spk, spk in enumerate(spks):
        tck = ipl.splrep(x0, spk, s=0)

        xn = np.arange(peakloc - findbwd, peakloc + findfwd, 1. / subsmp)
        yn = ipl.splev(xn, tck)

        dt = xn[peakfunc(yn)] - peakloc
        if np.abs(dt) > maxdt:
            dt = 0

        x = x0 + dt
        y = ipl.splev(x, tck)

        R[i_spk] = np.round(y).astype('int16')[cutat: cutat + outdim]
        #dts.append(dt)
    return R


def wavelet_core(spks, level=FEAT_WAVL_LEV):
    """Wavelet transform feature extraction"""
    feat = np.array([np.concatenate(wt.wavedec(spks[i], 'haar',
            level=level)) for i in xrange(spks.shape[0])])
    return feat


# -- clustering related ------------------------------------------------------
def skim_imgs(Mimg, Mimg_tabs, Msnp_tabs, t_adjust=0, tb0=SKIMSPK_TB,
        te0=SKIMSPK_TE, n_blk=20000, onlyonce=True):
    if onlyonce:
        idx_eachimg = [np.nonzero(Mimg == i_img)[0][0] for i_img
                in np.unique(Mimg)]
        t_eachimg = Mimg_tabs[idx_eachimg]
        i_eachimg = Mimg[idx_eachimg]
    else:
        t_eachimg = Mimg_tabs
        i_eachimg = Mimg

    ibie = []
    ib = 0
    ie = 0
    for t0 in t_eachimg:
        tb = t0 + tb0 - t_adjust
        te = t0 + te0 - t_adjust

        xb = np.searchsorted(Msnp_tabs[ib: ib + n_blk], tb)
        if xb >= n_blk:
            xb = np.searchsorted(Msnp_tabs[ib:], tb)
        ib += xb

        xe = np.searchsorted(Msnp_tabs[ie: ie + n_blk], te)
        if xe >= n_blk:
            xe = np.searchsorted(Msnp_tabs[ie:], te)
        ie += xe
        ibie.append((ib, ie))
    return ibie, i_eachimg


def get_example_spikes(Msnp, Msnp_ch, ibie, target_chs,
        nperimg=EXTRACT_NPERIMG, nmax=EXTRACT_NMAX):
    """Extract all spikes specified in ibie"""

    # not using defaultdict to save space
    res = []
    for _ in target_chs:
        res.append([])

    # sweep over each image
    for ib, ie in ibie:
        idx = range(ib, ie)
        Msnp_img = Msnp[idx]
        Msnp_ch_img = Msnp_ch[idx]

        for i_ch, ch in enumerate(target_chs):
            if np.sum([e.shape[0] for e in res[i_ch]]) >= nmax:
                continue
            res[i_ch].append(Msnp_img[Msnp_ch_img == ch][:nperimg])

    for i_ch in xrange(len(target_chs)):
        res[i_ch] = np.concatenate(res[i_ch])[:nmax]

    return res


def cluster_affinity_prop_core(feat, commonp=AFFINITYPRP_COMMONP):
    """Copied from the sklearn website"""
    X = np.array(feat)
    if X.shape[0] == 0:
        return np.zeros((0)), 0, np.zeros((0))

    # -- Compute similarities
    X_norms = np.sum(X ** 2, axis=1)
    S = - X_norms[:, np.newaxis] - X_norms[np.newaxis, :] + 2 * np.dot(X, X.T)

    if commonp == '10med':
        p = 10 * np.median(S)
    elif commonp == 'min':
        p = np.min(S)
    else:
        raise ValueError('Not recognized commonp')

    # -- Compute Affinity Propagation
    af = AffinityPropagation().fit(S, p)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    return labels, n_clusters_, cluster_centers_indices


def find_clusters_par(Msnp_feat_train, feat_use, n_jobs=NCPU,
        metd=CLUSTERING_ALG, **clu_cfg):
    assert len(Msnp_feat_train) == feat_use.shape[0]

    if metd == 'affinity_prop':
        func = cluster_affinity_prop_core
    else:
        raise ValueError('Not recognized clustering metd')

    r = Parallel(n_jobs=n_jobs, verbose=0)(delayed(func)(
            Msnp_feat_train[i][:, feat_use[i]], **clu_cfg)
            for i in xrange(feat_use.shape[0]))

    labels_all = []
    ulabels_all = []
    nclu_all = []
    cluctr_all = []

    for r0 in r:
        labels, nclu, cluctr = r0

        labels_all.append(labels)
        ulabels_all.append(np.unique(labels))
        nclu_all.append(nclu)
        cluctr_all.append(cluctr)

    return labels_all, ulabels_all, nclu_all, cluctr_all


def quality_meas_core(X0, labels):
    sig_quality = {}
    if X0.shape[0] == 0:
        return sig_quality

    for cl in np.unique(labels):
        inds = labels == cl
        N = inds.sum()

        m = X0[inds].mean(0)
        if N <= 1:
            v = np.zeros(X0.shape[1])
        else:
            v = X0[inds].var(0, ddof=1)
        v[np.abs(v) < ATOL] = 1. / ATOL / ATOL   # mute bad signals
        s = np.sqrt(v)
        devs, ps, _ = KS_all(X0[inds], full=True)

        SNRpt = np.mean(np.abs(m) / s)
        SNR = (np.max(m) - np.min(m)) / np.sqrt(np.mean(v))  # Nuo's def.
        SNRm = (np.max(m) - np.min(m)) / np.mean(s)
        KSp = ps
        KSd = devs

        sig_quality[cl] = {'N': N, 'KSp': KSp, 'KSd': KSd,
                'SNR': SNR, 'SNRm': SNRm, 'SNRpt': SNRpt,
                'mean': m, 'var': v}
    return sig_quality


def quality_meas_par(Msnp_train, labels_all,
        n_jobs=NCPU, **kwargs):
    func = quality_meas_core
    r = Parallel(n_jobs=n_jobs, verbose=0)(delayed(func)(
            X, lbl, **kwargs)
            for X, lbl in zip(Msnp_train, labels_all))
    return r


def quality_meas_par2(Msnp, Msnp_ch, Msnp_cid, target_chs,
        n_jobs=NCPU, **kwargs):
    Msnp_all = []
    labels_all = []

    for ch in target_chs:
        i_ch = np.nonzero(Msnp_ch == ch)[0]

        Msnp_all.append(Msnp[i_ch])
        labels_all.append(Msnp_cid[i_ch])

    return quality_meas_par(Msnp_all, labels_all, n_jobs=n_jobs,
            **kwargs)


def quality_ctrl_core(sig_quality, labels, ks_plevel=QC_KS_PLEVEL,
        min_snr=QC_MINSNR, min_size=QC_MINSIZE):
    lbl_conv = {}
    some_unsorted = False
    new_cids = set([UNSORTED])

    for cl in np.unique(labels):
        N = sig_quality[cl]['N']
        SNR = sig_quality[cl]['SNR']
        KSp = sig_quality[cl]['KSp']
        KS_passed = np.all(KSp > ks_plevel)

        if (N > min_size) and (SNR > min_snr) and KS_passed:
            # if passed the quality criteria...
            new_cid = len(new_cids)
            lbl_conv[cl] = new_cid
            new_cids.add(new_cid)
        else:
            some_unsorted = True
            lbl_conv[cl] = UNSORTED
    return lbl_conv, some_unsorted


def quality_ctrl_par(sig_quality_all, labels_all, ulabels_all,
        nclu_all, cluctr_all, n_jobs=NCPU, **kwargs):

    func = quality_ctrl_core
    r = Parallel(n_jobs=n_jobs, verbose=0)(delayed(func)(
            S, lbl, **kwargs)
            for S, lbl in zip(sig_quality_all, labels_all))

    for i_ch, (lbl_conv, some_unsorted) in enumerate(r):
        if len(lbl_conv) == 0:
            continue

        labels_all[i_ch] = np.array([lbl_conv[e] for e in labels_all[i_ch]])
        ulabels_all[i_ch] = np.unique(labels_all[i_ch])
        nclu_all[i_ch] = len(ulabels_all[i_ch])

        cluctr = cluctr_all[i_ch]
        old_cids = range(len(cluctr))
        new_cids = [lbl_conv[old_cid] for old_cid in old_cids]
        s = sorted([(new_cid, cluctr[old_cid]) for new_cid, old_cid in
                zip(new_cids, old_cids) if new_cid != UNSORTED])
        new_cluctr = [BADIDX] + [e[1] for e in s]

        base = 1
        if some_unsorted:
            base = 0
        assert len(new_cluctr) == nclu_all[i_ch] + base
        cluctr_all[i_ch] = new_cluctr


def nearest_neighbor_core(samples, feat, labels,
        nneigh=NN_NNEIGH, radius=NN_RADIUS):
    # samples = np.array(samples)
    assert samples.shape[0] == len(labels)
    assert samples.shape[1] == feat.shape[1]

    neigh = NearestNeighbors(n_neighbors=nneigh, radius=radius,
            warn_on_equidistant=False)
    neigh.fit(samples)
    Y = neigh.kneighbors(feat, return_distance=False)
    L = labels[Y]
    lbl, cnt = st.mode(L, axis=1)
    lbl = lbl.astype('int')
    cnt = cnt.astype('int')
    lbl[cnt == 1] = UNSORTED   # UNSORTED if no conclusive answers are given
    return lbl


def nearest_neighbor_par(Msnp_feat_train, Msnp_feat, Msnp_feat_use, Msnp_ch,
        labels_all, ulabels_all, target_chs, n_jobs=NCPU, **kwargs):

    idx_all = []
    feat_all = []
    fillwith_all = []
    samples_all = []
    Msnp_cid = np.zeros(Msnp_ch.shape, dtype='int')

    for ch in target_chs:
        i_ch = np.nonzero(Msnp_ch == ch)[0]

        idx_all.append(i_ch)
        nlbl = len(ulabels_all[ch])
        nich = len(i_ch)
        if nlbl <= 1 or nich == 0:
            # if there's only one or zero label, no need to do NN search
            samples_all.append(None)
            feat_all.append(None)
            fillwith_all.append(ulabels_all[ch][0] if nlbl > 0 and nich > 0
                    else UNSORTED)
        else:
            use = Msnp_feat_use[ch]
            samples_all.append(np.array(Msnp_feat_train[ch])[:, use])
            feat_all.append(Msnp_feat[i_ch][:, use])
            fillwith_all.append(None)

    assert len(target_chs) == len(samples_all) == len(idx_all) \
            == len(feat_all) == len(fillwith_all) == len(labels_all)

    func = nearest_neighbor_core
    r = Parallel(n_jobs=n_jobs, verbose=0)(delayed(func)(
            samples, feat, labels, **kwargs)
            for samples, feat, labels in
            zip(samples_all, feat_all, labels_all)
            if feat is not None)

    i_r = 0
    for idx, fillwith in zip(idx_all, fillwith_all):
        if len(idx) == 0:
            continue
        if fillwith is None:
            fillwith = r[i_r]
            i_r += 1
        Msnp_cid[idx] = fillwith
    assert i_r == len(r)

    Msnp_cid = Msnp_cid.astype('int16')
    return Msnp_cid


# ----------------------------------------------------------------------------
def get_features(fn_inp, fn_out, opts):
    config = {}
    config['rethreshold_mult'] = RETHRESHOLD_MULT
    config['align'] = {}
    config['align']['subsmp'] = ALIGN_SUBSMP
    config['align']['maxdt'] = ALIGN_MAXDT
    config['align']['peakloc'] = ALIGN_PEAKLOC
    config['align']['findbwd'] = ALIGN_FINDBWD
    config['align']['findfwd'] = ALIGN_FINDFWD
    config['align']['cutat'] = ALIGN_CUTAT
    config['align']['outdim'] = ALIGN_OUTDIM
    config['align']['peakfunc'] = ALIGN_PEAKFUNC

    config['feat'] = {}
    config['feat']['metd'] = FEAT_METHOD
    config['feat']['kwargs'] = {'level': FEAT_WAVL_LEV}

    n_jobs = NCPU

    # -- process opts
    if 'njobs' in opts:
        n_jobs = int(opts['njobs'])
        print '* n_jobs =', n_jobs
    # TODO: implement!!!

    # -- preps
    print '-> Initializing...'
    h5 = tbl.openFile(fn_inp)
    Msnp = h5.root.Msnp.read()
    Msnp_ch = h5.root.Msnp_ch.read()
    Msnp_pos = h5.root.Msnp_pos.read()
    Msnp_tabs = h5.root.Msnp_tabs.read()
    Mimg = h5.root.Mimg.read()
    Mimg_tabs = h5.root.Mimg_tabs.read()

    t_adjust = h5.root.meta.t_adjust.read()
    t_start0 = h5.root.meta.t_start0.read()
    t_stop0 = h5.root.meta.t_stop0.read()

    idx2iid = h5.root.meta.idx2iid.read()
    iid2idx_pk = h5.root.meta.iid2idx_pk.read()
    idx2ch = h5.root.meta.idx2ch.read()
    ch2idx_pk = h5.root.meta.ch2idx_pk.read()
    all_chs = range(len(idx2ch))

    # -- re-threshold
    print '-> Re-thresholding...'
    if type(config['rethreshold_mult']) is float or int:
        thr_sel, thrs = rethreshold_by_multiplier(Msnp, Msnp_ch,
                all_chs, config['rethreshold_mult'])

        Msnp = Msnp[thr_sel]
        Msnp_ch = Msnp_ch[thr_sel]
        Msnp_pos = Msnp_pos[thr_sel]
        Msnp_tabs = Msnp_tabs[thr_sel]
        Msnp_selected = np.nonzero(thr_sel)[0]

    else:
        thr_sel = None
        thrs = None
        Msnp_selected = None

    # -- align
    print '-> Aligning...'
    Msnp = par_comp(align_core, Msnp, n_jobs=n_jobs, **config['align'])

    # -- feature extraction
    print '-> Extracting features...'
    if config['feat']['metd'] == 'wavelet':
        Msnp_feat = par_comp(wavelet_core, Msnp, n_jobs=n_jobs,
                **config['feat']['kwargs'])

    elif config['feat']['metd'] != 'pca':
        config['feat']['kwargs'].pop('level')
        raise NotImplementedError('PCA not implemented yet')

    else:
        raise ValueError('Not recognized "feat_metd"')

    # -- done! write everything...
    print '-> Writing results...'
    filters = tbl.Filters(complevel=4, complib='blosc')
    t_int16 = tbl.Int16Atom()
    t_uint32 = tbl.UInt32Atom()
    t_uint64 = tbl.UInt64Atom()
    t_float32 = tbl.Float32Atom()

    h5o = tbl.openFile(fn_out, 'w')
    CMsnp = h5o.createCArray(h5o.root, 'Msnp', t_int16,
            Msnp.shape, filters=filters)
    CMsnp_tabs = h5o.createCArray(h5o.root, 'Msnp_tabs', t_uint64,
            Msnp_tabs.shape, filters=filters)
    CMsnp_ch = h5o.createCArray(h5o.root, 'Msnp_ch', t_uint32,
            Msnp_ch.shape, filters=filters)
    CMsnp_pos = h5o.createCArray(h5o.root, 'Msnp_pos', t_uint64,
            Msnp_pos.shape, filters=filters)
    CMsnp_feat = h5o.createCArray(h5o.root, 'Msnp_feat', t_float32,
            Msnp_feat.shape, filters=filters)
    # TODO: support when thr_sel is None
    CMsnp_selected = h5o.createCArray(h5o.root, 'Msnp_selected',
            t_uint64, Msnp_selected.shape, filters=filters)

    CMsnp[...] = Msnp
    CMsnp_tabs[...] = Msnp_tabs
    CMsnp_ch[...] = Msnp_ch
    CMsnp_pos[...] = Msnp_pos
    CMsnp_feat[...] = Msnp_feat
    CMsnp_selected[...] = Msnp_selected

    h5o.createArray(h5o.root, 'Mimg', Mimg)
    h5o.createArray(h5o.root, 'Mimg_tabs', Mimg_tabs)

    meta = h5o.createGroup('/', 'meta', 'Metadata')
    h5o.createArray(meta, 't_start0', t_start0)
    h5o.createArray(meta, 't_stop0', t_stop0)
    h5o.createArray(meta, 't_adjust', t_adjust)
    h5o.createArray(meta, 'config_feat_pk', pk.dumps(config))

    h5o.createArray(meta, 'idx2iid', idx2iid)
    h5o.createArray(meta, 'iid2idx_pk', iid2idx_pk)
    h5o.createArray(meta, 'idx2ch', idx2ch)
    h5o.createArray(meta, 'ch2idx_pk', ch2idx_pk)

    h5o.createArray(meta, 'thrs', thrs)
    h5o.createArray(meta, 'fn_inp', fn_inp)

    h5o.close()
    h5.close()


# ----------------------------------------------------------------------------
def cluster(fn_inp, fn_out, opts):
    config = {}
    config['skimspk'] = {}
    config['skimspk']['tb0'] = SKIMSPK_TB
    config['skimspk']['te0'] = SKIMSPK_TE
    config['extract'] = {}
    config['extract']['nperimg'] = EXTRACT_NPERIMG
    config['extract']['nmax'] = EXTRACT_NMAX

    config['feat'] = {}
    config['feat']['kssort'] = FEAT_KSSORT
    config['feat']['outdim'] = FEAT_OUTDIM

    config['cluster'] = {}
    config['cluster']['metd'] = CLUSTERING_ALG
    config['cluster']['commonp'] = AFFINITYPRP_COMMONP

    config['qc'] = {}
    config['qc']['qc'] = QC
    config['qc']['kwargs'] = {}
    config['qc']['kwargs']['min_snr'] = QC_MINSNR
    config['qc']['kwargs']['ks_plevel'] = QC_KS_PLEVEL
    config['qc']['kwargs']['min_size'] = QC_MINSIZE

    config['nn'] = {}
    config['nn']['nneigh'] = NN_NNEIGH
    config['nn']['radius'] = NN_RADIUS

    n_jobs = NCPU
    reference = None

    # -- process opts
    if 'njobs' in opts:
        n_jobs = int(opts['njobs'])
        print '* n_jobs =', n_jobs

    if 'ref' in opts:
        reference = opts['ref']
        print '* Using the reference file:', reference
        h5r = tbl.openFile(reference)
        config = pk.loads(h5r.root.meta.config_clu_pk.read())

    # TODO: implement other options!!!

    # -- preps
    print '-> Initializing...'
    h5 = tbl.openFile(fn_inp)
    Msnp = h5.root.Msnp.read()
    Msnp_feat = h5.root.Msnp_feat.read()
    Msnp_ch = h5.root.Msnp_ch.read()
    Msnp_pos = h5.root.Msnp_pos.read()
    Msnp_tabs = h5.root.Msnp_tabs.read()
    Msnp_selected = h5.root.Msnp_selected.read()
    Mimg = h5.root.Mimg.read()
    Mimg_tabs = h5.root.Mimg_tabs.read()

    t_adjust = h5.root.meta.t_adjust.read()
    t_start0 = h5.root.meta.t_start0.read()
    t_stop0 = h5.root.meta.t_stop0.read()

    idx2iid = h5.root.meta.idx2iid.read()
    iid2idx_pk = h5.root.meta.iid2idx_pk.read()
    idx2ch = h5.root.meta.idx2ch.read()
    ch2idx_pk = h5.root.meta.ch2idx_pk.read()
    all_chs = range(len(idx2ch))

    if reference is None:
        # -- get training examples...
        print '-> Collecting snippet examples...'
        ibie, iuimg = skim_imgs(Mimg, Mimg_tabs, Msnp_tabs,
                t_adjust, **config['skimspk'])

        clu_feat_train = get_example_spikes(Msnp_feat, Msnp_ch, ibie, all_chs,
                **config['extract'])
        clu_train = get_example_spikes(Msnp, Msnp_ch, ibie, all_chs,
                **config['extract'])

        # -- get feature indices to use...
        print '-> Finding useful axes...'
        outdim = config['feat']['outdim']
        if config['feat']['kssort']:
            Msnp_feat_use = []

            for i_ch in xrange(len(all_chs)):
                # get deviations from Gaussian
                devs, _ = KS_all(clu_feat_train[i_ch])
                # got top-n deviations
                devs = np.argsort(-devs)[:outdim]
                Msnp_feat_use.append(devs)
        else:
            Msnp_feat_use = [range(outdim)] * len(all_chs)
        Msnp_feat_use = np.array(Msnp_feat_use)

        # -- XXX: DEBUG SUPPORT
        __DBG__ = False
        if __DBG__:
            clu_feat_train = clu_feat_train[:4]
            clu_train = clu_train[:4]
            Msnp_feat_use = Msnp_feat_use[:4]
            all_chs = all_chs[:4]

        # -- get clusters...
        print '-> Clustering...'
        clu_labels, clu_ulabels, clu_nclus, clu_centers = \
                find_clusters_par(clu_feat_train,
                Msnp_feat_use, n_jobs=n_jobs,
                **config['cluster'])

        # -- quality control
        if config['qc']['qc']:
            print '-> Run signal quality-based screening...'
            clu_sig_q = quality_meas_par(clu_train, clu_labels)
            quality_ctrl_par(clu_sig_q, clu_labels, clu_ulabels,
                    clu_nclus, clu_centers, n_jobs=n_jobs,
                    **config['qc']['kwargs'])
    else:
        # -- Bypass all and get the pre-computed clustering template
        print '-> Loading reference data...'
        clu_feat_train = pk.loads(h5r.root.clu_pk.clu_feat_train_pk.read())
        clu_train = pk.loads(h5r.root.clu_pk.clu_train_pk.read())
        clu_labels = pk.loads(h5r.root.clu_pk.clu_labels_pk.read())
        clu_ulabels = pk.loads(h5r.root.clu_pk.clu_ulabels_pk.read())
        clu_nclus = pk.loads(h5r.root.clu_pk.clu_nclus_pk.read())
        clu_centers = pk.loads(h5r.root.clu_pk.clu_centers_pk.read())
        clu_sig_q = pk.loads(h5r.root.clu_pk.clu_sig_q_pk.read())
        Msnp_feat_use = h5r.root.Msnp_feat_use.read()
        h5r.close()

    # -- NN search
    print '-> Template matching...'
    Msnp_cid = nearest_neighbor_par(clu_feat_train, Msnp_feat,
            Msnp_feat_use, Msnp_ch, clu_labels, clu_ulabels,
            all_chs, n_jobs=n_jobs, **config['nn'])

    # -- final quality report...
    print '-> Computing the final signal quality report...'
    clu_sig_q = quality_meas_par2(Msnp, Msnp_ch, Msnp_cid,
            all_chs, n_jobs=n_jobs)
    # ... and update clu_ulabels, clu_nclus
    clu_ulabels = [np.array(clu_sig_q_ch.keys(), dtype='int') for
            clu_sig_q_ch in clu_sig_q]
    clu_nclus = [len(clu_sig_q_ch.keys()) for clu_sig_q_ch in clu_sig_q]

    # -- done! write everything...
    print '-> Writing results...'
    filters = tbl.Filters(complevel=4, complib='blosc')
    t_int16 = tbl.Int16Atom()
    t_uint32 = tbl.UInt32Atom()
    t_uint64 = tbl.UInt64Atom()

    h5o = tbl.openFile(fn_out, 'w')
    CMsnp_tabs = h5o.createCArray(h5o.root, 'Msnp_tabs', t_uint64,
            Msnp_tabs.shape, filters=filters)
    CMsnp_ch = h5o.createCArray(h5o.root, 'Msnp_ch', t_uint32,
            Msnp_ch.shape, filters=filters)
    CMsnp_cid = h5o.createCArray(h5o.root, 'Msnp_cid', t_int16,
            Msnp_cid.shape, filters=filters)
    CMsnp_pos = h5o.createCArray(h5o.root, 'Msnp_pos', t_uint64,
            Msnp_pos.shape, filters=filters)
    CMsnp_feat_use = h5o.createCArray(h5o.root, 'Msnp_feat_use', t_uint64,
            Msnp_feat_use.shape, filters=filters)
    CMsnp_selected = h5o.createCArray(h5o.root, 'Msnp_selected',
            t_uint64, Msnp_selected.shape, filters=filters)

    CMsnp_tabs[...] = Msnp_tabs
    CMsnp_ch[...] = Msnp_ch
    CMsnp_cid[...] = Msnp_cid
    CMsnp_pos[...] = Msnp_pos
    CMsnp_feat_use[...] = Msnp_feat_use
    CMsnp_selected[...] = Msnp_selected

    h5o.createArray(h5o.root, 'Mimg', Mimg)
    h5o.createArray(h5o.root, 'Mimg_tabs', Mimg_tabs)

    meta = h5o.createGroup('/', 'meta', 'Metadata')
    h5o.createArray(meta, 't_start0', t_start0)
    h5o.createArray(meta, 't_stop0', t_stop0)
    h5o.createArray(meta, 't_adjust', t_adjust)
    h5o.createArray(meta, 'config_clu_pk', pk.dumps(config))

    h5o.createArray(meta, 'idx2iid', idx2iid)
    h5o.createArray(meta, 'iid2idx_pk', iid2idx_pk)
    h5o.createArray(meta, 'idx2ch', idx2ch)
    h5o.createArray(meta, 'ch2idx_pk', ch2idx_pk)

    h5o.createArray(meta, 'fn_inp', fn_inp)

    clupk = h5o.createGroup('/', 'clu_pk', 'Pickles')
    h5o.createArray(clupk, 'clu_feat_train_pk', pk.dumps(clu_feat_train))
    h5o.createArray(clupk, 'clu_train_pk', pk.dumps(clu_train))
    h5o.createArray(clupk, 'clu_labels_pk', pk.dumps(clu_labels))
    h5o.createArray(clupk, 'clu_ulabels_pk', pk.dumps(clu_ulabels))
    h5o.createArray(clupk, 'clu_nclus_pk', pk.dumps(clu_nclus))
    h5o.createArray(clupk, 'clu_centers_pk', pk.dumps(clu_centers))
    h5o.createArray(clupk, 'clu_sig_q_pk', pk.dumps(clu_sig_q))

    h5o.close()
    h5.close()


# ----------------------------------------------------------------------------
def _get_good_clu(clu_sig_q):
    """EXPERIMENTAL!!! EXPECT BUGS!!
    from https://mh17:19999/932fdc93-1a2b-4d00-a520-96ebd44ee450
    """
    kind = 'N'
    xN = np.array([clu_sig_q_ch[cl][kind] for clu_sig_q_ch in
        clu_sig_q for cl in clu_sig_q_ch])
    kind = 'SNR'
    xSNR = np.array([clu_sig_q_ch[cl][kind] for clu_sig_q_ch in
        clu_sig_q for cl in clu_sig_q_ch])
    kind = 'mean'
    xmean = np.array([clu_sig_q_ch[cl][kind] for clu_sig_q_ch in
        clu_sig_q for cl in clu_sig_q_ch])
    kind = 'var'
    xvar = np.array([clu_sig_q_ch[cl][kind] for clu_sig_q_ch in
        clu_sig_q for cl in clu_sig_q_ch])
    kind = 'KSp'
    xKSp = np.array([clu_sig_q_ch[cl][kind] for clu_sig_q_ch in
        clu_sig_q for cl in clu_sig_q_ch])
    xch = np.array([i_ch for i_ch, clu_sig_q_ch in
        enumerate(clu_sig_q) for cl in clu_sig_q_ch])
    xcl = np.array([cl for clu_sig_q_ch in clu_sig_q for cl in
        clu_sig_q_ch])

    # count the number of sign changes
    xchg = np.sum(np.abs(np.diff(np.sign(np.diff(xmean, axis=1)),
        axis=1)), axis=1) / 2

    Nmin = np.sum(xN) / len(clu_sig_q) * 0.01
    # first pass
    idxx0 = (xSNR > 5) & (xN > Nmin) & (np.sum(xKSp > 0.01, axis=1)
            >= 16) & (xchg < 8)

    xmax = np.max(xmean[idxx0], axis=1)
    xmin = np.min(xmean[idxx0], axis=1)
    xvarvar = np.var(xvar[idxx0], axis=1)
    npass = len(xmin)
    npass0 = npass - 1

    minlowthr = sorted(xmin)[int(round(npass * 0.05))]
    maxlowthr = sorted(xmax)[int(round(npass * 0.025))]
    maxhighthr = sorted(xmax)[min(int(round(npass * 0.975)), npass0)]
    varvarhighthr = sorted(xvarvar)[min(int(round(npass * 0.95)), npass0)]

    idxx1 = (xmax > maxlowthr) & (xmax < maxhighthr) & (xmin >
            minlowthr) & (xvarvar < varvarhighthr)
    idxx = np.nonzero(idxx0)[0][idxx1]
    chxx = xch[idxx]
    clxx = xcl[idxx]

    res1 = sorted(zip(chxx, clxx))
    assert len(res1) == len(set(res1))  # unique!!

    from collections import defaultdict as dd
    res2 = dd(list)
    #res2 = []
    #for _ in xrange(len(clu_sig_q)):
    #    res2.append([])
    for ch, cid in res1:
        res2[ch].append(cid)

    # XXX: has to be 1-based
    res3 = dict([(e, i + 1) for i, e in enumerate(res1)])

    return res1, res2, res3


def collate(fn_inp, fn_out, opts):
    """EXPERIMENTAL!!!!  EXPECT BUGS!!!"""

    reference = None
    # -- process opts
    if 'ref' in opts:
        reference = opts['ref']
        print '* Using the reference file:', reference
    # TODO: implement others!!

    # -- preps
    print '-> Initializing...'
    h5 = tbl.openFile(fn_inp)

    Msnp_ch = h5.root.Msnp_ch.read()
    Msnp_cid = h5.root.Msnp_cid.read()
    Msnp_tabs = h5.root.Msnp_tabs.read()
    Mimg = h5.root.Mimg.read()
    Mimg_tabs = h5.root.Mimg_tabs.read()

    t_adjust = h5.root.meta.t_adjust.read()
    t_start0 = h5.root.meta.t_start0.read()
    t_stop0 = h5.root.meta.t_stop0.read()
    print '   DBG:', t_adjust, t_start0, t_stop0

    idx2iid = h5.root.meta.idx2iid.read()
    idx2ch = h5.root.meta.idx2ch.read()
    assert idx2ch == range(1, len(idx2ch) + 1), idx2ch

    # -- get good clusters
    print '-> Get useful clusters...'
    if reference is None:
        clu_sig_q = pk.loads(h5.root.clu_pk.clu_sig_q_pk.read())
    else:
        h5r = tbl.openFile(reference)
        clu_sig_q = pk.loads(h5r.root.clu_pk.clu_sig_q_pk.read())
        h5r.close()
    idx2gcid, cid_sel, gcid2idx = _get_good_clu(clu_sig_q)
    ###
    print '   DBG: #passed =', len(idx2gcid)

    # -- sweep images
    print '-> Sweep images...'
    ibie, iuimg = skim_imgs(Mimg, Mimg_tabs, Msnp_tabs,
            t_adjust, tb0=t_start0, te0=t_stop0, onlyonce=False)
    assert len(ibie) == len(iuimg) == len(Mimg_tabs)

    # XXX: has to be 1-based
    all_spike = {}
    actvelecs = range(1, len(idx2gcid) + 1)
    for el in actvelecs:
        all_spike[el] = {}

    for i0 in xrange(len(ibie)):
        ###
        print '   DBG: At: %d/%d         \r' % (i0 + 1, len(ibie)),
        sys.stdout.flush()
        ###
        ib, ie = ibie[i0]
        ii = iuimg[i0]
        iid = idx2iid[ii]
        t0 = Mimg_tabs[i0]

        idx = range(ib, ie)
        Xch = Msnp_ch[idx]
        Xcid = Msnp_cid[idx]
        Xt = Msnp_tabs[idx]

        if iid not in all_spike[actvelecs[0]]:
            for el in actvelecs:
                all_spike[el][iid] = []

        for ch in cid_sel:
            idxch = (Xch == ch)
            for cl in cid_sel[ch]:
                idxcl = idxch & (Xcid == cl)
                t_abs = Xt[idxcl]
                t_rel = (t_abs + t_adjust - t0).astype('int')
                # Not needed: t_rel = list(t_rel)
                el = gcid2idx[(ch, cl)]  # global cid (or channel)
                all_spike[el][iid].append(t_rel)

    # -- done
    h5.close()

    f = open(fn_out, 'w')
    out = {'all_spike': all_spike,
           't_start': t_start0,
           't_stop': t_stop0,
           't_adjust': t_adjust,
           'actvelecs': actvelecs,
           'idx2gcid': idx2gcid,
           'cid_sel': cid_sel,
           'gcid2idx': gcid2idx
           }
    pk.dump(out, f)
    f.close()


# ----------------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print USAGE
        return

    args, opts = parse_opts2(sys.argv[1:])
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


if __name__ == '__main__':
    main()
