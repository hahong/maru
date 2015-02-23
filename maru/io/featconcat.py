#!/usr/bin/env python
import tables
import numpy as np
import cPickle as pk
import os
from joblib import Parallel, delayed

I_TR = 1        # index for trial, in /spk
EPSSTD = 0.001  # allowed minimum stddv in whitening

ELMAP = {}   # Electrode map, ELMAP['rule'][local_elecid] = global_elecid
ELMAP['_A_'] = ELMAP['_A.'] = range(96)
ELMAP['_M_'] = ELMAP['_M.'] = range(96, 96 * 2)
ELMAP['_P_'] = ELMAP['_P.'] = range(96 * 2, 96 * 3)
ELMAP['_S130404_'] = ELMAP['_S130404.'] = [192, 193, 194, 195, 196, 197, 198,
        199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
        213, 214, 215, 217, 218, 219, 220, 222, 223, 224, 225, 226, 227, 228,
        229, 230, 231, 233, 235, 1, 2, 3, 6, 8, 9, 10, 12, 14, 15, 16, 18, 19,
        20, 21, 24, 25, 26, 27, 28, 29, 31, 32, 34, 47, 46, 43, 40, 50, 52, 53,
        54, 56, 57, 59, 60, 61, 62, 63, 65, 85, 91, 93, 94, 95, 239, 237, 236,
        241, 243, 245, 246, 248, 249, 250, 251, 252, 253, 255, 256, 257, 258,
        259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 274,
        275, 276, 277, 278, 280, 281, 282, 283, 284, 285, 286, 287]
ELMAP['_S130404M96'] = ELMAP['_S130404_'][40: 85] + ELMAP['_M_'] + \
        ELMAP['_S130404_'][:40] + ELMAP['_S130404_'][85:]
ELMAP['_S110720_'] = [100, 104, 105, 106, 107, 108, 111, 112, 113, 114, 115,
        116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 132, 133, 134, 135,
        136, 137, 138, 139, 140, 141, 142, 147, 145, 144, 218, 219, 222, 223,
        250, 248, 246, 245, 0, 4, 5, 7, 9, 12, 13, 14, 15, 17, 21, 23, 25, 27,
        29, 31, 47, 53, 52, 49, 48, 54, 55, 56, 57, 58, 59, 62, 63, 75, 78, 81,
        85, 86, 88, 89, 91, 92, 94, 95, 252, 254, 255, 268, 282, 283, 284, 285,
        286, 287, 148, 149, 150, 151, 152, 153, 155, 157, 158, 165, 166, 167,
        168, 169, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
        183, 184, 185, 186, 187, 188, 189, 190, 191]
ELMAP['_S110720P58'] = ELMAP['_S110720_'] + [220, 251, 212, 280, 221, 249, 253,
        281, 237, 215, 264, 239, 206, 216, 209, 278, 262, 207, 276, 266, 210,
        208, 271, 225, 261, 231, 202, 224, 213, 204, 272, 233, 195, 203, 196,
        279, 194, 214, 242, 217]
ELMAP['_S110720A_'] = [0, 4, 5, 7, 9, 12, 13, 14, 15, 17, 21, 23, 25, 27, 29,
        31, 47, 53, 52, 49, 48, 54, 55, 56, 57, 58, 59, 62, 63, 75, 78, 81, 85,
        86, 88, 89, 91, 92, 94, 95, 100, 104, 105, 106, 107, 108, 111, 112,
        113, 114, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 132,
        133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 147, 145, 144, 148,
        149, 150, 151, 152, 153, 155, 157, 158, 165, 166, 167, 168, 169, 171,
        172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
        186, 187, 188, 189, 190, 191]
# OLD: ELMAP['_S110204_'] = range(128)   # Chabo
ELMAP['_S110204_'] = [101, 103, 104, 105, 106, 107, 117, 118, 121, 122, 123,
        192, 193, 194, 199, 201, 202, 204, 205, 206, 207, 208, 209, 210, 211,
        212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 226,
        227, 228, 231, 232, 233, 235, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 39, 40, 41, 43, 47, 46, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 63, 86, 87, 90, 91, 93, 238, 237, 240, 241,
        243, 244, 246, 248, 249, 250, 251, 252, 253, 254, 255, 260, 263, 265,
        266, 268, 269, 272, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283,
        284, 285, 286, 287, 125, 158, 160, 166, 172, 190, 191]
ELMAP['_AMP'] = range(96 * 3)
ELMAP['default'] = ELMAP['_AMP']
ELMAP['pooled_all'] = range(128 + 96 * 3)
ELMAP['pooled_P58'] = range(128) + [e + 128 for e in ELMAP['_S110720P58']]
# must update this one:
ELRULEORDER = ['_A_', '_A.', '_M_', '_M.', '_P_', '_P.',
        '_S130404_', '_S130404.', '_S130404M96',
        '_S110720_', '_S110720P58', '_S110720A_',
        '_AMP', 'default', 'pooled_all', 'pooled_P58', '_S110204_']


# ---------------------------------
# for compatibility issue
def safe_list(iterable):
    if type(iterable) == list:
        return iterable
    l = []
    for k in sorted(iterable.keys()):
        l.append(iterable[k])
    return l


def get_total_elec(rule):
    assert rule in ELMAP, 'Invalid rule!'
    return len(ELMAP[rule])


def get_chconv(inrule0, outrule):
    # returns channel_list_from, channel_list_to

    found = False
    for inrule in ELRULEORDER:
        if inrule in inrule0:
            found = True
            break
    assert found, 'Invalid: %s' % inrule0

    inchs_l2g = ELMAP[inrule]
    outchs_l2g = ELMAP[outrule]

    ch_from = []
    ch_to = []

    for inch_l, inch_g in enumerate(inchs_l2g):
        if inch_g not in outchs_l2g:
            continue

        ch_from.append(inch_l)
        ch_to.append(outchs_l2g.index(inch_g))

    return ch_from, ch_to


##############################################################################
def get_evoked_core_py(spk0, orgfile0, i_bgs, i_srcs, i_chs=None):
    n_lb, _, _, n_ch = spk0.shape
    if i_chs is None:
        i_chs = range(n_ch)

    for i_src in i_srcs:
        # print i_src
        for i_lb in xrange(n_lb):
            for i_ch in i_chs:
                # select all i_src
                iiall_tr, iiall_ii = np.nonzero(orgfile0[i_lb, :, :, i_ch] ==
                        i_src)
                iibgs_tr = []
                iibgs_ii = []
                for xtr, xii in zip(iiall_tr, iiall_ii):
                    if xii not in i_bgs:
                        continue
                    iibgs_tr.append(xtr)
                    iibgs_ii.append(xii)
                if len(iibgs_ii) == 0:
                    continue
                # mean background response
                b = np.mean(spk0[i_lb, iibgs_tr, iibgs_ii, i_ch])
                spk0[i_lb, iiall_tr, iiall_ii, i_ch] -= b

    return spk0


def get_evoked(spk0, orgfile0, i_bgs, i_srcs, i_chs=None):
    n_ch = spk0.shape[-1]
    if i_chs is None:
        i_chs = range(n_ch)
    res = Parallel(n_jobs=-1, verbose=1)(delayed(get_evoked_core_py)(
        spk0[:, :, :, [i_ch]],
        orgfile0[:, :, :, [i_ch]], i_bgs, i_srcs) for i_ch in i_chs)
    for i_ch0, i_ch in enumerate(i_chs):
        spk0[:, :, :, [i_ch]] = res[i_ch0]


# --------
def get_whitened_core_ch_py(spk0, orgfile0, i_srcs, dbg_i_ch=None):
    n_lb, _, _ = spk0.shape

    for i_src in i_srcs:
        # print i_src
        for i_lb in xrange(n_lb):
            spk = spk0[i_lb, :, :]
            orgfile = orgfile0[i_lb, :, :]

            ii = (orgfile == i_src)
            if np.sum(ii) == 0:
                continue

            v = spk[ii]
            m = np.mean(v)
            s = np.std(v - m, ddof=1)

            if s > EPSSTD:
                spk[ii] -= m
                spk[ii] /= s
            elif dbg_i_ch is not None:
                print '    * Low std: i_ch0=%d, i_src=%d, i_lbl=%d; mean=%f' \
                        % (dbg_i_ch, i_src, i_lb, m)
                spk[ii] = 0
    return spk0


def get_whitened(spk0, orgfile0, i_srcs, i_chs=None):
    n_ch = spk0.shape[-1]
    if i_chs is None:
        i_chs = range(n_ch)
    res = Parallel(n_jobs=-1, verbose=1)(delayed(get_whitened_core_ch_py)(
        spk0[:, :, :, i_ch], orgfile0[:, :, :, i_ch], i_srcs, i_ch)
        for i_ch in i_chs)
    for i_ch0, i_ch in enumerate(i_chs):
        spk0[:, :, :, i_ch] = res[i_ch0]


##############################################################################
def concat_core_py(spk, spk0, ntrimg, ntrials0, orgfile, orgfile_t,
        i_lbls, i_lbl0s, i_iids, i_iid0s,
        i_chs, i_ch0s, n_tr):
    n_lbls = len(i_lbls)
    n_iids = len(i_iids)
    n_chs = len(i_chs)

    for xl in xrange(n_lbls):
        i_lbl = i_lbls[xl]
        i_lbl0 = i_lbl0s[xl]
        if i_lbl0 < 0:
            continue

        for xi in xrange(n_iids):
            i_iid = i_iids[xi]
            i_iid0 = i_iid0s[xi]
            if i_iid0 < 0:
                continue

            for xc in xrange(n_chs):
                i_ch = i_chs[xc]
                i_ch0 = i_ch0s[xc]

                # -- actual conversion
                i_trb = ntrimg[i_lbl, i_iid, i_ch]   # begin of trial index
                i_tre = min(i_trb + ntrials0[i_lbl0, i_iid0, i_ch0], n_tr)
                n = i_tre - i_trb

                spk[i_lbl, i_trb:i_tre, i_iid, i_ch] = \
                        spk0[i_lbl0, :n, i_iid0, i_ch0]
                orgfile[i_lbl, i_trb:i_tre, i_iid, i_ch] = \
                        orgfile_t[i_lbl0, :n, i_iid0, i_ch0]
                ntrimg[i_lbl, i_iid, i_ch] += n

try:
    from . import concat_core as cco

    def concat_core(spk, spk0, ntrimg, ntrials0, orgfile, orgfile_t, i_lbls,
            i_lbl0s, i_iids, i_iid0s,
            i_chs, i_ch0s, n_tr):
        assert ntrimg.dtype == np.uint16
        assert ntrials0.dtype == np.uint16
        assert orgfile.dtype == np.int16
        assert orgfile_t.dtype == np.int16
        assert i_lbls.dtype == np.int32
        assert i_lbl0s.dtype == np.int32
        assert i_iids.dtype == np.int32
        assert i_iid0s.dtype == np.int32
        assert i_chs.dtype == np.int32
        assert i_ch0s.dtype == np.int32
        assert type(n_tr) == int

        if spk.dtype == np.uint16 and spk0.dtype == np.uint16:
            cco.concat_core_pyx_u16u16(spk, spk0, ntrimg, ntrials0,
                    orgfile, orgfile_t, i_lbls, i_lbl0s,
                    i_iids, i_iid0s, i_chs, i_ch0s, n_tr)
        elif spk.dtype == np.float32 and spk0.dtype == np.float32:
            cco.concat_core_pyx_f32f32(spk, spk0, ntrimg, ntrials0,
                    orgfile, orgfile_t, i_lbls, i_lbl0s,
                    i_iids, i_iid0s, i_chs, i_ch0s, n_tr)
        else:
            assert False, 'Type error'

except ImportError:
    print '* Using python concat_core_py'
    concat_core = concat_core_py


###################################################
def featconcat(ifns, ofn, basename=False, include_iid=None, include_lbl=None,
        trim=False, correct_ntr=True, backtrim=False, outrule='default',
        inrule=None, evoked=None, whitened=False, fnrule=None, sortmeta=True,
        sortmetanum=False):
    # -- sweep the files and determine the layout
    print '* Sweeping...'

    iid2idx = {}
    idx2iid = []
    lbl2idx = {}
    idx2lbl = []
    src2idx = {}
    idx2src = []
    n_tr = 0
    n_trs = {}
    n_elec = get_total_elec(outrule)

    if fnrule is not None:
        assert len(fnrule) == len(ifns)

    all_uint16 = True
    for i_f, ifn in enumerate(ifns):
        inrule0 = ifn
        if inrule is not None:
            inrule0 = inrule
        if fnrule is None:
            get_chconv(inrule0, outrule)   # test if valid name
        else:
            assert len(fnrule[i_f]) == 2
        h5 = tables.openFile(ifn)
        _, n_tr0, _, _ = h5.root.spk.shape
        idx2iid0 = safe_list(pk.loads(h5.root.meta.idx2iid_pk.read()))
        idx2lbl0 = safe_list(pk.loads(h5.root.meta.idx2lbl_pk.read()))
        idx2src0 = safe_list(h5.root.meta.srcfiles.read())
        if h5.root.spk.atom != tables.UInt16Atom():
            all_uint16 = False
        print 'At (%d/%d): %s: n_images=%d' % \
                (i_f + 1, len(ifns), ifn, len(idx2iid0))

        lbl_valid = []
        iid_valid = []

        for src0 in idx2src0:
            if basename:
                src = os.path.basename(src0)
            else:
                src = src0
            if src not in src2idx:
                src2idx[src] = len(idx2src)
                idx2src.append(src)

        for iid0 in idx2iid0:
            if basename:
                iid = os.path.basename(iid0)
            else:
                iid = iid0

            if iid not in iid2idx:
                if include_iid is not None and (
                        not any([patt in iid for patt in include_iid])):
                    continue
                iid2idx[iid] = len(idx2iid)
                idx2iid.append(iid)
            iid_valid.append(iid)

        for lbl in idx2lbl0:
            if lbl not in lbl2idx:
                if include_lbl is not None and lbl not in include_lbl:
                    continue
                lbl2idx[lbl] = len(idx2lbl)
                idx2lbl.append(lbl)
            lbl_valid.append(lbl)

        for lbl in lbl_valid:
            for iid in iid_valid:
                if (lbl, iid) not in n_trs:
                    n_trs[(lbl, iid)] = 0
                n_trs[(lbl, iid)] += n_tr0
        h5.close()

    # for now, sort only iids
    if sortmetanum:
        si = sorted([(int(e.split('_')[-1]), e) for e in iid2idx.keys()])
        idx2iid = [e[1] for e in si]
        for idx, iid in enumerate(idx2iid):
            iid2idx[iid] = idx
    elif sortmeta:
        idx2iid = sorted(iid2idx.keys())
        for idx, iid in enumerate(idx2iid):
            iid2idx[iid] = idx

    # -- layout output hdf5 file
    n_lbl = len(idx2lbl)
    n_img = len(idx2iid)
    n_tr = int(max([n_trs[k] for k in n_trs]))

    # to be written at the end
    shape = (n_lbl, n_tr, n_img, n_elec)

    if evoked is not None:
        all_uint16 = False
    if whitened:
        all_uint16 = False

    if all_uint16:
        spk = np.zeros(shape, 'uint16')
    else:
        spk = np.zeros(shape, 'float32')
    ntrimg = np.zeros((n_lbl, n_img, n_elec), 'uint16')
    orgfile = np.zeros(shape, 'int16')
    orgfile[...] = -1

    print
    print '* Shape =', shape
    print '* All uint16?', all_uint16

    #fd, tmpf = tempfile.mkstemp()
    #os.close(fd)           # hdf5 module will handle the file. close it now.

    # -- concatenating
    print '* Concatenating...'
    for i_f, ifn in enumerate(ifns):
        print 'At (%d/%d): %s' % (i_f + 1, len(ifns), ifn)
        h5 = tables.openFile(ifn)
        lbl2idx0 = pk.loads(h5.root.meta.lbl2idx_pk.read())
        iid2idx0 = pk.loads(h5.root.meta.iid2idx_pk.read())
        idx2iid0 = safe_list(pk.loads(h5.root.meta.idx2iid_pk.read()))
        idx2src0 = h5.root.meta.srcfiles.read()
        ntrials0 = h5.root.meta.ntrials_img.read()
        orgfile0 = h5.root.meta.orgfile.read()
        inrule0 = ifn
        if inrule is not None:
            inrule0 = inrule
        if fnrule is None:
            i_ch0s, i_chs = get_chconv(inrule0, outrule)
        else:
            i_ch0s, i_chs = fnrule[i_f]
        spk0 = h5.root.spk.read()
        if not all_uint16:
            spk0 = np.array(spk0, 'float32')
        n_tr0 = spk0.shape[I_TR]   # assume that all iids have same #trials

        if correct_ntr:
            ntrials0[n_tr0 < ntrials0] = n_tr0

        if basename:
            tmp = {}
            for k in iid2idx0:
                tmp[os.path.basename(k)] = iid2idx0[k]
            iid2idx0 = tmp
            tmpl = []
            for k in idx2src0:
                tmpl.append(os.path.basename(k))
            idx2src0 = tmpl

        # compute evoked rate if requested
        if evoked is not None:
            print '--> Compute evoked response...'
            i_src0s = range(len(idx2src0))
            if type(evoked) is list:
                i_bgs = [i for i, e in enumerate(idx2iid0) if e in evoked]
            else:
                i_bgs = [i for i, e in enumerate(idx2iid0) if evoked in e]
            print '    * # bgs =', len(i_bgs)
            assert len(i_bgs) > 0
            get_evoked(spk0, orgfile0, i_bgs, i_src0s, i_ch0s)

        if whitened:
            print '--> Whitening...'
            i_src0s = range(len(idx2src0))
            get_whitened(spk0, orgfile0, i_src0s, i_ch0s)

        # take orgfile0 and map each entry to the corresponding
        # entry in (global) src2idx
        if len(ifns) == 1:
            orgfile_t = orgfile0
        else:
            srcmap = [-1]
            for fsrc0 in idx2src0:
                srcmap.append(src2idx[fsrc0])
            srcmap = np.array(srcmap, 'int16')
            orgfile_t = srcmap.take(orgfile0 + 1)

        # map local lbl to global lbl
        i_lbls = []
        i_lbl0s = []
        for i_lbl, lbl in enumerate(idx2lbl):
            i_lbls.append(i_lbl)
            if lbl not in lbl2idx0:
                i_lbl0s.append(-1)
            else:
                i_lbl0s.append(lbl2idx0[lbl])

        # map local iid to global iid
        i_iids = []
        i_iid0s = []
        for i_iid, iid in enumerate(idx2iid):
            i_iids.append(i_iid)
            if iid not in iid2idx0:
                i_iid0s.append(-1)
            else:
                i_iid0s.append(iid2idx0[iid])

        # for caching
        i_lbls = np.array(i_lbls, 'int32')
        i_lbl0s = np.array(i_lbl0s, 'int32')
        i_iids = np.array(i_iids, 'int32')
        i_iid0s = np.array(i_iid0s, 'int32')
        i_chs = np.array(i_chs, 'int32')
        i_ch0s = np.array(i_ch0s, 'int32')

        #if _debug: t0 = time.time()
        concat_core(spk, spk0, ntrimg, ntrials0, orgfile, orgfile_t,
                i_lbls, i_lbl0s, i_iids, i_iid0s, i_chs, i_ch0s, n_tr)
        #if _debug: print '--> %fs' % (time.time() - t0)
        h5.close()

    # -- finish up
    if trim:
        n_tr_new = np.min(ntrimg)
        ntrimg[n_tr_new < ntrimg] = n_tr_new
    else:
        n_tr_new = np.max(ntrimg)
    shape = (n_lbl, n_tr_new, n_img, n_elec)

    filters = tables.Filters(complevel=4, complib='blosc')
    h5o = tables.openFile(ofn, 'w')
    if all_uint16:
        spk_h5 = h5o.createCArray(h5o.root, 'spk', tables.UInt16Atom(),
                shape, filters=filters)
    else:
        spk_h5 = h5o.createCArray(h5o.root, 'spk', tables.Float32Atom(),
                shape, filters=filters)
    meta = h5o.createGroup("/", 'meta', 'Metadata')
    h5o.createArray(meta, 'iid2idx_pk', pk.dumps(iid2idx))
    h5o.createArray(meta, 'idx2iid_pk', pk.dumps(idx2iid))
    h5o.createArray(meta, 'lbl2idx_pk', pk.dumps(lbl2idx))
    h5o.createArray(meta, 'idx2lbl_pk', pk.dumps(idx2lbl))
    h5o.createArray(meta, 'idx2lbl', idx2lbl)
    h5o.createArray(meta, 'idx2iid', idx2iid)
    h5o.createArray(meta, 'srcfiles', idx2src)
    h5o.createArray(meta, 'ntrials_img', ntrimg)
    orgfile_h5 = h5o.createCArray(meta, 'orgfile', tables.Int16Atom(), shape,
            filters=filters)
    if backtrim:
        spk_h5[...] = spk[:, -n_tr_new:, :, :]
        orgfile_h5[...] = orgfile[:, -n_tr_new:, :, :]
    else:
        spk_h5[...] = spk[:, :n_tr_new, :, :]
        orgfile_h5[...] = orgfile[:, :n_tr_new, :, :]
    h5o.close()
