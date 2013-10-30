#!/usr/bin/env python
import sys
import numpy as np
import cPickle as pk
import tables
import signal
import tempfile
import os
from ..utils import makeavail

N_SLACK = 32


# Housekeeping functions -----------------------------------------------------
def save_tinfo_cleanup():
    # close all hdf5's
    for h5, name in [(save_tinfo_core.h5o, 'output file'),
            (save_tinfo_core.h5t, 'temp file')]:
        if h5 is not None and h5.isopen:
            print 'Closing', name
            h5.close()

    tmpf = save_tinfo_core.tmpf
    if os.path.isfile(tmpf):
        print 'Removing tempfile', tmpf
        os.unlink(tmpf)


def save_tinfo_signal_handler(sig, frame):
    print 'Got abort signal...'
    save_tinfo_cleanup()
    sys.exit(0)


# Main working functions -----------------------------------------------------
class IidIdx(tables.IsDescription):
    iid = tables.StringCol(512)      # string repr of iid
    iid_pk = tables.StringCol(1024)  # pickled repr of iid
    idx = tables.UInt32Col()         # index to the iid


def is_excluded(iid, exclude_img=None):
    if exclude_img is None:
        return False

    for patt in exclude_img:
        # if it's in the excluded list, don't put that one!!
        if patt in iid:
            return True

    return False


def save_tinfo_core(dat, outfn, n_img=None, n_maxtrial=None, save_spktch=False,
        n_elec=None, exclude_img=None, n_bins=None, t_min=None, t_max=None,
        verbose=1, n_slack=N_SLACK, t_adjust=None):
    iid2idx = {}                         # image id to index (1-th axis) table
    idx2iid = []                         # vice versa
    ch2idx = {}
    idx2ch = []

    # prepare tmp file
    fd, tmpf = tempfile.mkstemp()
    os.close(fd)             # hdf5 module will handle the file. close it now.
    save_tinfo_core.tmpf = tmpf
    frame_onset = None

    foffset_chidx = []
    foffset_imgidx = []
    foffset_tridx = []
    foffset_binidx = []
    foffset_pos = []

    # -- initialization
    fn_nominal = dat.get('filename', '__none__')
    fns_nominal = [fn_nominal]  # backward compatibility

    if n_img is None:
        # if `n_img` is not specified, determine the number of images
        # from the first psf.pk file. (with additional n_slack)
        el0 = dat['all_spike'].keys()[0]
        n_img = len(dat['all_spike'][el0]) + n_slack
    if n_elec is None:
        # if `n_elec` is not specified, determine the number of
        # electrodes from the first psf.pk file.
        # No additinal n_slack here!
        n_elec = len(dat['actvelecs'])
    if n_maxtrial is None:
        el0 = dat['all_spike'].keys()[0]
        ii0 = dat['all_spike'][el0].keys()[0]
        n_maxtrial = len(dat['all_spike'][el0][ii0]) + n_slack
    if t_min is None:
        t_min = dat['t_start']
    if t_max is None:
        t_max = dat['t_stop']
    if t_adjust is None:
        t_adjust = dat['t_adjust']
    if n_bins is None:
        n_bins = int(np.ceil((t_max - t_min) / 1000.) + 1)

    # number of bytes required for 1 trial
    n_bytes = int(np.ceil(n_bins / 8.))
    shape = (n_elec, n_img, n_maxtrial, n_bytes)
    shape_org = (n_img, n_elec, n_maxtrial)
    atom = tables.UInt8Atom()
    atom16 = tables.Int16Atom()
    atomu16 = tables.UInt16Atom()
    atomu32 = tables.UInt32Atom()
    atom64 = tables.Int64Atom()
    filters = tables.Filters(complevel=4, complib='blosc')

    save_tinfo_core.h5t = h5t = tables.openFile(tmpf, 'w')
    db = h5t.createCArray(h5t.root, 'db',
            atom, shape, filters=filters)    # spiking information
    org = h5t.createCArray(h5t.root, 'org',
            atom16, shape_org, filters=filters)   # origin info
    org[...] = -1
    tr = np.zeros((n_elec, n_img),
            dtype=np.uint16)    # num of trials per each ch & image

    if verbose > 0:
        print '* Allocated: (n_elec,'\
            ' n_img, n_maxtrial, n_bytes) = (%d, %d, %d, %d)' % shape
        print '* Temp hdf5:', tmpf

    # ----------------------------------------------------------------------
    # -- read thru the dats, store into the tmp.hdf5 file (= tmpf)
    # -- actual conversion for this file happens here
    for ch in sorted(dat['all_spike']):
        makeavail(ch, ch2idx, idx2ch)
        ie = ch2idx[ch]   # index to the electrode, 0-based

        if verbose > 0:
            print '* At: Ch/site/unit %d                          \r' % ie,
            sys.stdout.flush()

        for iid in sorted(dat['all_spike'][ch]):
            # -- main computation
            if is_excluded(iid, exclude_img):
                continue

            # do the conversion
            makeavail(iid, iid2idx, idx2iid)
            ii = iid2idx[iid]      # index to the image, 0-based
            trials = dat['all_spike'][ch][iid]   # get the chunk
            foffsets = None
            if 'all_foffset' in dat:
                foffsets = dat['all_foffset'][ch][iid]
                if len(trials) != len(foffsets):
                    foffsets = None

            ntr0 = len(trials)     # number of trials in the chunk
            itb = tr[ie, ii]       # index to the beginning trial#, 0-based
            ite = itb + ntr0       # index to the end
            n_excess = 0           # number of excess trials in this chunk
            if ite > n_maxtrial:
                n_excess = ite - n_maxtrial
                ite = n_maxtrial
                if verbose > 0:
                    print '** Reached n_maxtrial(=%d): ch=%s, iid=%s' % \
                            (n_maxtrial, str(ch), str(iid))
            # number of actual trials to read in the chunk
            ntr = ntr0 - n_excess
            # bit-like spike timing info
            tr_bits = np.zeros((ntr, n_bytes * 8), dtype=np.uint8)

            # sweep the chunk, and bit-pack the data
            trials = trials[:ntr]
            trials_enum = np.concatenate([[i] * len(e)
                    for i, e in enumerate(trials)])
            trials = np.concatenate(trials)

            # selected bins
            sb = np.round((trials - t_min) / 1000.).astype('int')
            si = np.nonzero((sb >= 0) & (sb < n_bins))[0]
            sb = sb[si]
            st = trials_enum[si]
            tr_bits[st, sb] = 1   # there was a spike
            spk = np.packbits(tr_bits, axis=1)

            if foffsets is not None:
                foffsets = np.concatenate(foffsets)
                if len(foffsets) != len(trials):
                    # shouldn't happen
                    print '** Length of foffsets and trials is different'
                    foffsets = [-1] * len(trials)
                    foffsets = np.array(foffsets)

                nevs = len(sb)
                foffset_chidx.extend([ie] * nevs)
                foffset_imgidx.extend([ii] * nevs)
                foffset_tridx.extend(st)
                foffset_binidx.extend(sb)
                foffset_pos.extend(foffsets[si])

            # finished this image in this electrode; store the data
            db[ie, ii, itb:ite, :] = spk
            org[ii, ie, itb:ite] = 0   # mainly for backward compatibility
            tr[ie, ii] += ntr

    # -- additional movie data conversion
    # XXX: this assumes `multi=False`
    if 'frame_onset' in dat and len(dat['frame_onset']) > 0:
        print '* Collecting frame onset info'
        if frame_onset is None:
            frame_onset = dat['frame_onset']
        else:
            frame_onset0 = dat['frame_onset']
            for iid in frame_onset0:
                frame_onset[iid].extend(frame_onset0[iid])

    # ----------------------------------------------------------------------
    # -- finished main conversion; now save into a new optimized hdf5 file
    n_img_ac = len(iid2idx)   # actual number of images
    n_tr_ac = np.max(tr)      # actual maximum number of trials

    shape_img = (n_img_ac, n_elec, n_tr_ac, n_bytes)   # img-major form
    shape_ch = (n_elec, n_img_ac, n_tr_ac, n_bytes)    # ch-major form
    shape_org = (n_img_ac, n_elec, n_tr_ac)

    if verbose > 0:
        print 'Optimizing...                                 '
        print '* Actual #images:', n_img_ac
        print '* Actual #trials:', n_tr_ac
        print '* New allocated: (n_elec, n_img, n_maxtrial, n_bytes)' \
                ' = (%d, %d, %d, %d)' % shape_ch

    # -- layout output hdf5 file
    save_tinfo_core.h5o = h5o = tables.openFile(outfn, 'w')
    # /spktimg: bit-packed spike-time info matrix, image-id-major
    spktimg = h5o.createCArray(h5o.root, 'spkt_img',
            atom, shape_img, filters=filters)
    # /meta: metadata group
    meta = h5o.createGroup("/", 'meta', 'Metadata')
    # /meta/iid2idx: iid to matrix-index info
    t_iididx = h5o.createTable(meta, 'iididx', IidIdx,
            'Image ID and its index')
    # /meta/orgfile_img: file origin info, image-id-major
    orgfile = h5o.createCArray(meta, 'orgfile_img',
            atom16, shape_org, filters=filters)   # origin info

    # -- fill metadata
    # some metadata records
    h5o.createArray(meta, 'srcfiles', fns_nominal)
    h5o.createArray(meta, 'nbins', n_bins)
    h5o.createArray(meta, 't_start0', t_min)
    h5o.createArray(meta, 'tmin', t_min)         # backward compatibility
    h5o.createArray(meta, 't_stop0', t_max)
    h5o.createArray(meta, 'tmax', t_max)         # backward compatibility
    h5o.createArray(meta, 't_adjust', t_adjust)
    h5o.createArray(meta, 'iid2idx_pk', pk.dumps(iid2idx))
    h5o.createArray(meta, 'idx2iid_pk', pk.dumps(idx2iid))
    h5o.createArray(meta, 'idx2iid', idx2iid)
    h5o.createArray(meta, 'ch2idx_pk', pk.dumps(ch2idx))
    h5o.createArray(meta, 'idx2ch', idx2ch)
    # save as img-major order (tr is in channel-major)
    h5o.createArray(meta, 'ntrials_img', tr[:, :n_img_ac].T)
    h5o.createArray(meta, 'frame_onset_pk', pk.dumps(frame_onset))
    # cluster related stuffs
    for clu_k in ['idx2gcid', 'cid_sel', 'gcid2idx']:
        if clu_k not in dat:
            continue
        h5o.createArray(meta, clu_k + '_pk', pk.dumps(dat[clu_k]))
    # this is deprecated.  mainly for backward compatibility
    orgfile[...] = org[:n_img_ac, :, :n_tr_ac]

    # populate /meta/iididx
    r = t_iididx.row
    for iid in iid2idx:
        r['iid'] = str(iid)
        r['iid_pk'] = pk.dumps(iid)
        r['idx'] = iid2idx[iid]
        r.append()
    t_iididx.flush()

    # -- store spiking time data
    for i in xrange(n_img_ac):
        if verbose > 0:
            print '* At: Image %d                          \r' % i,
            sys.stdout.flush()
        spktimg[i, :, :, :] = db[:, i, :n_tr_ac, :]

    if save_spktch:
        # /spktch: bit-packed spike-time info matrix, channel-major
        spktch = h5o.createCArray(h5o.root, 'spkt_ch',
                atom, shape_ch, filters=filters)
        for i in xrange(n_elec):
            if verbose > 0:
                print '* At: Ch/site/unit %d                   \r' % i,
                sys.stdout.flush()
            spktch[i, :, :, :] = db[i, :n_img_ac, :n_tr_ac, :]

    # foffset stuffs
    foffset_chidx = np.array(foffset_chidx, dtype='uint16')
    foffset_imgidx = np.array(foffset_imgidx, dtype='uint32')
    foffset_tridx = np.array(foffset_tridx, dtype='uint16')
    foffset_binidx = np.array(foffset_binidx, dtype='uint16')
    foffset_pos = np.array(foffset_pos, dtype='int64')

    for src, name, atom0 in zip(
            [foffset_chidx, foffset_imgidx, foffset_tridx,
                foffset_binidx, foffset_pos],
            ['foffset_chidx', 'foffset_imgidx', 'foffset_tridx',
                'foffset_binidx', 'foffset_pos'],
            [atomu16, atomu32, atomu16, atomu16, atom64]
            ):
        if len(src) == 0:
            continue
        dst = h5o.createCArray(meta, name,
            atom0, src.shape, filters=filters)
        dst[:] = src[:]

    if verbose > 0:
        print

    h5o.close()
    h5t.close()
save_tinfo_core.tmpf = None
save_tinfo_core.h5o = None
save_tinfo_core.h5t = None


def save_tinfo(dat, outfn, n_img=None, n_maxtrial=None,
        n_elec=None, exclude_img=None, n_bins=None, t_min=None,
        verbose=1, n_slack=N_SLACK):
    try:
        signal.signal(signal.SIGINT, save_tinfo_signal_handler)
        signal.signal(signal.SIGTERM, save_tinfo_signal_handler)
        save_tinfo_core(dat, outfn,
                exclude_img=exclude_img, n_maxtrial=n_maxtrial, n_bins=n_bins)
    finally:
        save_tinfo_cleanup()
