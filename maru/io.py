#!/usr/bin/env python
import sys
import numpy as np
import cPickle as pk
import tables
import signal
import tempfile
import os
from .utils import makeavail

N_MAXTRIAL = 35
N_SLACK = 32
FKEY = ['_A_', '_M_', '_P_']   # deprecated: not supported anymore

USAGE = \
"""collect_PS_tinfo.py [options] <out prefix> <in1.psf.pk> [in2.psf.pk] ...
Collects peri-stimuli spike time info from *.psf.pk (of collect_PS_firing.py).
The output file is in the hdf5 format.

Options:
   --n_elec=#            Number of electrodes/channels/units/sites
   --n_maxtrial=#        Hard maximum of the number of trials
   --n_img=#             Hard maximum of the number of images/stimuli
   --n_bins=#            Number of 1ms-bins
   --t_min=#             Low-bound of the spike time for stimuli
   --multi               Merge multiple array data (A, M, P) into one output
   --key=string          Comma separated keys for each array (only w/ --multi)
   --exclude_img=string  Comma separated image names to be excluded
   --flist=string        CR separated input psf.pk file list to be added
   --flistpx=string      (Path) prefix to be added before each line in `flist`
"""


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


def save_tinfo_core(dats, outfn, n_img=None, n_maxtrial=N_MAXTRIAL,
        n_elec=None, exclude_img=None, n_bins=None, t_min=None, t_max=None,
        verbose=1, n_slack=N_SLACK, multi=False, fkey=FKEY, t_adjust=None):
    n_files = len(dats)
    iid2idx = {}                         # image id to index (1-th axis) table
    idx2iid = []                         # vice versa
    initialized = False
    fns_nominal = []

    # prepare tmp file
    fd, tmpf = tempfile.mkstemp()
    os.close(fd)             # hdf5 module will handle the file. close it now.
    save_tinfo_core.tmpf = tmpf
    proc_cluster = None
    clus_info = None
    frame_onset = None

    foffset_fnum = []
    foffset_binnum = []
    foffset_elidx = []
    foffset_imgidx = []
    foffset_pos = []

    # -- read thru the dats, store into the tmp.hdf5 file (= tmpf)
    for i_f, dat in enumerate(dats):
        fn_nominal = dat.get('filename', '__none__')
        fns_nominal.append(fn_nominal)

        if verbose > 0:
            print 'At (%d/%d): %s' % (i_f + 1, n_files, fn_nominal)

        if proc_cluster is None:
            if 'max_clus' in dat and 'clus_info' in dat:
                proc_cluster = True
                max_clus = dat['max_clus']
                clus_info = dat['clus_info']
                print '* Collecting cluster info'
            else:
                proc_cluster = False
        elif proc_cluster:
            # DEPRECATED
            clus_info0 = dat['clus_info']
            for k in clus_info0:
                if k not in clus_info:
                    clus_info[k] = clus_info0[k]
                    continue
                assert clus_info[k]['nclusters'] == \
                        clus_info0[k]['nclusters']
                assert clus_info[k]['unsorted_cid'] == \
                        clus_info0[k]['unsorted_cid']
                clus_info[k]['nbad_isi'] += clus_info0[k]['nbad_isi']
                clus_info[k]['nspk'] += clus_info0[k]['nspk']

        # -- initialization
        if not initialized:
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
                # multi is now deprecated.
                if multi:              # if multiple array data are processed..
                    n_elec1 = n_elec   # number of electrode/ch per one array
                    n_elec *= len(fkey)
            if t_min is None:
                t_min = dat['t_start']
            if t_max is None:
                t_max = dat['t_stop']
            if t_adjust is None:
                t_adjust = dat['t_adjust']
            if n_bins is None:
                n_bins = t_max - t_min + 1

            # number of bytes required for 1 trial
            n_bytes = int(np.ceil(n_bins / 8.))
            shape = (n_elec, n_img, n_maxtrial, n_bytes)
            shape_org = (n_img, n_elec, n_maxtrial)
            atom = tables.UInt8Atom()
            atom16 = tables.Int16Atom()
            filters = tables.Filters(complevel=4, complib='blosc')

            save_tinfo_core.h5t = h5t = tables.openFile(tmpf, 'w')
            db = h5t.createCArray(h5t.root, 'db',
                    atom, shape, filters=filters)    # spiking information
            org = h5t.createCArray(h5t.root, 'org',
                    atom16, shape_org, filters=filters)   # origin info
            org[...] = -1
            tr = np.zeros((n_elec, n_img),
                    dtype=np.uint16)    # num of trials per each ch & image
            initialized = True

            if verbose > 0:
                print '* Allocated: (n_elec,'\
                    ' n_img, n_maxtrial, n_bytes) = (%d, %d, %d, %d)' % shape
                print '* Temp hdf5:', tmpf

        # bit-like spike timing info in one trial
        tr_bits = np.zeros((n_bytes * 8), dtype=np.uint8)
        eo = 0    # electrode offset (0-based)
        if multi:
            # multi is now DEPRECATED.
            found = False
            for i_k, k in enumerate(fkey):
                if k in fn_nominal:
                    found = True
                    break
            if not found:
                print '** Not matching file name:', fn_nominal
            eo = n_elec1 * i_k
            if verbose > 0:
                print '* Electrode offset: %d           ' % eo

        # -- actual conversion for this file
        for el in sorted(dat['all_spike']):
            fill_blank = False
            if proc_cluster:
                # DEPRECATED
                # ie: index to the electrode, 0-based
                ie = (el[0] - 1 + eo) * max_clus + el[1]
                if el not in clus_info:
                    ch1, i_unit = el
                    if i_unit >= clus_info[(ch1, 0)]['nclusters']:
                        continue

                    if verbose:
                        print '* Filling blanks:', el
                    fill_blank = True

            else:
                ie = el - 1 + eo  # index to the electrode, 0-based
            if verbose > 0:
                print '* At: Ch/site/unit %d                          \r' % ie,
                sys.stdout.flush()

            for iid in sorted(dat['all_spike'][el]):
                # -- main computation
                if is_excluded(iid, exclude_img):
                    continue

                # do the conversion
                makeavail(iid, iid2idx, idx2iid)
                ii = iid2idx[iid]      # index to the image, 0-based
                trials = dat['all_spike'][el][iid]   # get the chunk
                foffsets = None
                if 'all_foffset' in dat:
                    foffsets = dat['all_foffset'][el][iid]
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
                        print '** Reached n_maxtrial(=%d): el=%d, iid=%s' % \
                                (n_maxtrial, el, str(iid))
                # number of actual trials to read in the chunk
                ntr = ntr0 - n_excess
                # temprary spike conut holder
                spk = np.zeros((ntr, n_bytes), dtype=np.uint8)

                if not fill_blank:
                    # sweep the chunk, and bit-pack the data
                    for i in xrange(ntr):
                        trial = trials[i]
                        tr_bits[:] = 0

                        # bit-pack all the spiking times in `trial`
                        for t0 in trial:
                            # t0: spiking time relative to the
                            # onset of the stimulus
                            # The following converts us -> ms
                            # 0 at t_min
                            t = int(np.round((t0 - t_min) / 1000.))
                            if t < 0 or t >= n_bins:
                                continue
                            tr_bits[t] = 1   # set the bit

                        spk[i, :] = np.packbits(tr_bits)

                # finished this image in this electrode; store the data
                db[ie, ii, itb:ite, :] = spk
                org[ii, ie, itb:ite] = i_f
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
    # /spktch: bit-packed spike-time info matrix, channel-major
    spktch = h5o.createCArray(h5o.root, 'spkt_ch',
            atom, shape_ch, filters=filters)
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
    h5o.createArray(meta, 't_stop0', t_max)
    h5o.createArray(meta, 't_adjust', t_adjust)
    h5o.createArray(meta, 'iid2idx_pk', pk.dumps(iid2idx))
    h5o.createArray(meta, 'idx2iid_pk', pk.dumps(idx2iid))
    h5o.createArray(meta, 'idx2iid', idx2iid)
    # save as img-major order (tr is in channel-major)
    h5o.createArray(meta, 'ntrials_img', tr[:, :n_img_ac].T)
    h5o.createArray(meta, 'frame_onset_pk', pk.dumps(frame_onset))
    # this is deprecated.  mainly for backward compatibility
    h5o.createArray(meta, 'clus_info_pk', pk.dumps(clus_info))
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

    for i in xrange(n_elec):
        if verbose > 0:
            print '* At: Ch/site/unit %d                   \r' % i,
            sys.stdout.flush()
        spktch[i, :, :, :] = db[i, :n_img_ac, :n_tr_ac, :]
    if verbose > 0:
        print

    h5o.close()
    h5t.close()
save_tinfo_core.tmpf = None
save_tinfo_core.h5o = None
save_tinfo_core.h5t = None


def save_tinfo(dats, outfn, n_img=None, n_maxtrial=N_MAXTRIAL,
        n_elec=None, exclude_img=None, n_bins=None, t_min=None,
        verbose=1, n_slack=N_SLACK, multi=False, fkey=FKEY):
    try:
        signal.signal(signal.SIGINT, save_tinfo_signal_handler)
        signal.signal(signal.SIGTERM, save_tinfo_signal_handler)
        save_tinfo_core(dats, outfn, multi=multi, fkey=fkey,
                exclude_img=exclude_img, n_maxtrial=n_maxtrial, n_bins=n_bins)
    finally:
        save_tinfo_cleanup()
