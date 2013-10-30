#!/usr/bin/env python
import sys
import numpy as np
import cPickle as pk
import tables
import signal
import tempfile
import os
from common_fn import parse_opts2

T_MIN = -100000
N_BINS = 500
N_MAXTRIAL = 35
N_SLACK = 32
FKEY = ['_A_', '_M_', '_P_']

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
def cleanup():
    # close all hdf5's
    for h5, name in [(convert.h5o, 'output file'), \
            (convert.h5t, 'temp file')]:
        if h5 != None and h5.isopen:
            print 'Closing', name
            h5.close()

    tmpf = convert.tmpf
    if os.path.isfile(tmpf):
        print 'Removing tempfile', tmpf
        os.unlink(tmpf)


def signal_handler(sig, frame):
    print 'Got abort signal...'
    cleanup()
    sys.exit(0)


# Main working functions -----------------------------------------------------
class IidIdx(tables.IsDescription):
    iid = tables.StringCol(512)      # string repr of iid
    iid_pk = tables.StringCol(1024)  # pickled repr of iid
    idx = tables.UInt32Col()         # index to the iid


def is_excluded(iid, exclude_img=None):
    if exclude_img == None:
        return False

    for patt in exclude_img:
        # if it's in the excluded list, don't put that one!!
        if patt in iid:
            return True

    return False


def convert(files, opref, n_img=None, n_maxtrial=N_MAXTRIAL, \
        n_elec=None, exclude_img=None, n_bins=N_BINS, t_min=T_MIN, \
        verbose=1, n_slack=N_SLACK, multi=False, fkey=FKEY):
    n_files = len(files)
    n_bytes = int(np.ceil(n_bins / 8.))  # number of bytes required for 1 trial
    iid2idx = {}                         # image id to index (1-th axis) table
    idx2iid = {}                         # vice versa
    initialized = False

    # prepare tmp file
    fd, tmpf = tempfile.mkstemp()
    os.close(fd)             # hdf5 module will handle the file. close it now.
    convert.tmpf = tmpf
    proc_cluster = None
    clus_info = None
    frame_onset = None

    # -- read thru the files, store into the tmp.hdf5 file (= tmpf)
    for i_f, f in enumerate(files):
        if verbose > 0:
            print 'At (%d/%d): %s' % (i_f + 1, n_files, f)
        dat = pk.load(open(f))

        if proc_cluster == None:
            if 'max_clus' in dat and 'clus_info' in dat:
                proc_cluster = True
                max_clus = dat['max_clus']
                clus_info = dat['clus_info']
                print '* Collecting cluster info'
            else:
                proc_cluster = False
        elif proc_cluster:
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
            if n_img == None:
                # if `n_img` is not specified, determine the number of images
                # from the first psf.pk file. (with additional n_slack)
                el0 = dat['all_spike'].keys()[0]
                n_img = len(dat['all_spike'][el0]) + n_slack
            if n_elec == None:
                # if `n_elec` is not specified, determine the number of
                # electrodes from the first psf.pk file.
                # No additinal n_slack here!
                n_elec = len(dat['all_spike'])
                if multi:              # if multiple array data are processed..
                    n_elec1 = n_elec   # number of electrode/ch per one array
                    n_elec *= len(fkey)

            shape = (n_elec, n_img, n_maxtrial, n_bytes)
            shape_org = (n_img, n_elec, n_maxtrial)
            atom = tables.UInt8Atom()
            atom16 = tables.Int16Atom()
            filters = tables.Filters(complevel=4, complib='blosc')

            convert.h5t = h5t = tables.openFile(tmpf, 'w')
            db = h5t.createCArray(h5t.root, 'db', \
                    atom, shape, filters=filters)    # spiking information
            org = h5t.createCArray(h5t.root, 'org', \
                    atom16, shape_org, filters=filters)   # origin info
            org[...] = -1
            tr = np.zeros((n_elec, n_img), \
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
            found = False
            for i_k, k in enumerate(fkey):
                if k in f:
                    found = True
                    break
            if not found:
                print '** Not matching file name:', f
            eo = n_elec1 * i_k
            if verbose > 0:
                print '* Electrode offset: %d           ' % eo

        # -- actual conversion for this file
        for el in sorted(dat['all_spike']):
            fill_blank = False
            if proc_cluster:
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

            for iid in dat['all_spike'][el]:
                # -- main computation
                if is_excluded(iid, exclude_img):
                    continue

                # some preps
                if iid not in iid2idx:
                    # iid2idx: index to the image, 0-based
                    iid2idx[iid] = ii = len(iid2idx)
                    idx2iid[ii] = iid

                # do the conversion
                ii = iid2idx[iid]      # index to the image, 0-based
                trials = dat['all_spike'][el][iid]   # get the chunk

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
        if 'frame_onset' in dat:
            print '* Collecting frame onset info'
            if frame_onset == None:
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
    convert.h5o = h5o = tables.openFile(opref + '.h5', 'w')
    # /spktimg: bit-packed spike-time info matrix, image-id-major
    spktimg = h5o.createCArray(h5o.root, 'spkt_img', \
            atom, shape_img, filters=filters)
    # /spktch: bit-packed spike-time info matrix, channel-major
    spktch = h5o.createCArray(h5o.root, 'spkt_ch', \
            atom, shape_ch, filters=filters)
    # /meta: metadata group
    meta = h5o.createGroup("/", 'meta', 'Metadata')
    # /meta/iid2idx: iid to matrix-index info
    t_iididx = h5o.createTable(meta, 'iididx', IidIdx, \
            'Image ID and its index')
    # /meta/orgfile_img: file origin info, image-id-major
    orgfile = h5o.createCArray(meta, 'orgfile_img', \
            atom16, shape_org, filters=filters)   # origin info

    # -- fill metadata
    # some metadata records
    h5o.createArray(meta, 'srcfiles', files)
    h5o.createArray(meta, 'nbins', n_bins)
    h5o.createArray(meta, 'tmin', t_min)
    h5o.createArray(meta, 'iid2idx_pk', pk.dumps(iid2idx))
    h5o.createArray(meta, 'idx2iid_pk', pk.dumps(idx2iid))
    # save as img-major order (tr is in channel-major)
    h5o.createArray(meta, 'ntrials_img', tr[:, :n_img_ac].T)
    h5o.createArray(meta, 'frame_onset_pk', pk.dumps(frame_onset))
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
convert.tmpf = None
convert.h5o = None
convert.h5t = None


# -----------------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print USAGE
        return

    args, opts = parse_opts2(sys.argv[1:])

    of = args[0]
    files = args[1:]
    print 'Output prefix:', of

    # -- process options
    if 'flist' in opts:
        flistpx = ''
        if 'flistpx' in opts:
            flistpx = opts['flistpx']

        for f in open(opts['flist']).readlines():
            fp = f.strip()
            if flistpx != '':
                fp = os.path.join(flistpx, fp)
            files.append(fp)
        print 'Using the followings:'
        for f in files:
            print '  ', f

    multi = False
    fkey = FKEY
    if 'multi' in opts:
        multi = True
        if 'key' in opts:
            fkey = opts['key'].split(',')
        print 'Enable multiple array data processing'
        print 'Multiple array data key:', fkey

        noerr = True
        for f in files:
            found = False
            for i_k, k in enumerate(fkey):
                if k in f:
                    found = True
                    break
            if not found:
                noerr = False
                print '** Not matching file name:', f

        if not noerr:
            print '** Error occured!'
            return

    exclude_img = None
    if 'exclude_img' in opts:
        exclude_img = opts['exclude_img'].split(',')
        print 'Exclude unwanted images:', exclude_img

    n_maxtrial = N_MAXTRIAL
    if 'n_maxtrial' in opts:
        n_maxtrial = int(opts['n_maxtrial'])
        print 'Setting n_maxtrial:', n_maxtrial

    n_bins = N_BINS
    if 'n_bins' in opts:
        n_bins = int(opts['n_bins'])
        print 'Setting n_bins:', n_bins

    assert all([os.path.exists(f) for f in files])

    # -- main loop
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        convert(files, of, multi=multi, fkey=fkey, \
                exclude_img=exclude_img, n_maxtrial=n_maxtrial, n_bins=n_bins)
    finally:
        print 'Cleaning up...'
        cleanup()

    print 'Done.'

if __name__ == '__main__':
    main()
