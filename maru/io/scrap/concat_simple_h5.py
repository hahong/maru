#!/usr/bin/env python
import sys
sys.path.append('lib')
import tables
import comm_io_funcs as cio
import numpy as np
import cPickle as pk
import os

I_TR = 1   # index for trial, in /spk

# for compatibility issue
def safe_list(iterable):
    if type(iterable) == list: return iterable
    l = []
    for k in sorted(iterable.keys()):
        l.append(iterable[k])
    return l


# TODO: i_addaxis is HALF-BAKED.  Now this script only works along the 0-th axis!
def concat(ifns, ofn, basename=False, include_iid=None, include_lbl=None, i_addaxis=0):
    # -- sweep the files and determine the layout
    print '* Sweeping...'

    iid2idx = {}
    idx2iid = []
    lbl2idx = {}
    idx2lbl = []
    n_trs = {}
    shape = None

    for i_f, ifn in enumerate(ifns):
        h5 = tables.openFile(ifn)
        shape0 = list(h5.root.spk.shape)
        if shape is None: shape = shape0
        else: shape[i_addaxis] += shape0[i_addaxis]

        idx2iid0 = safe_list(pk.loads(h5.root.meta.idx2iid_pk.read()))
        idx2lbl0 = safe_list(pk.loads(h5.root.meta.idx2lbl_pk.read()))
        print 'At (%d/%d): %s: n_images=%d' % (i_f+1, len(ifns), ifn, len(idx2iid0))

        lbl_valid = []
        iid_valid = []

        for iid0 in idx2iid0:
            if basename: iid = os.path.basename(iid0)
            else: iid = iid0

            if iid not in iid2idx:
                if include_iid != None and (not any([patt in iid for patt in include_iid])): continue
                iid2idx[iid] = len(idx2iid)
                idx2iid.append(iid)
            iid_valid.append(iid)

        for lbl in idx2lbl0:
            if lbl not in lbl2idx:
                if include_lbl != None and lbl not in include_lbl: continue
                lbl2idx[lbl] = len(idx2lbl)
                idx2lbl.append(lbl)
            lbl_valid.append(lbl)

        for lbl in lbl_valid:
            for iid in iid_valid:
                if (lbl, iid) not in n_trs: n_trs[(lbl, iid)] = 0
                n_trs[(lbl, iid)] += 1
        h5.close()


    # -- layout output hdf5 file
    n_lbl = len(idx2lbl)
    n_img = len(idx2iid)

    shape = tuple(shape)
    filters = tables.Filters(complevel=4, complib='blosc')

    print 
    print '* Shape =', shape

    h5o = tables.openFile(ofn, 'w')
    #spk = h5o.createCArray(h5o.root, 'spk', tables.UInt16Atom(), shape, filters=filters)
    spk = h5o.createCArray(h5o.root, 'spk', tables.Float32Atom(), shape, filters=filters)   # XXX: kludge code for simple concatenation!
    meta = h5o.createGroup("/", 'meta', 'Metadata')
    h5o.createArray(meta, 'iid2idx_pk', pk.dumps(iid2idx))
    h5o.createArray(meta, 'idx2iid_pk', pk.dumps(idx2iid))
    h5o.createArray(meta, 'lbl2idx_pk', pk.dumps(lbl2idx))
    h5o.createArray(meta, 'idx2lbl_pk', pk.dumps(idx2lbl))
    h5o.createArray(meta, 'idx2lbl', idx2lbl)
    h5o.createArray(meta, 'idx2iid', idx2iid)

    # -- concatenating
    print '* Concatenating...'
    for i_f, ifn in enumerate(ifns):
        print 'At (%d/%d): %s' % (i_f+1, len(ifns), ifn)
        h5 = tables.openFile(ifn)
        spk0 = h5.root.spk
        lbl2idx0 = pk.loads(h5.root.meta.lbl2idx_pk.read())
        iid2idx0 = pk.loads(h5.root.meta.iid2idx_pk.read())
        if basename:
            tmp = {}
            for k in iid2idx0: tmp[os.path.basename(k)] = iid2idx0[k]
            iid2idx0 = tmp

        """
        for i_lbl, lbl in enumerate(idx2lbl):
            if lbl not in lbl2idx0: continue
            i_lbl0 = lbl2idx0[lbl]
        """

        for i_iid, iid in enumerate(idx2iid):
            if iid not in iid2idx0: continue
            i_iid0 = iid2idx0[iid]
            #print ' --> %d, %d    \r' % (i_lbl0, i_iid0), 
            #sys.stdout.flush()
            spk[i_iid] = spk0[i_iid0]

        h5.close()
    # -- finish up
    h5o.close()



# ----------
def main(argv):
    args, opts = cio.parse_opts(argv[1:]) 
    if len(args) != 2:
        print 'concat_h5.py <input repr files or list> <output repr file> [opts]'
        print 'Input repr h5 files can be comma separated.'
        print
        print 'Options:'
        print '--basename                         Only take basename of image ids'
        print '--include_iid=<files or +list>     Image ids to be included (others are discarded)'
        print '--include_lbl=<python expr>        Labels to be included (others are discarded)'
        print
        sys.exit(0)

    ifns, ofn = args
    ifns = cio.prep_files(ifns) 

    # -- process options
    basename = False
    if 'basename' in opts:
        basename = True
        print '** Taking basename'

    include_iid = None
    if 'include_iid' in opts:
        include_iid = cio.prep_files(opts['include_iid'], extchk=False)
        print '** Only include these iids:', include_iid

    include_lbl = None
    if 'include_lbl' in opts:
        include_lbl = eval(opts['include_lbl'])
        print '** Only include these lbls:', include_lbl

    concat(ifns, ofn, basename=basename, include_iid=include_iid, include_lbl=include_lbl)

if __name__ == '__main__':
    main(sys.argv)
