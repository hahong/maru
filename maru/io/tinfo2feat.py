#!/usr/bin/env python
import sys
sys.path.append('lib')
import tables
import comm_io_funcs as cio
import numpy as np
import cPickle as pk


def convert(ifn, ofn, tbs, tes, nooversized=True, preservefmt=False):
    h5 = tables.openFile(ifn)
    si = h5.root.spkt_img
    if preservefmt:
        n_tr = h5.root.spkt_img.shape[2]
    else:
        n_tr = np.min(h5.root.meta.ntrials_img.read())
    nbins = h5.root.meta.nbins.read()
    tmin = h5.root.meta.tmin.read() / 1000         # us -> ms
    tmax = tmin + nbins                            # XXX: assume 1bin = 1ms
    iid2idx_pk = h5.root.meta.iid2idx_pk.read()
    idx2iid_pk = h5.root.meta.idx2iid_pk.read()

    if nooversized:
        idx2lbl = [(tb, te) for tb, te in zip(tbs, tes) if tb >= tmin and te <= tmax]
    else:
        idx2lbl = zip(tbs, tes)

    lbl2idx = {}
    for i, tbte in enumerate(idx2lbl):
        lbl2idx[tbte] = i

    n_lbl = len(idx2lbl)
    n_img, n_elec, _, _ = h5.root.spkt_img.shape

    # -- layout output hdf5 file
    shape = (n_lbl, n_tr, n_img, n_elec) 
    atom = tables.UInt16Atom()
    atomi16 = tables.Int16Atom()
    filters = tables.Filters(complevel=4, complib='blosc')
    print '* shape =', shape

    h5o = tables.openFile(ofn, 'w')
    spk = h5o.createCArray(h5o.root, 'spk', atom, shape, filters=filters)
    meta = h5o.createGroup("/", 'meta', 'Metadata')
    h5o.createArray(meta, 'iid2idx_pk', iid2idx_pk)
    h5o.createArray(meta, 'idx2iid_pk', idx2iid_pk)
    h5o.createArray(meta, 'lbl2idx_pk', pk.dumps(lbl2idx))
    h5o.createArray(meta, 'idx2lbl_pk', pk.dumps(idx2lbl))
    # -- preserving structures
    if preservefmt:
        ntrials_img0 = h5.root.meta.ntrials_img.read()
        ntrials_img = np.empty((n_lbl,) + ntrials_img0.shape, dtype=np.uint16)
        orgfile_img0 = np.transpose(h5.root.meta.orgfile_img.read(), [2,0,1])

        orgfile = h5o.createCArray(meta, 'orgfile', atomi16, (n_lbl, n_tr, n_img, n_elec), filters=filters)   # origin info
        for ilbl in xrange(n_lbl):
            ntrials_img[ilbl,:,:] = ntrials_img0[:,:]
            orgfile[ilbl,:,:,:] = orgfile_img0[...]

        h5o.createArray(meta, 'srcfiles', h5.root.meta.srcfiles.read())
        h5o.createArray(meta, 'ntrials_img', ntrials_img)


    M = np.empty((n_tr, n_img, n_elec))
    for i, (tb, te) in enumerate(idx2lbl):
        print '* At: %d/%d                   \r' % (i+1, n_lbl),
        sys.stdout.flush()
        ib = tb - tmin
        ie = te - tmin 
        if ib < 0: ib = 0
        if ie > nbins: ie = nbins

        for ii in xrange(n_img):
            # spkt_bits: (units, trials, time in bit-repr)
            spkt_bits = np.unpackbits(si[ii,:,:n_tr,:], axis=2)
            # r: (units, trials) -> spike count in [tb,te]
            r = spkt_bits[:,:,ib:ie].sum(axis=2).T
            M[:,ii,:] = r

        spk[i,:,:,:] = M 

    h5o.close()
    h5.close()



# ----------
def main(argv):
    args, opts = cio.parse_opts(argv[1:]) 
    if len(args) != 5:
        print 'conv_h5tinfo2h5.py tbte <tinfo.h5> <output_repr.h5> <t_begins> <t_ends> [opts]'
        print '- t_begins: beginnings of reading window (0=stim onset), comma-separated list'
        print '- t_ends: ends of reading window, comma-separated, length must be same as t_begins'
        print
        print 'conv_h5tinfo2h5.py twmp <tinfo.h5> <output_repr.h5> <delta_t> <midpt> [opts]'
        print '- delta_t: reading window width, comma-separated list'
        print '- midpt: mid-point of window, comma-separated'
        print
        print 'Notes:'
        print 'All time values are in ms.'
        print
        print 'Options:'
        print '  --preservefmt    do not compact file'
        print
        sys.exit(0)

    mode, ifn, ofn, t1, t2 = args
    t1 = [int(t) for t in t1.split(',')]
    t2 = [int(t) for t in t2.split(',')]

    if mode == 'tbte':
        tbs = t1
        tes = t2
        nooversized=False
    elif mode == 'twmp':
        tbs = []
        tes = []
        for dt in t1:
            dt2 = dt / 2
            for mp in t2:
                tbs.append(mp - dt2)
                tes.append(mp + dt2)
        nooversized=True
    else:
        print '* unrecognized mode "%s"' % mode
        return

    preservefmt = False
    if 'preservefmt' in opts:
        print '* Preserve format'
        preservefmt = True

    assert len(tbs) == len(tes)
    convert(ifn, ofn, tbs, tes, nooversized=nooversized, preservefmt=preservefmt)

if __name__ == '__main__':
    main(sys.argv)
