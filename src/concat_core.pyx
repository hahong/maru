import numpy as np
cimport numpy as np
cimport cython

ctypedef np.int16_t i16_t
ctypedef np.int32_t i32_t
ctypedef np.uint16_t u16_t
ctypedef np.float32_t f32_t

@cython.boundscheck(False)
def concat_core_pyx_u16u16(np.ndarray[u16_t, ndim=4] spk, np.ndarray[u16_t, ndim=4] spk0, \
		np.ndarray[u16_t, ndim=3] ntrimg, np.ndarray[u16_t, ndim=3] ntrials0, \
		np.ndarray[i16_t, ndim=4] orgfile, np.ndarray[i16_t, ndim=4] orgfile_t, \
		np.ndarray[i32_t, ndim=1] i_lbls, np.ndarray[i32_t, ndim=1] i_lbl0s, \
		np.ndarray[i32_t, ndim=1] i_iids, np.ndarray[i32_t, ndim=1] i_iid0s, \
		np.ndarray[i32_t, ndim=1] i_chs, np.ndarray[i32_t, ndim=1] i_ch0s, \
		int n_tr):
    cdef Py_ssize_t n_lbls, n_iids, n_chs
    cdef Py_ssize_t i_lbl, i_iid, i_ch
    cdef Py_ssize_t i_lbl0, i_iid0, i_ch0
    cdef Py_ssize_t xl, xi, xc
    cdef Py_ssize_t i_trb, i_tre, n, t

    n_lbls = len(i_lbls)
    n_iids = len(i_iids)
    n_chs = len(i_chs)

    for xl in xrange(n_lbls):
        i_lbl = i_lbls[xl]
        i_lbl0 = i_lbl0s[xl]
        if i_lbl0 < 0: continue

        for xi in xrange(n_iids):
            i_iid = i_iids[xi]
            i_iid0 = i_iid0s[xi]
            if i_iid0 < 0: continue

            for xc in xrange(n_chs):
                i_ch = i_chs[xc]
                i_ch0 = i_ch0s[xc]
        
                # -- actual conversion
                i_trb = ntrimg[i_lbl,i_iid,i_ch]   # begin of trial index
                # i_tre = min(i_trb + ntrials0[i_lbl0,i_iid0,i_ch0], n_tr)
                t = i_trb + ntrials0[i_lbl0,i_iid0,i_ch0]
                if t < n_tr: i_tre = t
                else: i_tre = n_tr
                n = i_tre - i_trb

                spk[i_lbl,i_trb:i_tre,i_iid,i_ch] = spk0[i_lbl0,:n,i_iid0,i_ch0]
                orgfile[i_lbl,i_trb:i_tre,i_iid,i_ch] = orgfile_t[i_lbl0,:n,i_iid0,i_ch0]
                ntrimg[i_lbl,i_iid,i_ch] += n
		

# dead copy of above
@cython.boundscheck(False)
def concat_core_pyx_f32f32(np.ndarray[f32_t, ndim=4] spk, np.ndarray[f32_t, ndim=4] spk0, \
		np.ndarray[u16_t, ndim=3] ntrimg, np.ndarray[u16_t, ndim=3] ntrials0, \
		np.ndarray[i16_t, ndim=4] orgfile, np.ndarray[i16_t, ndim=4] orgfile_t, \
		np.ndarray[i32_t, ndim=1] i_lbls, np.ndarray[i32_t, ndim=1] i_lbl0s, \
		np.ndarray[i32_t, ndim=1] i_iids, np.ndarray[i32_t, ndim=1] i_iid0s, \
		np.ndarray[i32_t, ndim=1] i_chs, np.ndarray[i32_t, ndim=1] i_ch0s, \
		int n_tr):
    cdef Py_ssize_t n_lbls, n_iids, n_chs
    cdef Py_ssize_t i_lbl, i_iid, i_ch
    cdef Py_ssize_t i_lbl0, i_iid0, i_ch0
    cdef Py_ssize_t xl, xi, xc
    cdef Py_ssize_t i_trb, i_tre, n, t

    n_lbls = len(i_lbls)
    n_iids = len(i_iids)
    n_chs = len(i_chs)

    for xl in xrange(n_lbls):
        i_lbl = i_lbls[xl]
        i_lbl0 = i_lbl0s[xl]
        if i_lbl0 < 0: continue

        for xi in xrange(n_iids):
            i_iid = i_iids[xi]
            i_iid0 = i_iid0s[xi]
            if i_iid0 < 0: continue

            for xc in xrange(n_chs):
                i_ch = i_chs[xc]
                i_ch0 = i_ch0s[xc]
        
                # -- actual conversion
                i_trb = ntrimg[i_lbl,i_iid,i_ch]   # begin of trial index
                # i_tre = min(i_trb + ntrials0[i_lbl0,i_iid0,i_ch0], n_tr)
                t = i_trb + ntrials0[i_lbl0,i_iid0,i_ch0]
                if t < n_tr: i_tre = t
                else: i_tre = n_tr
                n = i_tre - i_trb

                spk[i_lbl,i_trb:i_tre,i_iid,i_ch] = spk0[i_lbl0,:n,i_iid0,i_ch0]
                orgfile[i_lbl,i_trb:i_tre,i_iid,i_ch] = orgfile_t[i_lbl0,:n,i_iid0,i_ch0]
                ntrimg[i_lbl,i_iid,i_ch] += n

