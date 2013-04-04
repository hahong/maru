#!/usr/bin/env python

import numpy as np
import os
import warnings
import sys
sys.path.append('lib')
from mergeutil import *
from mworks.data import *
from common_fn import parse_opts_adapter, T_START, T_STOP, get_stim_info, seq_search, sort_uniq, prep_files, set_new_threshold
from collections import defaultdict

BAD_ISI = 3000   # spiking within 3ms is bad
CLU_NMAX = 5     # up to five clusters
CLU_SEP = ','

# Setup the hash-table ---------------------------------------------------
def hash_tbl(lst):
    tbl = {}
    for i, k in enumerate(lst):
        tbl[k] = i
    return tbl

def find(needle, haystack):
    if needle not in haystack: return None
    return haystack[needle]


# Setup the cluster id across multiple sessions -------------------------
def get_standard_cid(clu, tclu, nmax=np.inf, _debug=False):
    def sorted_cid(clu):
        # counts # of each cluster id
        cluc = defaultdict(int)         
        for cid in clu: cluc[cid] += 1
        # sort
        return sorted([(cluc[cid], cid) for cid in cluc], reverse=True), len(cluc)

    # -- load other info
    cluu = list(set(list(clu)))     # unique cluster id
    nclu = len(cluu)
    # -- if there're too many clusters, coalesce small clusters into one
    ucid = -1   # cid of "unsorted" group (default = -1, means no unsorted group)
    if nclu > nmax:
        clus, _ = sorted_cid(clu)
        # coalesce
        ccid = hash_tbl([cid for _, cid in clus[nmax - 1:]])  # clusters to coalesce
        # ucid: now we have the unsorted group. assign the cid of "unsorted" group
        ucid = nclu + 1
        for i in xrange(len(clu)):
            if clu[i] in ccid: clu[i] = ucid
        for i in xrange(len(tclu)):
            if tclu[i] in ccid: tclu[i] = ucid
    # -- reassign cluster id
    clus, nclu = sorted_cid(clu)
    # old cid -> new cid mapping
    ncid = {}   
    for i, (_, ocid) in enumerate(clus): ncid[ocid] = i 
    for i in xrange(len(tclu)):
        tclu[i] = ncid[tclu[i]]
    if ucid >= 0: ucid = ncid[ucid]  # if there's "unsorted" group, get the new cid of it.

    #if _debug:
    #    import cPickle as pk
    #    pk.dump(ncid, open('ncid.pk', 'wb'))
    #    print 'Dump done.'
    return nclu, ucid


def load_superset(pf0, pfs, ch, multipath=True):
    clus = []
    dir0 = os.path.dirname(pf0)
    for pf in pfs:
        if multipath:
            fn_clu = pf + '.clu.' + str(ch)
        else:
            bname = os.path.basename(pf)
            fn_clu = dir0 + os.sep + bname + '.clu.' + str(ch)
        clus.append(np.loadtxt(fn_clu, skiprows=1).astype('int'))
    return np.concatenate(clus)

# Rejection of spikes based on .inf.* files -------------------------------
# Returns True if rejection is needed.
def reject_inf(wav_info, ts):
    ch = wav_info['id']
    pos = wav_info['file_pos']
    if ch not in reject_inf.valid:
        try:
            inf = np.loadtxt(reject_inf.prefix + '.inf.' + str(ch), skiprows=1)
            ninf0 = len(inf)
            inf, = sort_uniq(inf[:,0].astype('int'), inf)
            ninf1 = len(inf)
            if ninf1 != ninf0: print 'Ch:', ch, '\tchange in size: %d -> %d' % (ninf0, ninf1)
            reject_inf.valid[ch] = hash_tbl([int(l) for l in inf[:,0]])
        except IOError, e:
            print '*** Bad channel?:', e
            reject_inf.valid[ch] = hash_tbl([])

    # do search
    i = find(pos, reject_inf.valid[ch])
    if i == None: return True      # not found: reject
    # it's in the record.
    return False                   # no rejection

reject_inf.prefix = ''       # (static) .inf prefix
reject_inf.valid = {}        # (static) valid positions


# Clustering of spikes based on .inf.* and .clu.* files -------------------
# Returns None if rejection is needed.
def cluster(wav_info, ts):
    ch = wav_info['id']
    pos = wav_info['file_pos']
    if ch not in cluster.valid:
        try:
            # -- load and sort in terms of inf[:,0]
            inf = np.loadtxt(cluster.prefix_inf + '.inf.' + str(ch), skiprows=1)
            clu = np.loadtxt(cluster.prefix_clu + '.clu.' + str(ch), skiprows=1).astype('int') 
            assert len(inf) == len(clu)
            ninf0 = len(inf)
            inf, clu = sort_uniq(inf[:,0].astype('int'), inf, clu)
            ninf1 = len(inf)
            if ninf1 != ninf0: print 'Ch:', ch, '\tchange in size: %d -> %d' % (ninf0, ninf1)
            cluster.valid[ch] = hash_tbl([int(l) for l in inf[:,0]])
            # get the "nicer" cluster id (clu_ref: reference cluster id's)
            if cluster.prefix_clu_all == None: clu_ref = clu
            else: clu_ref = load_superset(cluster.prefix_clu, cluster.prefix_clu_all, ch)
            nclu, ucid = get_standard_cid(clu_ref, clu, nmax=cluster.nmax)
            # -- check bad ISIs
            nbad = []
            nspk = []
            for cid in range(nclu):
                ii = np.nonzero(cid == clu)[0]
                isi = np.diff(np.sort(inf[ii,1]))
                nbad.append(np.sum(isi < BAD_ISI))
                nspk.append(len(ii))
            # -- done
            cluster.ucid[ch] = ucid
            cluster.nbad[ch] = nbad
            cluster.nspk[ch] = nspk
            cluster.nclu[ch] = nclu
            cluster.clu[ch] = list(clu)
            print 'Ch:', ch, '\t#Spk:', nspk, ' #Bad:', nbad, '  Unsorted ID:', ucid
        except IOError, e:
            print '*** Bad channel?:', e
            cluster.valid[ch] = hash_tbl([])

    # -- give the cluster data
    # do search 
    i = find(pos, cluster.valid[ch])
    if i == None: return None      # not found: reject

    # it's in the cluster record.
    cid = cluster.clu[ch][i]
    rval = {'cluster_id': int(cid), 
            'unsorted_cid': int(cluster.ucid[ch]), 
            'nclusters': int(cluster.nclu[ch]), 
            'nbad_isi': int(cluster.nbad[ch][cid]),
            'nspk': int(cluster.nspk[ch][cid])}
    return rval

cluster.prefix_inf = ''       # (static) prefix
cluster.prefix_clu = ''       # (static) prefix
cluster.prefix_clu_all = None # (static) prefix of all .clu files in this superset
cluster.valid = {}            # (static) valid positions
cluster.clu = {}              # (static) cluster id
cluster.nclu = {}             # (static) number of clusters
cluster.nbad = {}             # (static) number of bad ISIs
cluster.nspk = {}             # (static) number of spikes in this cluster
cluster.ucid = {}             # (static) cluster id of "unsorted" group (-1 if there's no such group)
cluster.nmax = CLU_NMAX       # (static) maximum units/ch


def set_new_threshold_rng(wav, thr):
    return set_new_threshold(wav, thr, rng=(11, 13), i_chg=32)
    # return set_new_threshold(wav, thr)

# Arbiter ----------------------------------------------------------------
def main():
    warnings.simplefilter('once')

    if len(sys.argv) < 3:
        print 'mergy.py [options] <mwk> <nev/plx>'
        print 'Aligns and merges mwk and nev/plx files.'
        print
        print 'Options:'
        print '   --nowav                       - do not include waveform data (recommended)'
        print '   --filter_inf=<.inf prefix>    - reject any spikes not in the inf files'
        print '   --cluster=<.inf/.clu prefix>  - apply clustering information'
        print '   --cluster_all=<.clu prefix 1,.clu prefix 2,...>'
        print '                                 - use other clustering info (in this superset)'
        return

    args, opts = parse_opts_adapter(sys.argv[1:], 2)
    fn_mwk, fn_nev = args[:2]

    # -- parse options
    if 'nowav' in opts:
        print 'merge: no waveform processing'
        proc_wav = False
    else:
        print 'merge: including waveform data'
        proc_wav = True

    callback_reject = None
    if 'filter_inf' in opts:
        inf_prefix = opts['filter_inf']
        print 'merge: spike rejection based on', inf_prefix
        reject_inf.prefix = inf_prefix
        callback_reject = reject_inf

    callback_cluster = None
    if 'cluster' in opts:
        prefixes = opts['cluster'].split(CLU_SEP)
        if len(prefixes) != 2:
            prefix = prefixes[0]
            print 'merge: clustering based on', prefix
            cluster.prefix_inf = prefix
            cluster.prefix_clu = prefix
        else:
            print 'merge: clustering based on', prefixes
            cluster.prefix_inf = prefixes[0]
            cluster.prefix_clu = prefixes[1]
        
        if 'cluster_all' in opts:
            clu_all = prep_files(opts['cluster_all'], sep=CLU_SEP, extchk=False)
            print 'merge: using other info', clu_all
            cluster.prefix_clu_all = clu_all
        callback_cluster = cluster

    code_sent = 'wordout_var'
    if 'code_sent' in opts:
        code_sent = opts['code_sent']
        print 'merge: code_sent =', code_sent

    code_recv = 'wordSent'
    if 'code_recv' in opts:
        code_recv = opts['code_recv']
        print 'merge: code_recv =', code_recv

    adj_reject = None
    amp_reject = None
    if 'adj_reject' in opts:
        adj_reject = float(opts['adj_reject'])
        amp_reject = set_new_threshold_rng
        print 'merge: amplitude rejection multiplier =', adj_reject

    _force_timetransf = None
    if 'force_timetransf' in opts:
        a, b, delay = [float(e) for e in opts['force_timetransf'].split(',')]
        _force_timetransf = {}
        _force_timetransf['C'] = [a, b]
        _force_timetransf['delay'] = delay
        print 'merge: time transformation =', _force_timetransf

    # -- execute
    m = Merge(fn_mwk, fn_nev)
    if m.merge(proc_wav=proc_wav, code_sent=code_sent, code_recv=code_recv, \
            adj_reject=adj_reject, amp_reject=amp_reject, \
            callback_reject=callback_reject, callback_cluster=callback_cluster, \
            _force_timetransf=_force_timetransf):
        print 'merge: merged successfully.'

if __name__ == '__main__':
    main()
