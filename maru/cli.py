#!/usr/bin/env python
import warnings
from .utils import parse_opts_adapter, set_new_threshold_rng

MERGE_USAGE = \
"""
mergy.py [options] <mwk> <nev/plx>
Aligns and merges mwk and nev/plx files.

Options:
   --procwav                include waveform data (not recommended)
   --code_sent=<code>       specify the code name for sending out timestamps
   --code_recv=<code>       specify the code name for returned timestamp bits
   --force_timetransf=<slope,intercept,delay>
                            use the supplied parameters for merging.
"""


def main_merge(args):
    from .merge import (merge, DEF_CODE_SENT, DEF_CODE_RECV,
        DEF_PROC_WAV, DEF_ADJ_REJECT, DEF_AMP_REJECT, DEF_TIMETRANSF)

    if len(args) < 2:
        print MERGE_USAGE
        return 1

    warnings.simplefilter('once')
    args, opts = parse_opts_adapter(args, 2)
    fn_mwk, fn_neu = args[:2]

    proc_wav = DEF_PROC_WAV
    # -- parse options
    if 'nowav' in opts:
        # for backward compatibility
        print 'merge: no waveform processing'
        proc_wav = False
    if 'procwav' in opts:
        print 'merge: including waveform data'
        proc_wav = True

    code_sent = DEF_CODE_SENT
    if 'code_sent' in opts:
        code_sent = opts['code_sent']
        print 'merge: code_sent =', code_sent

    code_recv = DEF_CODE_RECV
    if 'code_recv' in opts:
        code_recv = opts['code_recv']
        print 'merge: code_recv =', code_recv

    adj_reject = DEF_ADJ_REJECT
    amp_reject = DEF_AMP_REJECT
    if 'adj_reject' in opts:
        adj_reject = float(opts['adj_reject'])
        amp_reject = set_new_threshold_rng
        print 'merge: amplitude rejection multiplier =', adj_reject

    _force_timetransf = DEF_TIMETRANSF
    if 'force_timetransf' in opts:
        a, b, delay = [float(e) for e in opts['force_timetransf'].split(',')]
        _force_timetransf = {}
        _force_timetransf['C'] = [a, b]
        _force_timetransf['delay'] = delay
        print 'merge: time transformation =', _force_timetransf

    if merge(fn_mwk, fn_neu, proc_wav=proc_wav,
        code_sent=code_sent, code_recv=code_recv,
        adj_reject=adj_reject, amp_reject=amp_reject,
        _force_timetransf=_force_timetransf):
        return 0
    return 1
