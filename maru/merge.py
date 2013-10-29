#!/usr/bin/env python
import warnings
from .common import parse_opts_adapter, set_new_threshold_rng

USAGE = \
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


def main(args):
    warnings.simplefilter('once')

    if len(args) < 2:
        print USAGE
        return

    args, opts = parse_opts_adapter(args, 2)
    fn_mwk, fn_nev = args[:2]

    proc_wav = False
    # -- parse options
    if 'nowav' in opts:
        # for backward compatibility
        print 'merge: no waveform processing'
        proc_wav = False
    if 'procwav' in opts:
        print 'merge: including waveform data'
        proc_wav = True

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
    if m.merge(proc_wav=proc_wav,
            code_sent=code_sent,
            code_recv=code_recv,
            adj_reject=adj_reject,
            amp_reject=amp_reject,
            _force_timetransf=_force_timetransf):
        print 'merge: merged successfully.'

# --
# if __name__ == '__main__':
#     import sys
#     main(sys.argv[1:])
