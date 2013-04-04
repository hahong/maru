#!/usr/bin/env python

import sys
from mergeutil import Merge, BRReader

# ------------
def chk(fn, until=15, slack=0.3):
    sout = fn + ': '

    br = BRReader(fn)
    assert br.open()

    ts_reset = 0
    ts_reset_ts = []
    t = -1

    tstamps_set = []
    tstamps = []

    # ===
    while t < until:
        dat = br.read_once()
        if dat == None: break
        
        # chk whether there's a time reset
        t_new = dat['timestamp'] / 1000000.   # us -> s
        if t_new < t: 
            ts_reset += 1
            tstamps_set.append(tstamps)
            tstamps = []
            ts_reset_ts.append(t)

        t = t_new
        if dat['id'] == 0:
            tstamps.append(t)
    tstamps_set.append(tstamps)

    # === All done
    if ts_reset == 0: 
        sout += 'All OK.'
    else:
        ts_reset_t = max(ts_reset_ts)
        tstamps_after = tstamps_set[-1]
        tstamps_before = []
        for tss in tstamps_set[:-1]: tstamps_before.extend(tss)
        """
        print tstamps_set
        print tstamps_before
        print tstamps_after
        """

        if len(tstamps_before) == 0:
            sout += 'Resynced (%d): ' % ts_reset

            if len(tstamps_after) > 0:
                if tstamps_after[0] - slack < ts_reset_t:
                    sout += 'potential interference: ' +  str(tstamps_after[0] - ts_reset_t)
                else:
                    sout += 'OK'
            else:
                sout += 'OK (no after-resync ts?)'
        else:
            sout += 'bad (before-resync ts!) ' + str(tstamps_set)

    """
    print 'Time stamp reset:', (ts_reset != 0, ts_reset, ts_reset_t)
    print 'Before:', tstamps_before
    print 'After:', tstamps_after
    """
    print ts_reset_t, tstamps_after, tstamps_set
    print sout

# ------------
def main():
    if len(sys.argv) == 1:
        print 'chk_tstamp_corruption.py <nev file name>'
        return
    chk(sys.argv[1])

if __name__ == '__main__':
    main()
