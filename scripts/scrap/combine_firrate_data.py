#!/usr/bin/env python

import sys
import cPickle as pk

TARGET = 'on'
BLANKS = 'blanks'

# ----------------------------------------------------------------------------
def make_db(fis, _verbose=True):
    db = {'dat':{}, 'iid2num':{}, 'num2iid':{}}
    dat = db['dat']            # dat[ch] = {iid1:[#spk1, #spk2, ...], ...}
    iid2num = db['iid2num']    # iid2num[iid] = num
    num2iid = db['num2iid']    # num2iid[num] = iid

    for i, fi in enumerate(fis):
        print 'At (%d/%d):' % (i+1, len(fis)), fi
        src = pk.load(open(fi))

        for ch in src:
            if ch not in dat:
                dat[ch] = {}  # not using defaultdict to save space

            for iid in src[ch]:
                if iid not in iid2num:
                    num = len(iid2num)
                    iid2num[iid] = num
                    num2iid[num] = iid
                else:
                    num = iid2num[iid]
                
                # source data
                if iid != BLANKS: frates = list(src[ch][iid][TARGET])
                else: frates = list(src[ch][iid])

                if num not in dat[ch]: dat[ch][num] = []
                # append to the end
                dat[ch][num].extend(frates)

    return db

# ----------------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print 'combine_firrate_data.py <output file name> <input file 1> [input file 2] [...]'
        return

    fo = sys.argv[1]
    fis = sys.argv[2:]

    db = make_db(fis)
    pk.dump(db, open(fo, 'wb'))


if __name__ == '__main__':
    main()
