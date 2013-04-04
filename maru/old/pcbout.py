#!/usr/bin/env python

import sys
import numpy as np
from collections import namedtuple
import csv

ARR_NAME = ['A', 'M', 'P']
RI_LOWER = [65, 112]        # inclusive
RI_HIGHER = [185, 232]      # inclusive
RO_LOWER = [1, 64]          # actual output pin number inclusive
RO_HIGHER = [121, 184]      # idem
N_MAXOUT = 128              # maximum electrodes to take (must be multiples of 2)
EDN_TEMPL_OPEN = \
"""(edif scard  (edifVersion 2 0 0)
    (edifLevel 0)
    (keywordMap (keywordLevel 0))
    (library SCARD
        (edifLevel 0)
        (cell SCARD (cellType generic)
            (view schematic (viewType netlist)
                (contents"""

EDN_TEMPL_CLOSE = \
"""
                )
            )
        )
    )
) 
"""
EDN_TEMPL_NET = \
"""
                    (net Net${netid}
                        (joined
                            (portRef ${inp} (instanceRef ${inbase}))
                            (portRef ${outp} (instanceRef ${outbase}))
                        )
                    )"""

# Electrode info.
# pin: 1-based pin number in the DDR240 socket
# el: 1-based electrode id in the patient cable (1 <= el <= 96)
ElInfo = namedtuple('ElInfo', 'pin el')

# ----------------------------------------------------------------------------
def trans_default(el, arr_name=ARR_NAME):
    """
    trans_default(el, arr_name=ARR_NAME): get 0-based/1-based electrode number for array and array name
    el: should be 0-based!
    """
    if el < 0 or el >= 288:
        raise ValueError, 'wrong electrode number in the input file!'
    arr = arr_name[el/96]
    el %= 96
    # array number, 0-based electrode number in the bank, 1-based electrode number in the bank
    return arr, el, el + 1

def trans_pin(ch):
    """
    trans_pin(ch): trnaslate 0-based electrode id into pin number in the DDR240 connector
    """
    # ch should be 0-based.
    if ch <= 47: return ch + 65
    elif ch >= 48 and ch <= 95: return 280 - ch
    raise ValueError, 'wrong input electrode number!'

def trans_outpin(pin):
    """
    trans_outpin(pin): translate pin number in the DDR240 connector into 1-based
    output electrode id.
    """
    if 33 <= pin and pin <= 64: return 65 - pin             # bank A
    elif 1 <= pin and pin <= 32: return 33 - pin + 32       # bank B
    elif 121 <= pin and pin <= 152: return pin - 120 + 64   # bank C
    elif 153 <= pin and pin <= 184: return pin - 152 + 96   # bank D
    else: raise ValueError, 'wrong input pin number!'

def load_selected(inp, arr_name=ARR_NAME, trans=trans_default, transp=trans_pin):
    selected = {}
    for arr in arr_name: selected[arr] = []

    for line in open(inp).readlines():
        for el in line.strip().split():
            el = int(el)
            arr, el0, el1 = trans(el)
            eli = ElInfo(pin=transp(el0), el=el1)
            selected[arr].append(eli)
    return selected

# ----------------------------------------------------------------------------
def prepare(sel, rh=RI_HIGHER, rl=RI_LOWER, nmax=N_MAXOUT):
    n_higher = 0; n_lower = 0; na = len(sel.keys()); nhalf = nmax/2

    # estimate inbalance
    for arr in sel:
        a = np.array([eli.pin for eli in sel[arr]])
        nh = np.sum((a >= rh[0]) & (a <= rh[1])) 
        n_higher += nh
        n_lower += len(arr) - nh
    n_excess = n_higher - nhalf

    # balance between higher pins and lower pins
    data = {}; nleft = na + 1
    print 'pcbout: n_excess =', n_excess
    for arr in sel:
        nleft -= 1
        a = np.array([eli.pin for eli in sel[arr]])
        ih = np.nonzero((a >= rh[0]) & (a <= rh[1]))[0]; nh = len(ih)
        il = np.nonzero((a >= rl[0]) & (a <= rl[1]))[0]; nl = len(il)
        elh = [sel[arr][i] for i in ih]
        ell = [sel[arr][i] for i in il]
        
        if n_excess > 0:
            # number of items to move to lower
            nm = min(int(np.round(n_excess/float(nleft))), nhalf - nl)
            sh = sorted(elh)   # sorted w.r.t pin#
            high = sh[:-nm]
            low = ell 
            mvhl = sh[-nm:]    # things to move from high to low
            mvlh = []
            n_excess -= nm
        elif n_excess < 0:
            # number of items to move to higher
            nm = min(int(np.round(-n_excess/float(nleft))), nhalf - nh)
            sl = sorted(ell)   # sorted w.r.t pin#
            low = sl[:-nm]
            high = elh 
            mvlh = sl[-nm:]    # things to move from low to high
            mvhl = []
            n_excess += nm
        else:
            low = ell
            high = elh
            mvlh = mvhl = []
        data[arr] = {}
        data[arr]['high'] = high
        data[arr]['low'] = low
        data[arr]['mvlh'] = mvlh
        data[arr]['mvhl'] = mvhl

    if n_excess != 0:
        raise ValueError, 'cannot realize wiring network!'
    
    return data


# ----------------------------------------------------------------------------
def makenet(data, rl=RO_LOWER, rh=RO_HIGHER):
    # left lower pins left, left higher pins left
    ll = range(rl[0], rl[1]+1); lh = range(rh[0], rh[1]+1)
    # nets: realized network. net = {'A': [(from, to), ], ...}
    nets = {}

    for arr in data:
        d = data[arr]
        net = []

        # do lower output pins first ------
        a = sorted(d['mvhl'], reverse=True) + sorted(d['low'], reverse=True)  # sorted w.r.t pin#
        for ch in a:
            outpin = ll.pop(0)
            net.append((ch, outpin))

        # do higher pins ------------------
        a = sorted(d['mvlh'], reverse=True) + sorted(d['high'], reverse=True) # sorted w.r.t pin#
        for ch in a:
            outpin = lh.pop(0)
            net.append((ch, outpin))

        nets[arr] = net
    return nets


# ----------------------------------------------------------------------------
def writenet_edif(net, out):
    # quick-and-dirty EDIF writer
    f = open(out, 'w')
    print >>f, EDN_TEMPL_OPEN

    netid = 0
    for eli, outp in net:
        inp = eli.pin
        if inp <= 112: inbase = 'U1'
        else: inp -= 120; inbase = 'U2'
        if outp <= 112: outbase = 'U1'
        else: outp -= 120; outbase = 'U2'
        netid += 1

        outs = EDN_TEMPL_NET
        outs = outs.replace('${inp}', str(inp))
        outs = outs.replace('${inbase}', str(inbase))
        outs = outs.replace('${outp}', str(outp))
        outs = outs.replace('${outbase}', str(outbase))
        outs = outs.replace('${netid}', str(netid))
        print >>f, outs

    print >>f, EDN_TEMPL_CLOSE
    f.close()


def get_mapping_sorted(nets):
    mapping = []
    for arr in nets:
        net = nets[arr]
        for eli, outp in net:
            mapping.append((trans_outpin(outp), arr, eli.el))
    return sorted(mapping)


def writenet(nets, out):
    # write EDIF files
    for arr in nets:
        writenet_edif(nets[arr], out + '_' + arr + '.edn')

    # write mapping csv file
    mapping = get_mapping_sorted(nets)
    csvw = csv.writer(open(out + '_map.csv', 'wb'))
    csvw.writerow(['output_elec_id', 'array', 'original_elec_id_in_the_array'])
    csvw.writerows(mapping)

    # sanity check
    all_el = set(range(1, N_MAXOUT + 1))
    all_out = set([e[0] for e in mapping])
    if len(all_el ^ all_out) != 0:
        print '* error: possible duplication or omission'
        print '  - Out - All:', all_out - all_el
        print '  - All - Out:', all_el - all_out

# ----------------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print 'pcbout.py <0-base selected 128.txt> <output prefix> [opts] ...'
        print 'pcbout.py: builds netlist files for switching card design.'
        return

    inp = sys.argv[1]
    out = sys.argv[2]

    # load the input file: this gives 0-based selected electrodes
    # selected = {`electrode name`: [el 1, el 2, ...], ...}
    selected = load_selected(inp)

    # preparation: balance the number of lower (pins 1-64) vs higher pins (pins 121-184)
    # selected = {`electrode name`: {'high': [el 1, ...], 'low': [el ...]}, ...}
    netsraw = prepare(selected)

    # determine links
    nets = makenet(netsraw)
    # write all
    writenet(nets, out)

    print 'pcbout: done.'
    

if __name__ == '__main__':
    main()
