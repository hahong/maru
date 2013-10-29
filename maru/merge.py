# Author: Ha Hong (hahong@mit.edu)

import warnings
import numpy as np
import os
from pymworks import data as mdata
from pymario import brreader, plxreader
from difflib import SequenceMatcher

DEF_CODE_SENT = 'wordout_var'
DEF_CODE_RECV = 'wordSent'
DEF_PROC_WAV = False
DEF_ADJ_REJECT = None
DEF_AMP_REJECT = None
DEF_TIMETRANSF = None


# TODO: handle continuous waveform


class Merge:
    # Properties -----------------------------------------------------------
    # private: should never be changed in the instances!
    _TS_ALGORITHM_VER = 0x0002
    _SPK_REPRESENTATION_VER = 0x0002
    C_MAGIC = '#merged_data_toc'
    K_TSTAMP = 'timestamp'
    K_SPIKE = 'spikes'
    K_SPIKEWAV = 'spikes_waveform'
    K_VERALG = 'version_align_alg'
    K_VERSPK = 'version_spike_repr'

    # public:
    C_SENT = 'wordout_var'
    C_RECV = 'wordSent'
    C_TSTAMP = 'merged_remote_timestamp'
    C_SPIKE = 'merged_spikes'
    C_SPIKEWAV = 'merged_spikes_wave_id%03d'

    # Methods -------------------------------------------------------------
    def __init__(self, mwk_filename=None, neu_filename=None):
        self.mwk_filename = None
        self.neu_filename = None
        self.mf = None
        self.nf = None
        if mwk_filename is not None and neu_filename is not None:
            self.set_files(mwk_filename, neu_filename)

    def set_files(self, mwk_filename=None, neu_filename=None):
        self.mwk_filename = mwk_filename
        self.neu_filename = neu_filename
        if self.mf is not None:
            self.mf.close()
            self.mf = None
        if self.nf is not None:
            self.nf.close()
            self.nf = None
        # TODO: catch errors
        if mwk_filename is not None:
            self.mf = mdata.MWKFile(mwk_filename)
            self.mf.open()
        if neu_filename is not None:
            ext = os.path.splitext(neu_filename)[1]
            if ext.lower() == '.nev':
                self.nf = brreader.BRReader(neu_filename)
            else:
                self.nf = plxreader.PLXReader(neu_filename)
            self.nf.open()

    # quick-and-dirty neural ile reader -----
    def neu_extract_timestamp(self):
        nf = self.nf
        if nf is None or not nf.valid:
            raise IOError('Neural data not ready.')

        return nf.extract_all(only_timestamp=True)[:3]

    # quick-and-dirty mwk file reader -----
    def mwk_extract(self, code_sent=None, code_recv=None):
        mf = self.mf
        if mf is None or not mf.valid:
            raise IOError('MWK file not ready.')

        if code_sent is None:
            code_sent = Merge.C_SENT
        if code_recv is None:
            code_recv = Merge.C_RECV

        sent = mf.get_events(codes=[code_sent])
        # sent timestamp words
        dat_sent = [x.value for x in sent if x.value != 0]
        # when the time stamps are sent to ITC? (in us)
        t_sent = [x.time for x in sent if x.value != 0]
        start_time = t_sent[0]

        recv = mf.get_events(codes=[code_recv],
                time_range=[start_time, mf.maximum_time])
        # when the MWorks got the loopback bit? (in us)
        t_recv = [x.time for x in recv]

        return t_sent, t_recv, dat_sent

    # ------------------------------------------------------------------------
    def _debug_info(self, t_sent, t_recv, dat_sent,
            t_nsp, dat_nsp, plot_result=False):
        # TODO: clean up!!!
        # calculating onset of each event
        I_SENT = 0
        I_LPBACK = 1
        I_NSP = 2   # Mac -> ITC, ITC -> MAC, NSP
        t_dat = np.zeros((len(t_nsp), 3))
        t_dat[:, I_SENT] = t_sent
        t_dat[:, I_LPBACK] = t_recv
        t_dat[:, I_NSP] = t_nsp

        # calculating deviation from "Sent"
        t_deldat = np.array(t_dat)
        t_deldat[:, I_LPBACK] -= t_deldat[:, I_SENT]
        t_deldat[:, I_NSP] -= t_deldat[:, I_SENT]

        # finished calculation. now prints all -------
        # check if data sent correctly
        mismatched_bad = 0
        mismatched_lessbad = 0
        matched = 0
        for x, y in zip(dat_sent, dat_nsp):
            if x != y:
                if x - y == 128:
                    mismatched_lessbad += 1   # MSB tends to be less reliable..
                else:
                    mismatched_bad += 1
            else:
                matched += 1

        print 'Mismatch (Bad):', mismatched_bad
        print 'Mismatch (Less Bad):', mismatched_lessbad
        print 'Matched:', matched
        print

        # onset time
        print 'Onset Time (Sent | delta-Received | delta-NSP):'
        print t_deldat
        print 'Onset Time Mean (delta-Received | delta-NSP):'
        print '                    ', t_deldat[:, 1:].mean(0)
        print

        if plot_result:
            import pylab as pl
            from matplotlib import rc

            # plot onset time shift
            rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
            pl.figure()
            pl.plot(t_deldat[:, I_LPBACK], 'g-')
            pl.plot(t_deldat[:, I_NSP], 'r-')
            pl.xlabel('Timestamp Number')
            pl.ylabel(r'${\mu}$s')
            pl.title('Time Shift from Mac')
            pl.show()

    # ------------------------------------------------------------------------
    def _align(self, dat_sent0, dat_nsp0, t_sent0,
            t_recv0, t_nsp0, i_nsp0):
        dat_sent = []
        dat_nsp = []
        t_sent = []
        t_recv = []
        t_nsp = []
        i_nsp = []

        diff = SequenceMatcher(None, dat_sent0, dat_nsp0)
        for i, j, n in diff.get_matching_blocks():
            dat_sent.extend(dat_sent0[i: i + n])
            t_sent.extend(t_sent0[i: i + n])
            t_recv.extend(t_recv0[i: i + n])
            dat_nsp.extend(dat_nsp0[j: j + n])
            t_nsp.extend(t_nsp0[j: j + n])
            i_nsp.extend(i_nsp0[j: j + n])

        return dat_sent, dat_nsp, t_sent, t_recv, t_nsp, i_nsp

    # ------------------------------------------------------------------------
    def _time_translation(self, sent, recv, nsp):
        delay = (recv - sent).mean()    # estimated delay between MAC -> NSP
        # do the linear fit between `nsp` and `sent`.
        # C0[0] : slope
        # C0[1] : intercept
        (C0, residues, rank, singular_values, rcond) = np.polyfit(nsp, sent,
                1, full=True)
        return self._time_translation_core(C0, delay)

    def _time_translation_core(self, C0, delay):
    # delay should be accounted. TODO: check is this fine. (truncation?)
        C0[1] += delay
        return C0, lambda C, t_nsp: C[0] * t_nsp + C[1], delay

    # -----------------------------------------------------------------------
    def merge(self, proc_wav=False, code_sent=None, code_recv=None,
              _info_msg=True, _debug=False, callback_reject=None,
              callback_cluster=None, adj_reject=None, amp_reject=None,
              _force_timetransf=None):
        mf = self.mf
        nf = self.nf
        if mf is None or not mf.valid:
            raise IOError('Invalid mwk file.')
        if nf is None or not nf.valid:
            raise IOError('Invalid neural event file.')
        if Merge.C_MAGIC in mf.reverse_codec:
            warnings.warn('Already merged mwk file.')
            return True

        # set new thresholds
        if adj_reject is not None and amp_reject is not None:
            new_thr = {}
            for ch in nf.chn_info:
                lthr = nf.chn_info[ch]['low_thr']
                hthr = nf.chn_info[ch]['high_thr']
                if lthr == 0:
                    thr0 = hthr
                else:
                    thr0 = lthr
                new_thr[ch] = thr0 * adj_reject
        else:
            adj_reject = amp_reject = None

        if proc_wav or adj_reject is not None:
            proc_wav0 = True
        else:
            proc_wav0 = False

        # -- Do the time translation
        if _force_timetransf is None:
            # (Mac->ITC, ITC->Mac loopback, sent timestamp words)
            (t_sent, t_recv, dat_sent) = self.mwk_extract(code_sent=code_sent,
                                                          code_recv=code_recv)
            # (ITC->NSP, received timestamp words by NSP, corresponding index
            # in all_data` for `t_nsp` and `dat_nsp`, all other information
            # including waveforms and electrode name, etc.)
            (t_nsp, dat_nsp, i_nsp) = self.neu_extract_timestamp()
            if len(t_sent) != len(t_recv):
                warnings.warn('Warning: length mismatch in ITC-loopbacks'
                        '(sent=%d, recv=%d). Ignoring delay' %
                        (len(t_sent), len(t_recv)))
                t_recv = t_sent

            dat_sent, dat_nsp, t_sent, t_recv, t_nsp, i_nsp = \
                    self._align(dat_sent, dat_nsp, t_sent,
                            t_recv, t_nsp, i_nsp)
            if _debug:
                self._debug_info(t_sent, t_recv, dat_sent, t_nsp, dat_nsp)
            if len(dat_sent) != len(dat_nsp):
                raise ValueError('Alignment failed.')

            t_sent = np.array(t_sent)
            t_recv = np.array(t_recv)
            t_nsp = np.array(t_nsp)

            # get parameters `C` and time translation function `f`
            C, f, delay = self._time_translation(t_sent, t_recv, t_nsp)
        else:
            C0 = _force_timetransf['C']
            delay = _force_timetransf['delay']
            C0[1] -= delay
            C, f, delay = self._time_translation_core(C0, delay)

        if _info_msg:
            print 'Linear transformation parameters:', list(C)
            print 'Estimated delay:', delay
        valid_elec_ids = nf.spike_id

        # do the Mac-NSP time translation and write down to the mwk file -----
        # 0. first, write down the table of contents data -------------------
        toc_info = {Merge.K_SPIKE: self.C_SPIKE,
                    Merge.K_VERALG: int(Merge._TS_ALGORITHM_VER),
                    Merge.K_VERSPK: int(Merge._SPK_REPRESENTATION_VER),
                    'align_info': {'params': list(C),
                                   'delay': delay},
                    'neu_filename': os.path.basename(self.neu_filename)
                   }
        if _force_timetransf is None:
            toc_info[Merge.K_TSTAMP] = self.C_TSTAMP
        if proc_wav:   # if actual waveform data processing is requested,
            toc_info[Merge.K_SPIKEWAV] = {}
            wv_codes = {}
            for eid in valid_elec_ids:
                tagname = self.C_SPIKEWAV % eid
                toc_info[Merge.K_SPIKEWAV][eid] = tagname
                wv_codes[eid] = \
                        mf.queue_code(code_metadata={'tagname': tagname})

        header_ev_blocks = [(mf.minimum_time + 1, toc_info)]
        mf.queue_code(code_metadata={'tagname': self.C_MAGIC},
                      event_blocks=header_ev_blocks)

        # 1. second, match the timestamps (i.e. ts got from digin port) -----
        if _force_timetransf is None:
            ts_ev_blocks = []
            t_nsp_mactime = f(C, t_nsp)
            for i in range(len(t_nsp)):
                # t0: original time in NSP's clock
                # timestamp_word: received timestamp
                payload = {'t0': t_nsp[i], 'timestamp_word': dat_nsp[i]}
                ts_ev_blocks.append((long(np.round(t_nsp_mactime[i])),
                    payload))
            mf.queue_code(code_metadata={'tagname': self.C_TSTAMP},
                          event_blocks=ts_ev_blocks)
            # free up unnecessary data
            del dat_sent, dat_nsp, t_sent, t_recv, t_nsp, i_nsp

        # 2. save actual spike data if there's any -------------------------
        spk_code = mf.queue_code(code_metadata={'tagname': self.C_SPIKE})
        # write down the all toc info and new codec
        mf.write_queued_codes(delayed_writing=True)

        chn_info = nf.chn_info
        t_sample = nf.t_sample
        nf.goto_first_data()

        BUF_SZ = 8192
        ev_buf = []

        def write_events():
            for code, ts, payload in ev_buf:
                mf.write_one_event(code, ts, payload)

        # collect all information of all electrodes
        while True:
            the_data = nf.read_once(proc_wav=proc_wav0)
            if the_data is None:              # reached EOF
                write_events()
                ev_buf = []
                break
            if len(ev_buf) >= BUF_SZ:         # buffer full
                write_events()                # write to the new mwk file
                ev_buf = []

            # elec_id: actual electrode id
            # spk_t: onset of the spike in NSP
            # ts: onset of the spike in MWorks time
            # pos: absolute data offset of the spike
            elec_id = the_data['id']
            spk_t = the_data['timestamp']
            ts = long(np.round(f(C, spk_t)))
            pos = the_data['file_pos']

            # bunch of rejection regimes
            if not elec_id in valid_elec_ids:
                continue
            if callback_reject is not None and callback_reject(the_data, ts):
                continue
            if amp_reject is not None:
                wav = amp_reject(the_data['waveform'], new_thr[elec_id])
                if wav is None:
                    continue
                the_data['waveform'] = wav

            cinfo = None                        # clustering information
            if callback_cluster is not None:
                cinfo = callback_cluster(the_data, ts)
                if cinfo is None:
                    continue

            spk_data = {'id': elec_id, 'foffset': pos}
            if cinfo is not None:
                spk_data.update(cinfo)
            ev_buf.append((spk_code, ts, spk_data))

            if proc_wav:
                wav = the_data['waveform']          # waveform data
                if len(wav) < 1:
                    continue

                # dfac: digitization factor in uV/bit
                dfac = chn_info[elec_id]['dig_nv'] / 1000.
                # alt: wav = np.array(wav) * dfac
                wav_code = wv_codes[elec_id]
                # t_sample_us: sampling period in us
                wav_data = {'id': elec_id,
                            't_sample_us': t_sample,
                            'raw_to_uV_factor': dfac,
                            'waveform_raw': wav    # alt: wav.tolist()
                           }
                ev_buf.append((wav_code, ts, wav_data))

        # 3. finished all tasks. clean up.
        mf.finish_writing_codes()
        return True


def merge(fn_mwk, fn_neu, proc_wav=DEF_PROC_WAV,
        code_sent=DEF_CODE_SENT, code_recv=DEF_CODE_RECV,
        adj_reject=DEF_ADJ_REJECT, amp_reject=DEF_AMP_REJECT,
        _force_timetransf=DEF_TIMETRANSF):
    """Merges a mwk file and a nev/plx file.

    Parameters
    ----------
    fn_mwk: mwk file name

    fn_neu: nev or plx file name

    proc_wav: if true, the merged mwk file will retain snippets

    code_sent: the code name for outbound (mworks -> blackrock/plexon)
        time stamps

    code_recv: the code name for returning time stamp bits

    amp_reject: amplitude-based rejection function

    adj_reject: multiplier-based rejection function

    _force_timetransf: set the time transformation. must be:
        [slope,intercept,delay]
    """

    # -- execute
    m = Merge(fn_mwk, fn_neu)
    status = m.merge(proc_wav=proc_wav,
            code_sent=code_sent,
            code_recv=code_recv,
            adj_reject=adj_reject,
            amp_reject=amp_reject,
            _force_timetransf=_force_timetransf)
    if status:
        print 'merge: merged successfully.'
    return status
