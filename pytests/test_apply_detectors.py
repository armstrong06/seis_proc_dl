import pytest
from apply_to_continuous import apply_detectors
from obspy.core.utcdatetime import UTCDateTime as UTC
import numpy as np
import obspy

def test_n_windows():
    npts = 1000
    window_length = 200
    sliding_interval = 100
    # A time-series of length 1000, with a window length of 200 and a sliding interval of 100 
    # should produce 9 windows
    assert apply_detectors.get_n_windows(npts, window_length, sliding_interval) == 9

def test_load_data_different_sampling_rate_issue():
    # I know this has gaps in it
    file = '/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/pytests/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
    st, _ = apply_detectors.load_channel_data(file, min_signal_percent=0)
    assert len(st.get_gaps()) == 0

def test_load_data_not_enough_signal():
    # I know this has gaps in it
    file = '/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/pytests/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
    st, gaps = apply_detectors.load_channel_data(file, min_signal_percent=5)
    assert st == None
    assert len(gaps) == 1

def test_load_data_filling_ends():
    # I know this has gaps in it
    file = '/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/pytests/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
    st, _ = apply_detectors.load_channel_data(file, min_signal_percent=0)
    # Check the startime
    assert (st[0].stats.starttime - UTC("2002-01-01")) < 1
    # Check the endtime
    assert (UTC("2002-01-02") - st[0].stats.endtime) < 1
    # Check number of points is what is expected (+- 1 sample)
    assert abs(st[0].stats.npts - 8640000) <= 1

def test_load_data_end_gaps_added():
    file = '/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/pytests/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
    _, gaps = apply_detectors.load_channel_data(file, min_signal_percent=0)  
    # Know there is 1 gap between signal and signals do not go to the end of the hour
    assert len(gaps) == 3 
    assert gaps[0][4] == UTC('2002-1-1')
    assert gaps[2][5] == UTC('2002-1-2')

def test_load_data_small_gaps():
    file = "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/pytests/example.mseed"
    
    # Make test example
    ex = np.zeros(24*60*60*100)
    tr = obspy.Trace()
    tr.data = ex
    tr.stats.sampling_rate=100
    tr.stats.delta = 0.01
    tr.stats.starttime = UTC('2002-01-01')
    st = obspy.Stream()
    st += tr.copy()
    # trim time
    t = UTC('2002-01-01-12')
    st.trim(endtime=t)
    # Trim 5 samples later
    tr.trim(starttime=t+0.05)
    st += tr
    st.write(file, format="MSEED")
    
    st, gaps = apply_detectors.load_channel_data(file, min_signal_percent=0)

    # There should be no gaps returned and the trace should be continuous
    assert len(gaps) == 0
    assert st[0].stats.npts == 8640000
    assert len(st.get_gaps()) == 0 

if __name__ == '__main__':
   test_load_data_small_gaps()