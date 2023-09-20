import pytest
import apply_detectors

def test_n_windows():
    npts = 1000
    window_length = 200
    sliding_interval = 100
    # A time-series of length 1000, with a window length of 200 and a sliding interval of 100 
    # should produce 9 windows
    assert apply_detectors.get_n_windows(npts, window_length, sliding_interval) == 9

def test_load_data_different_sampling_rate_issue():
    # I know this has gaps in it
    file = '/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/apply_to_continuous/pytests/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
    st = apply_detectors.load_data(file, min_signal_percent=0)
    assert len(st.get_gaps()) == 0

def test_load_data_not_enough_signal():
    # I know this has gaps in it
    file = '/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/apply_to_continuous/pytests/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
    st = apply_detectors.load_data(file, min_signal_percent=5)
    assert len(st.get_gaps()) == 0

if __name__ == '__main__':
    test_load_data_different_sampling_rate_issue()