import pytest
from apply_to_continuous import apply_detectors
from obspy.core.utcdatetime import UTCDateTime as UTC
import numpy as np
import obspy
from obspy.core.util.attribdict import AttribDict

examples_dir = '/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/pytests/example_files'

class TestDataLoader():

    def test_load_data_different_sampling_rate_issue(self):
        dl = apply_detectors.DataLoader()
        # I know this has gaps in it
        file = f'{examples_dir}/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        st, _ = dl.load_channel_data(file, min_signal_percent=0)
        assert len(st.get_gaps()) == 0

    def test_load_data_not_enough_signal(self):
        dl = apply_detectors.DataLoader()
        # I know this has gaps in it
        file = f'{examples_dir}/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        st, gaps = dl.load_channel_data(file, min_signal_percent=5)
        assert st == None
        assert len(gaps) == 1
        assert gaps[0][-1] == 8640000
        assert gaps[0][-2] == 86400

    def test_load_data_filling_ends(self):
        dl = apply_detectors.DataLoader()
        # I know this has gaps in it
        file = f'{examples_dir}/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        st, _ = dl.load_channel_data(file, min_signal_percent=0)
        # Check the startime
        assert (st[0].stats.starttime - UTC("2002-01-01")) < 1
        # Check the endtime
        assert (UTC("2002-01-02") - st[0].stats.endtime) < 1
        # Check number of points is what is expected (+- 1 sample)
        assert st[0].stats.npts == 8640000

    def test_load_data_end_gaps_added(self):
        dl = apply_detectors.DataLoader()
        file = f'{examples_dir}/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        _, gaps = dl.load_channel_data(file, min_signal_percent=0)  
        # Know there is 1 gap between signal and signals do not go to the end of the hour
        assert len(gaps) == 3 
        assert gaps[0][4] == UTC('2002-1-1')
        assert gaps[2][5] == UTC('2002-1-2')

    def test_load_data_small_gaps(self):
        file = f'{examples_dir}/example.mseed'
        
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

        dl = apply_detectors.DataLoader()
        st, gaps = dl.load_channel_data(file, min_signal_percent=0)

        # There should be no gaps returned and the trace should be continuous
        assert len(gaps) == 0
        assert st[0].stats.npts == 8640000
        assert len(st.get_gaps()) == 0 

    def test_load_3c_data(self):
        fileE = f'{examples_dir}/WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileN = f'{examples_dir}/WY.YMR..HHN__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileZ = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        
        dl = apply_detectors.DataLoader()
        dl.load_3c_data(fileE, fileN, fileZ, min_signal_percent=0)
        assert dl.continuous_data.shape == (8640000, 3)
        assert len(dl.metadata.keys()) == 10
        assert len(dl.gaps) == 3

    def test_save_meta_data_3c(self):
        # Make dummy stats values
        stats = obspy.core.trace.Stats()
        stats.sampling_rate = 100.0
        stats.delta = 0.01
        stats.starttime = UTC(2002, 1, 1, 0, 0, 0, 6000)
        stats.npts = 8640000
        stats.network = 'WY'
        stats.station = 'YMR'
        stats.channel = 'HHE'
        stats._format = 'MSEED'
        stats.mseed =  AttribDict({'dataquality': 'M', 
                                            'number_of_records': 1, 
                                            'encoding': 'STEIM1', 
                                            'byteorder': '>', 
                                            'record_length': 4096, 
                                            'filesize': 9326592})
        dl = apply_detectors.DataLoader()
        dl.save_meta_data(stats, three_channels=True)

        # End time is a read only field in Stats => this is what is should end up being
        endtime = UTC(2002, 1, 1, 23, 59, 59, 996000)

        dl_meta = dl.metadata
        # Mostly checking I didn't make any typos in the key name and that I didn't set the values
        # in meta_data to the wrong stats field
        assert dl_meta['sampling_rate'] == 100.0
        assert dl_meta['dt'] == 0.01
        assert dl_meta['starttime'] == stats.starttime
        assert dl_meta['endtime'] == endtime    
        assert dl_meta['npts'] == stats.npts
        assert dl_meta['network'] == stats.network
        assert dl_meta['station'] == stats.station
        # Make sure the epoch time converts back to the utc time correctly
        assert UTC(dl_meta['starttime_epoch']) == stats.starttime
        assert UTC(dl_meta['endtime_epoch']) == endtime
        # Make sure the 3C channel code has a ? for the orientation 
        assert dl_meta['channel'] == "HH?"

    def test_save_meta_1c(self):
        # Make dummy stats values
        stats = obspy.core.trace.Stats()
        stats.sampling_rate = 100.0
        stats.delta = 0.01
        stats.starttime = UTC(2002, 1, 1, 0, 0, 0, 6000)
        stats.npts = 8640000
        stats.network = 'WY'
        stats.station = 'YMR'
        stats.channel = 'HHZ'
        stats._format = 'MSEED'
        stats.mseed =  AttribDict({'dataquality': 'M', 
                                            'number_of_records': 1, 
                                            'encoding': 'STEIM1', 
                                            'byteorder': '>', 
                                            'record_length': 4096, 
                                            'filesize': 9326592})
        dl = apply_detectors.DataLoader()
        dl.save_meta_data(stats, three_channels=False)
        # Just need to check that the channel code for 1C stations has a Z for the orientation 
        # everything else is the same as 3C
        assert dl.metadata['channel'] == "HHZ"

    def test_load_3c_data_skip_day(self):
        fileE = f'{examples_dir}/WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileN = f'{examples_dir}/WY.YMR..HHN__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileZ = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        
        dl = apply_detectors.DataLoader()
        dl.load_3c_data(fileE, fileN, fileZ, min_signal_percent=99.5)
        assert dl.continuous_data == None
        assert dl.metadata == None
        assert len(dl.gaps) == 1
        assert dl.gaps[0][3] == "HH?"
        
    def test_load_1c_data(self):
        file = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        
        dl = apply_detectors.DataLoader()
        dl.load_1c_data(file, min_signal_percent=0)
        assert dl.continuous_data.shape == (8640000, 1)
        assert len(dl.metadata.keys()) == 10
        assert len(dl.gaps) == 1
        assert dl.gaps[0][3] == "HHZ"

    def test_load_1c_data_skip_day(self):
        file = f'{examples_dir}/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        
        dl = apply_detectors.DataLoader()
        dl.load_1c_data(file, min_signal_percent=1)
        assert dl.continuous_data == None
        assert dl.metadata == None
        assert len(dl.gaps) == 1
        assert dl.gaps[0][3] == "EHZ"

    def test_reset_loader(self):
        file = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        dl = apply_detectors.DataLoader()
        dl.load_1c_data(file, min_signal_percent=0)
        dl.reset_loader()
        assert dl.continuous_data == None
        assert dl.metadata == None
        assert dl.gaps == None

    def test_load_3c_data_reset_loader(self):
        fileE = f'{examples_dir}/WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileN = f'{examples_dir}/WY.YMR..HHN__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileZ = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        # Load data succesfully
        dl = apply_detectors.DataLoader()
        dl.load_3c_data(fileE, fileN, fileZ, min_signal_percent=0)
        # Try to load data but skip the day because not enough signal
        dl.load_3c_data(fileE, fileN, fileZ, min_signal_percent=99.5)
        # Continous_data and metadata should now be None, but gaps contains the entire day as a gap
        assert dl.continuous_data == None
        assert dl.metadata == None
    
    def test_load_1c_data_reset_loader(self):
        fileZ = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        # Load data succesfully
        dl = apply_detectors.DataLoader()
        dl.load_1c_data(fileZ, min_signal_percent=0)
        # Try to load data but skip the day because not enough signal
        dl.load_1c_data(fileZ, min_signal_percent=99.5)
        # Continous_data and metadata should now be None, but gaps contains the entire day as a gap
        assert dl.continuous_data == None
        assert dl.metadata == None

    def test_load_data_1c_prepend_previous(self):
        file1 = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        # Load data succesfully
        dl = apply_detectors.DataLoader(store_N_seconds=10)
        dl.load_1c_data(file1, min_signal_percent=0)
        previous_endtime = dl.metadata['endtime']

        # Check previous data is loaded
        assert dl.previous_continuous_data.shape == (1000, 1)
        assert dl.previous_endtime == previous_endtime

        file2 = f'{examples_dir}/WY.YMR..HHZ__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed'
        dl.load_1c_data(file2, min_signal_percent=0)
        
        # Check that the continuous data has been correctly updated
        assert dl.continuous_data.shape == (8641000, 1)
        assert dl.metadata['starttime'] == previous_endtime

        # Check that the previous data has been correctly updated
        assert dl.previous_endtime == dl.metadata['endtime']
        assert np.array_equal(dl.previous_continuous_data, dl.continuous_data[-1000:, :])


    def test_load_data_3c_prepend_previous(self):
        fileE = f'{examples_dir}/WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileN = f'{examples_dir}/WY.YMR..HHN__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileZ = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'        # Load data succesfully
        dl = apply_detectors.DataLoader(store_N_seconds=10)
        dl.load_3c_data(fileE, fileN, fileZ, min_signal_percent=0)
        previous_endtime = dl.metadata['endtime']

        # Check previous data is loaded
        assert dl.previous_continuous_data.shape == (1000, 3)
        assert dl.previous_endtime == previous_endtime

        fileE = f'{examples_dir}/WY.YMR..HHE__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed'
        fileN = f'{examples_dir}/WY.YMR..HHN__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed'
        fileZ = f'{examples_dir}/WY.YMR..HHZ__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed'
        dl.load_3c_data(fileE, fileN, fileZ, min_signal_percent=0)
        
        # Check that the continuous data has been correctly updated
        assert dl.continuous_data.shape == (8641000, 3)
        assert dl.metadata['starttime'] == previous_endtime

        # Check that the previous data has been correctly updated
        assert dl.previous_endtime == dl.metadata['endtime']
        assert np.array_equal(dl.previous_continuous_data, dl.continuous_data[-1000:, :])

    def test_load_data_1c_prepend_previous_trimmed(self):
        file1 = f'{examples_dir}/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        # Load data succesfully
        dl = apply_detectors.DataLoader(store_N_seconds=10)
        dl.load_1c_data(file1, min_signal_percent=0)
        previous_endtime = dl.metadata['endtime']

        # Check previous data is loaded
        assert dl.previous_continuous_data.shape == (1000, 1)
        assert dl.previous_endtime == previous_endtime

        file2 = f'{examples_dir}/WY.YWB..EHZ__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed'
        dl.load_1c_data(file2, min_signal_percent=0)
        
        # Check that the continuous data has been correctly updated
        assert dl.continuous_data.shape == (8641000, 1)
        assert dl.metadata['starttime'] == previous_endtime

        # Check that the previous data has been correctly updated
        assert dl.previous_endtime == dl.metadata['endtime']
        assert np.array_equal(dl.previous_continuous_data, dl.continuous_data[-1000:, :])
  
    def test_load_3c_data_reset_previous_day(self):
        fileE = f'{examples_dir}/WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileN = f'{examples_dir}/WY.YMR..HHN__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileZ = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        # Load data succesfully
        dl = apply_detectors.DataLoader(store_N_seconds=10)
        dl.load_3c_data(fileE, fileN, fileZ, min_signal_percent=0)
        # Try to load data but skip the day because not enough signal
        assert dl.previous_continuous_data is not None
        assert dl.previous_endtime is not None

        dl.load_3c_data(fileE, fileN, fileZ, min_signal_percent=99.5)
        # Continous_data and metadata should now be None, but gaps contains the entire day as a gap
        assert dl.previous_continuous_data == None
        assert dl.previous_endtime == None

    def test_load_1c_data_reset_previous_day(self):
        fileZ = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        # Load data succesfully
        dl = apply_detectors.DataLoader(store_N_seconds=10)
        dl.load_1c_data(fileZ, min_signal_percent=0)
        assert dl.previous_continuous_data is not None
        assert dl.previous_endtime is not None

        # Try to load data but skip the day because not enough signal
        dl.load_1c_data(fileZ, min_signal_percent=99.5)
        # Continous_data and metadata should now be None, but gaps contains the entire day as a gap
        assert dl.previous_continuous_data == None
        assert dl.previous_endtime == None

    def test_process_3c_p_runs(self):
        fileZ = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        # Load data succesfully
        dl = apply_detectors.DataLoader(store_N_seconds=10)
        dl.load_1c_data(fileZ, min_signal_percent=0)
        processed_data = dl.preprocess_1c_p()
        assert processed_data.shape == (8640000, 3)

class TestPhaseDetector():
    def test_n_windows(self):
        npts = 1000
        window_length = 200
        sliding_interval = 100
        pd = apply_detectors.PhaseDetector(window_length, sliding_interval)
        # A time-series of length 1000, with a window length of 200 and a sliding interval of 100 
        # should produce 9 windows
        assert pd.get_n_windows(npts) == 9


    if __name__ == '__main__':
        from pytests.test_apply_detectors import TestDataLoader
        dltester = TestDataLoader()
        dltester.test_process_3c_p_runs()