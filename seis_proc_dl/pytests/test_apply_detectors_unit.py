from seis_proc_dl.apply_to_continuous import apply_detectors
from obspy.core.utcdatetime import UTCDateTime as UTC
import numpy as np
import obspy
from obspy.core.util.attribdict import AttribDict
import os
import json

examples_dir = '/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/seis_proc_dl/pytests/example_files'
models_path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"

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
        assert len(dl.metadata.keys()) == 15
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
        dl.store_meta_data(stats, three_channels=True)

        # End time is a read only field in Stats => this is what is should end up being
        endtime = UTC(2002, 1, 1, 23, 59, 59, 996000)

        dl_meta = dl.metadata
        # Mostly checking I didn't make any typos in the key name and that I didn't set the values
        # in meta_data to the wrong stats field
        assert dl_meta['sampling_rate'] == 100.0
        assert dl_meta['dt'] == 0.01
        assert dl_meta['starttime'] == stats.starttime
        assert dl_meta['original_starttime'] == stats.starttime
        assert dl_meta['original_endtime'] == endtime    
        assert dl_meta['npts'] == stats.npts
        assert dl_meta['network'] == stats.network
        assert dl_meta['station'] == stats.station
        assert dl_meta['previous_appended'] == False
        assert dl_meta['original_npts'] == stats.npts

        # Make sure the epoch time converts back to the utc time correctly
        assert UTC(dl_meta['starttime_epoch']) == stats.starttime
        assert UTC(dl_meta['endtime_epoch']) == endtime
        assert UTC(dl_meta['original_starttime_epoch']) == stats.starttime

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
        dl.store_meta_data(stats, three_channels=False)
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
        assert len(dl.metadata.keys()) == 15
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
        previous_endtime = dl.metadata['original_endtime']
        
        # Previous_data should be none until the next day is read in 
        assert dl.previous_continuous_data == None
        assert dl.previous_endtime == None

        file2 = f'{examples_dir}/WY.YMR..HHZ__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed'
        dl.load_1c_data(file2, min_signal_percent=0)
        file2_endtime = dl.metadata['original_endtime']
        file2_enddata = dl.continuous_data[-1000:, :]

        # Check previous data is loaded
        saved_previous = dl.previous_continuous_data
        assert dl.previous_continuous_data.shape == (1000, 1)
        assert dl.previous_endtime == previous_endtime

        # load file2 without prepending previous so I can make sure the signals are the same
        dl2 = apply_detectors.DataLoader(store_N_seconds=0)
        dl2.load_1c_data(file2, min_signal_percent=0)

        # Check that the continuous data and metadata has been correctly updated
        assert dl.continuous_data.shape == (8641000, 1)
        assert dl.metadata['starttime'] == dl.metadata['original_starttime'] - 10
        assert UTC(dl.metadata['starttime_epoch']) == dl.metadata['original_starttime'] - 10
        assert dl.metadata['npts'] == 8641000
        assert dl.metadata['previous_appended'] == True
        assert np.array_equal(saved_previous, dl.continuous_data[0:1000, :])
        assert np.array_equal(dl2.continuous_data, dl.continuous_data[1000:, :])

        # Check that the original information is still preserved in the metadata
        st2 = obspy.read(file2)
        assert dl.metadata['original_starttime'] == st2[0].stats.starttime
        assert UTC(dl.metadata['original_starttime_epoch']) == st2[0].stats.starttime
        assert dl.metadata['original_npts'] == 8640000

        # Check that the previous data has been correctly updated
        file3 = f'{examples_dir}/WY.YMR..HHZ__2002-01-03T00:00:00.000000Z__2002-01-04T00:00:00.000000Z.mseed'
        dl.load_1c_data(file3, min_signal_percent=0)
        assert dl.previous_endtime == file2_endtime
        assert np.array_equal(dl.previous_continuous_data, file2_enddata)

    def test_load_data_3c_prepend_previous(self):
        fileE = f'{examples_dir}/WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileN = f'{examples_dir}/WY.YMR..HHN__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileZ = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'        # Load data succesfully
        dl = apply_detectors.DataLoader(store_N_seconds=10)
        dl.load_3c_data(fileE, fileN, fileZ, min_signal_percent=0)
        previous_endtime = dl.metadata['original_endtime']
        previous_enddata = dl.continuous_data[-1000:, :]

        # Previous_data should be none until the next day is read in 
        assert dl.previous_continuous_data == None
        assert dl.previous_endtime == None

        fileE = f'{examples_dir}/WY.YMR..HHE__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed'
        fileN = f'{examples_dir}/WY.YMR..HHN__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed'
        fileZ = f'{examples_dir}/WY.YMR..HHZ__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed'
        dl.load_3c_data(fileE, fileN, fileZ, min_signal_percent=0)
        
        # Check previous data is loaded
        assert dl.previous_continuous_data.shape == (1000, 3)
        assert dl.previous_endtime == previous_endtime
        assert np.array_equal(dl.previous_continuous_data, previous_enddata)

        # Check that the continuous data has been correctly updated
        assert dl.continuous_data.shape == (8641000, 3)
        assert dl.metadata['starttime'] == dl.metadata['original_starttime'] - 10
        assert np.array_equal(previous_enddata, dl.continuous_data[0:1000, :])

        # load the 2nd set of files without prepending previous so I can make sure the signals are the same
        dl2 = apply_detectors.DataLoader(store_N_seconds=0)
        dl2.load_3c_data(fileE, fileN, fileZ, min_signal_percent=0)
        assert np.array_equal(dl2.continuous_data, dl.continuous_data[1000:, :])

    def test_load_data_1c_prepend_previous_trimmed(self):
        """ Check that the prepending works when the previous data has be interpolated on the end"""

        # File 1 has gaps on either end, so the end has been filled 
        file1 = f'{examples_dir}/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        # Load data succesfully
        dl = apply_detectors.DataLoader(store_N_seconds=10)
        dl.load_1c_data(file1, min_signal_percent=0)
        previous_endtime = dl.metadata['original_endtime']

        # Previous_data should be none until the next day is read in 
        assert dl.previous_continuous_data == None
        assert dl.previous_endtime == None

        file2 = f'{examples_dir}/WY.YWB..EHZ__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed'
        dl.load_1c_data(file2, min_signal_percent=0)
        file2_enddtime = dl.metadata['original_endtime']
        file2_enddata = dl.continuous_data[-1000:, :]

        # Check previous data is loaded
        assert dl.previous_continuous_data.shape == (1000, 1)
        assert dl.previous_endtime == previous_endtime

        # Check that the continuous data has been correctly updated
        assert dl.continuous_data.shape == (8641000, 1)
        assert dl.metadata['starttime'] == dl.metadata['original_starttime'] - 10 

        file3 = f'{examples_dir}/WY.YWB..EHZ__2002-01-03T00:00:00.000000Z__2002-01-04T00:00:00.000000Z.mseed'
        dl.load_1c_data(file3, min_signal_percent=0)
        # Check that the previous data has been correctly updated
        assert dl.previous_endtime == file2_enddtime
        assert np.array_equal(dl.previous_continuous_data, file2_enddata)
  
    def test_load_3c_data_reset_previous_day(self):
        fileE = f'{examples_dir}/WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileN = f'{examples_dir}/WY.YMR..HHN__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileZ = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        # Load data succesfully
        dl = apply_detectors.DataLoader(store_N_seconds=10)
        dl.load_3c_data(fileE, fileN, fileZ, min_signal_percent=0)
        assert dl.previous_continuous_data == None
        assert dl.previous_endtime == None
       
        # Try to load data but skip the day because not enough signal
        dl.load_3c_data(fileE, fileN, fileZ, min_signal_percent=99.5)
        # Continous_data and metadata should now be None, but gaps contains the entire day as a gap
        assert dl.previous_continuous_data == None
        assert dl.previous_endtime == None

    def test_load_1c_data_reset_previous_day(self):
        fileZ = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        # Load data succesfully
        dl = apply_detectors.DataLoader(store_N_seconds=10)
        dl.load_1c_data(fileZ, min_signal_percent=0)
        assert dl.previous_continuous_data == None
        assert dl.previous_endtime == None

        # Try to load data but skip the day because not enough signal
        dl.load_1c_data(fileZ, min_signal_percent=99.5)
        # Continous_data and metadata should now be None, but gaps contains the entire day as a gap
        assert dl.previous_continuous_data == None
        assert dl.previous_endtime == None

    def test_n_windows(self):
        npts = 1000
        window_length = 200
        sliding_interval = 100
        dl = apply_detectors.DataLoader()
        # A time-series of length 1000, with a window length of 200 and a sliding interval of 100 
        # should produce 9 windows
        assert dl.get_n_windows(npts, window_length, sliding_interval) == 9

    def test_start_indices(self):
        npts = 1000
        window_length = 200
        sliding_interval = 100
        dl = apply_detectors.DataLoader()
        # A time-series of length 1000, with a window length of 200 and a sliding interval of 100 
        # should produce 9 windows - starting every 100 samples from 0 to 800
        start_inds = dl.get_sliding_window_start_inds(npts, window_length, sliding_interval)
        assert np.array_equal(start_inds, np.arange(0, 900, 100))
        assert start_inds[-1]+window_length == npts

    def test_start_indices_one_window(self):
        npts = 1008
        window_length = 1008
        sliding_interval = 500
        dl = apply_detectors.DataLoader()
        start_inds = dl.get_sliding_window_start_inds(npts, window_length, sliding_interval)
        assert np.array_equal(start_inds, np.arange(0, 1))
        assert start_inds[-1]+window_length == npts

    def test_get_padding_need_partial_window(self):
        npts = 1008
        window_length = 200
        sliding_interval = 100
        dl = apply_detectors.DataLoader()
        # Don't add anything to the start, will be 8 samples hanging over at the end. 
        # To include the 8 samples, will need to add 1/2 of the window length - 8
        total_npts, start_npts, end_npts = dl.get_padding(npts, window_length, 
                                                          sliding_interval, pad_start=False)
        
        assert start_npts == 0
        assert end_npts == 92
        assert total_npts == 1100

    def test_get_padding_include_last_edge(self):
        npts = 300
        window_length = 100
        sliding_interval = 100
        dl = apply_detectors.DataLoader()
        # Don't add anything to the start, will be 8 samples hanging over at the end. 
        # To include the 8 samples, will need to add 1/2 of the window length - 8
        total_npts, start_npts, end_npts = dl.get_padding(npts, window_length, 
                                                          sliding_interval, pad_start=False)
        
        assert start_npts == 0
        assert end_npts == 100
        assert total_npts == 400

    def test_get_padding_include_last_edge2(self):
        npts = 1008
        window_length = 1008
        sliding_interval = 500
        dl = apply_detectors.DataLoader()
        # Don't add anything to the start, will be 8 samples hanging over at the end. 
        # To include the 8 samples, will need to add 1/2 of the window length - 8
        total_npts, start_npts, end_npts = dl.get_padding(npts, window_length, 
                                                          sliding_interval, pad_start=False)
        
        assert start_npts == 0
        assert end_npts == 500
        assert total_npts == 1508

    def test_get_padding_double_pad(self):
        """Have to pad to be evenly divisible by sliding window/window_length and
          have to pad to include the last trace edge"""
        npts = 1000
        window_length = 1008
        sliding_interval = 500
        dl = apply_detectors.DataLoader()
        total_npts, start_npts, end_npts = dl.get_padding(npts, window_length, 
                                                          sliding_interval, pad_start=False)
        
        assert start_npts == 0
        assert end_npts == 508
        assert total_npts == 1508

    def test_get_pad_start(self):
        npts = 1008
        window_length = 1008
        sliding_interval = 500
        dl = apply_detectors.DataLoader()
        total_npts, start_npts, end_npts = dl.get_padding(npts, window_length, 
                                                          sliding_interval, pad_start=True)
        
        assert start_npts == 254
        assert end_npts == 746 #(1008 - (254+8)) => 8 is the number of left over samples at the end after adding the extra edge (254)
        assert total_npts == 2008 # works out to adding 2 extra sliding windows

    # The following tests building on each other/are all components of the last function test
    def test_get_padding(self):
        npts = 2000
        window = 1008
        slide = 500
        dl = apply_detectors.DataLoader()
        npts_padded, start_pad, end_pad = dl.get_padding(npts, window, slide)
        assert npts_padded == 2508
        assert start_pad == 254
        assert end_pad == 254

    def test_get_n_windows(self):
        npts = 2508
        window = 1008
        slide = 500
        dl = apply_detectors.DataLoader()
        assert dl.get_n_windows(npts, window, slide) == 4

    def test_get_sliding_window_start_inds(self):
        npts = 2508
        window = 1008
        slide = 500
        dl = apply_detectors.DataLoader()
        assert np.array_equal(dl.get_sliding_window_start_inds(npts, window, slide), np.arange(0, 2000, 500))

    def test_add_padding_3C(self):
        continuous_data = np.arange(6000).reshape((2000, 3))
        start_pad = 254
        end_pad = 254
        dl = apply_detectors.DataLoader()
        padded = dl.add_padding(continuous_data, start_pad, end_pad)
        assert np.array_equal(np.unique(padded[0:start_pad, :]), [0, 1, 2])
        assert np.array_equal(np.unique(padded[-end_pad:, :]), [5997, 5998, 5999])

    def test_add_padding_1C(self):
        continuous_data = np.expand_dims(np.arange(2000), 1)
        start_pad = 254
        end_pad = 254
        dl = apply_detectors.DataLoader()
        padded = dl.add_padding(continuous_data, start_pad, end_pad)
        assert np.array_equal(np.unique(padded[0:start_pad, :]), [0])
        assert np.array_equal(np.unique(padded[-end_pad:, :]), [1999])

    def test_normalize(self):
        dl = apply_detectors.DataLoader()
        X = np.zeros((1008, 3))
        X[100, 0] = 1000
        X[200, 1] = -2000
        normalized = dl.normalize_example(X)
        assert np.allclose(np.max(abs(normalized), axis=0), [1, 1, 0])

    def test_format_continuous_for_unet_no_proc_3c(self):
        dl = apply_detectors.DataLoader()
        #proc_func = dl.process_3c_P
        dl.continuous_data = np.arange(6000).reshape((3, 2000)).T
        window = 1008
        slide = 500
        formatted, start_pad_npts, end_pad_npts = dl.format_continuous_for_unet(window, slide, normalize=False)
        assert start_pad_npts == 254
        assert end_pad_npts == 254
        
        assert formatted.shape == (4, 1008, 3)
        assert np.array_equal(np.unique(formatted[0, :start_pad_npts, :]), [0, 2000, 4000])
        assert np.array_equal(np.unique(formatted[-1, -end_pad_npts:, :]), [1999, 3999, 5999])
        
        # Check first trace values
        assert np.array_equal(formatted[0, start_pad_npts:, 0], np.arange(0, 754))
        assert np.array_equal(formatted[0, start_pad_npts:, 1], np.arange(2000, 2754))
        assert np.array_equal(formatted[0, start_pad_npts:, 2], np.arange(4000, 4754))

        # Check second trace values
        assert np.array_equal(formatted[1, :, 0], np.arange(246, 1254))
        assert np.array_equal(formatted[1, :, 1], np.arange(2246, 3254))
        assert np.array_equal(formatted[1, :, 2], np.arange(4246, 5254))

        # Check third trace values
        assert np.array_equal(formatted[2, :, 0], np.arange(746, 1754))
        assert np.array_equal(formatted[2, :, 1], np.arange(2746, 3754))
        assert np.array_equal(formatted[2, :, 2], np.arange(4746, 5754))

        # Check fourth trace values
        assert np.array_equal(formatted[3, :-end_pad_npts, 0], np.arange(1246, 2000))
        assert np.array_equal(formatted[3, :-end_pad_npts, 1], np.arange(3246, 4000))
        assert np.array_equal(formatted[3, :-end_pad_npts, 2], np.arange(5246, 6000))

    def test_format_continuous_for_unet_no_proc_1c(self):
        dl = apply_detectors.DataLoader()
        #proc_func = dl.process_3c_P
        dl.continuous_data = np.expand_dims(np.arange(2000), 1)
        window = 1008
        slide = 500
        formatted, start_pad_npts, end_pad_npts = dl.format_continuous_for_unet(window, slide, normalize=False)
        assert start_pad_npts == 254
        assert end_pad_npts == 254
        
        assert formatted.shape == (4, 1008, 1)
        assert np.array_equal(np.unique(formatted[0, :start_pad_npts, :]), [00])
        assert np.array_equal(np.unique(formatted[-1, -end_pad_npts:, :]), [1999])
        
        # Check first trace values
        assert np.array_equal(formatted[0, start_pad_npts:, 0], np.arange(0, 754))

        # Check second trace values
        assert np.array_equal(formatted[1, :, 0], np.arange(246, 1254))

        # Check third trace values
        assert np.array_equal(formatted[2, :, 0], np.arange(746, 1754))

        # Check fourth trace values
        assert np.array_equal(formatted[3, :-end_pad_npts, 0], np.arange(1246, 2000))

    def test_format_continuous_for_unet_proc_3c_P(self):
        dl = apply_detectors.DataLoader()
        proc_func = dl.process_3c_P
        dl.continuous_data = np.arange(6000).reshape((3, 2000)).T
        window = 1008
        slide = 500
        formatted, start_pad_npts, end_pad_npts = dl.format_continuous_for_unet(window, slide,
                                                                                 processing_function=proc_func,
                                                                                 normalize=True)

        assert formatted.shape == (4, 1008, 3)

    def test_format_continuous_for_unet_proc_3c_S(self):
        dl = apply_detectors.DataLoader()
        proc_func = dl.process_3c_S
        dl.continuous_data = np.arange(6000).reshape((3, 2000)).T
        window = 1008
        slide = 500
        formatted, start_pad_npts, end_pad_npts = dl.format_continuous_for_unet(window, slide,
                                                                                 processing_function=proc_func,
                                                                                 normalize=True)
        assert formatted.shape == (4, 1008, 3)

    def test_format_continuous_for_unet_proc_1c_P(self):
        dl = apply_detectors.DataLoader()
        proc_func = dl.process_1c_P
        dl.continuous_data = np.expand_dims(np.arange(2000), 1)
        window = 1008
        slide = 500
        formatted, start_pad_npts, end_pad_npts = dl.format_continuous_for_unet(window, slide,
                                                                                 processing_function=proc_func,
                                                                                 normalize=True)
        assert formatted.shape == (4, 1008, 1)

    def test_format_continuous_for_unet_3c_P_normalize(self):
        dl = apply_detectors.DataLoader()
        dl.continuous_data = np.arange(6000).reshape((3, 2000)).T
        window = 1008
        slide = 500
        formatted, start_pad_npts, end_pad_npts = dl.format_continuous_for_unet(window, slide,
                                                                                 normalize=True)

        assert formatted.shape == (4, 1008, 3)
        assert np.allclose(np.max(abs(formatted), axis=1), np.ones((4, 3)))

    def test_format_continuous_for_unet_3c_S_normalize(self):
        dl = apply_detectors.DataLoader()
        dl.continuous_data = np.arange(6000).reshape((3, 2000)).T
        window = 1008
        slide = 500
        formatted, start_pad_npts, end_pad_npts = dl.format_continuous_for_unet(window, slide,
                                                                                 normalize=True)
        assert formatted.shape == (4, 1008, 3)
        assert np.allclose(np.max(abs(formatted), axis=1), np.ones((4, 3)))

    def test_format_continuous_for_unet_1c_P_normalize(self):
        dl = apply_detectors.DataLoader()
        dl.continuous_data = np.expand_dims(np.arange(2000), 1)
        window = 1008
        slide = 500
        formatted, start_pad_npts, end_pad_npts = dl.format_continuous_for_unet(window, slide,
                                                                                 normalize=True)
        assert formatted.shape == (4, 1008, 1)
        assert np.allclose(np.max(formatted, axis=1), np.ones((4, 1)))

    def test_process_1c_P(self):
        vert_mat = np.loadtxt(f'{examples_dir}/ben_data/detectors/uNetOneComponentP/PB.B206.EHZ.zrunet_p.txt', delimiter=',')
        vertical_ref = np.loadtxt(f'{examples_dir}/ben_data/detectors/uNetOneComponentP/PB.B206.EHZ.PROC.zrunet_p.txt', delimiter=',')[:, 1]
        vertical = vert_mat[:, 1]
        t = vert_mat[:, 0]

        assert len(vertical) == 360000
        assert len(vertical) == len(vertical_ref)

        sampling_rate = round(1./(t[1] - t[0]))
        assert sampling_rate == 100, 'sampling rate should be 100 Hz'

        dl = apply_detectors.DataLoader()
        vertical_proc = dl.process_1c_P(vertical[:, None], desired_sampling_rate=sampling_rate)

        assert max(abs(vertical_proc[:, 0] - vertical_ref)) < 1.e-1

    def test_process_3c_P(self):
        vert_mat = np.loadtxt(f'{examples_dir}/ben_data/detectors/uNetThreeComponentP/PB.B206.EHZ.zrunet_p.txt', delimiter=',')
        north = np.loadtxt(f'{examples_dir}/ben_data/detectors/uNetThreeComponentP/PB.B206.EH1.zrunet_p.txt', delimiter=',')[:, 1]
        east  = np.loadtxt(f'{examples_dir}/ben_data/detectors/uNetThreeComponentP/PB.B206.EH2.zrunet_p.txt', delimiter=',')[:, 1]
        vertical_ref = np.loadtxt(f'{examples_dir}/ben_data/detectors/uNetThreeComponentP/PB.B206.EHZ.PROC.zrunet_p.txt', delimiter=',')[:, 1]
        north_ref = np.loadtxt(f'{examples_dir}/ben_data/detectors/uNetThreeComponentP/PB.B206.EH1.PROC.zrunet_p.txt', delimiter=',')[:, 1]
        east_ref = np.loadtxt(f'{examples_dir}/ben_data/detectors/uNetThreeComponentP/PB.B206.EH2.PROC.zrunet_p.txt', delimiter=',')[:, 1]
        vertical = vert_mat[:, 1]
        t = vert_mat[:, 0]

        assert len(vertical) == 360000
        assert len(vertical) == len(north)
        assert len(vertical) == len(east)
        assert len(vertical) == len(vertical_ref)
        assert len(vertical) == len(north_ref)
        assert len(vertical) == len(east_ref)

        sampling_rate = round(1./(t[1] - t[0]))
        assert sampling_rate == 100, 'sampling rate should be 100 Hz'

        wfs = np.stack([east, north, vertical], axis=1)
        assert np.array_equal(wfs[:, 0], east)
        assert np.array_equal(wfs[:, 1], north)  
        assert np.array_equal(wfs[:, 2], vertical)  

        dl = apply_detectors.DataLoader()
        processed = dl.process_3c_P(wfs, desired_sampling_rate=sampling_rate)

        east_proc = processed[:, 0]
        north_proc = processed[:, 1]
        vertical_proc = processed[:, 2]
        assert max(abs(vertical_proc - vertical_ref)) < 1.e-1
        assert max(abs(north_proc - north_ref)) < 1.e-1
        assert max(abs(east_proc - east_ref)) < 1.e-1


    def test_process_3c_S(self):
        vert_mat = np.loadtxt(f'{examples_dir}/ben_data/detectors/uNetThreeComponentS/PB.B206.EHZ.zrunet_s.txt', delimiter=',')
        north = np.loadtxt(f'{examples_dir}/ben_data/detectors/uNetThreeComponentS/PB.B206.EH1.zrunet_s.txt', delimiter=',')[:, 1]
        east  = np.loadtxt(f'{examples_dir}/ben_data/detectors/uNetThreeComponentS/PB.B206.EH2.zrunet_s.txt', delimiter=',')[:, 1]
        vertical_ref = np.loadtxt(f'{examples_dir}/ben_data/detectors/uNetThreeComponentS/PB.B206.EHZ.PROC.zrunet_s.txt', delimiter=',')[:, 1]
        north_ref = np.loadtxt(f'{examples_dir}/ben_data/detectors/uNetThreeComponentS/PB.B206.EH1.PROC.zrunet_s.txt', delimiter=',')[:, 1]
        east_ref = np.loadtxt(f'{examples_dir}/ben_data/detectors/uNetThreeComponentS/PB.B206.EH2.PROC.zrunet_s.txt', delimiter=',')[:, 1]
        vertical = vert_mat[:, 1]
        t = vert_mat[:, 0]

        assert len(vertical) == 360000
        assert len(vertical) == len(north)
        assert len(vertical) == len(east)
        assert len(vertical) == len(vertical_ref)
        assert len(vertical) == len(north_ref)
        assert len(vertical) == len(east_ref)

        sampling_rate = round(1./(t[1] - t[0]))
        assert sampling_rate == 100, 'sampling rate should be 100 Hz'

        wfs = np.stack([east, north, vertical], axis=1)
        assert np.array_equal(wfs[:, 0], east)
        assert np.array_equal(wfs[:, 1], north)  
        assert np.array_equal(wfs[:, 2], vertical)  

        dl = apply_detectors.DataLoader()
        processed = dl.process_3c_S(wfs, desired_sampling_rate=sampling_rate)

        east_proc = processed[:, 0]
        north_proc = processed[:, 1]
        vertical_proc = processed[:, 2]
        assert max(abs(vertical_proc - vertical_ref)) < 1.e-1
        assert max(abs(north_proc - north_ref)) < 1.e-1
        assert max(abs(east_proc - east_ref)) < 1.e-1

    def test_format_continuous_for_unet_1c_P_real_data(self):
        dl = apply_detectors.DataLoader()
        file1 = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        dl.load_1c_data(file1)
        window_length = 1008
        sliding_interval = 500
        data_unproc, start_pad_npts, end_pad_npts = dl.format_continuous_for_unet(window_length,
                                                                           sliding_interval,
                                                                           normalize=False)
        assert np.array_equal(np.unique(data_unproc[-1][-end_pad_npts:]), dl.continuous_data[-1])
        assert np.array_equal(np.unique(data_unproc[0][:start_pad_npts]), dl.continuous_data[0])
        assert np.array_equal(data_unproc[0][start_pad_npts:], dl.continuous_data[0:(window_length-start_pad_npts)])
        assert np.array_equal(data_unproc[-1][0:-end_pad_npts], dl.continuous_data[-(window_length-end_pad_npts):])

    def test_write_data_info_1c(self):
        dl = apply_detectors.DataLoader()
        file1 = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        dl.load_1c_data(file1)
        dl.write_data_info("gaps.json", examples_dir)
    
        with open(os.path.join(examples_dir, "gaps.json"), "r") as fp:
            json_dict = json.load(fp)

        assert UTC(json_dict['starttime']) == UTC(json_dict['starttime_epoch'])
        assert UTC(json_dict['original_starttime']) == UTC(json_dict['original_starttime_epoch'])
        assert UTC(json_dict['original_endtime']) == UTC(json_dict['original_endtime_epoch'])
        assert json_dict['npts'] == json_dict['original_npts']
        assert len(json_dict['gaps']) == 1
        assert len(json_dict['gaps'][0]) == 6
        assert UTC(json_dict['gaps'][0][2]) > UTC(json_dict['gaps'][0][1])
        assert UTC(json_dict['gaps'][0][1]) > UTC(json_dict['starttime'])
        assert UTC(json_dict['gaps'][0][2]) > UTC(json_dict['starttime'])
        assert UTC(json_dict['gaps'][0][1]) < UTC(json_dict['original_endtime'])
        assert UTC(json_dict['gaps'][0][2]) < UTC(json_dict['original_endtime'])
        assert int((json_dict['gaps'][0][2]-json_dict['gaps'][0][1])*json_dict['sampling_rate']) == json_dict['gaps'][0][3]
        assert int(json_dict['gaps'][0][5]-json_dict['gaps'][0][4]) == json_dict['gaps'][0][3]
        assert json_dict['channel'] == 'HHZ'
        assert json_dict['channel'] == json_dict['gaps'][0][0]

    def test_write_data_info_3c(self):
        dl = apply_detectors.DataLoader()
        fileE = f'{examples_dir}/WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileN = f'{examples_dir}/WY.YMR..HHN__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        fileZ = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        dl.load_3c_data(fileE, fileN, fileZ)
        dl.write_data_info("gaps.json", examples_dir)
    
        with open(os.path.join(examples_dir, "gaps.json"), "r") as fp:
            json_dict = json.load(fp)

        assert UTC(json_dict['starttime']) == UTC(json_dict['starttime_epoch'])
        assert UTC(json_dict['original_starttime']) == UTC(json_dict['original_starttime_epoch'])
        assert UTC(json_dict['original_endtime']) == UTC(json_dict['original_endtime_epoch'])
        assert json_dict['npts'] == json_dict['original_npts']
        assert len(json_dict['gaps']) == 3
        assert len(json_dict['gaps'][0]) == 6
        assert UTC(json_dict['gaps'][0][2]) > UTC(json_dict['gaps'][0][1])
        assert UTC(json_dict['gaps'][0][1]) > UTC(json_dict['starttime'])
        assert UTC(json_dict['gaps'][0][2]) > UTC(json_dict['starttime'])
        assert UTC(json_dict['gaps'][0][1]) < UTC(json_dict['original_endtime'])
        assert UTC(json_dict['gaps'][0][2]) < UTC(json_dict['original_endtime'])
        assert int((json_dict['gaps'][0][2]-json_dict['gaps'][0][1])*json_dict['sampling_rate']) == json_dict['gaps'][0][3]
        assert int(json_dict['gaps'][0][5]-json_dict['gaps'][0][4]) == json_dict['gaps'][0][3]
        assert json_dict['channel'] != json_dict['gaps'][0][0]
        assert json_dict['channel'] == 'HH?'

class TestPhaseDetector():
    def test_class_init(self):
        model_file = f"{models_path}/oneCompPDetectorMEW_model_022.pt"
        pdet = apply_detectors.PhaseDetector(model_file, 1, min_presigmoid_value=-70, device="cpu")
        assert pdet.device.type == 'cpu'
        assert pdet.min_presigmoid_value == -70
        assert pdet.unet is not None

    def test_apply_model_to_batch_no_center(self):
        X = np.zeros((2, 1008, 1))
        model_file = f"{models_path}/oneCompPDetectorMEW_model_022.pt"
        pdet = apply_detectors.PhaseDetector(model_file, 1, min_presigmoid_value=-70, device="cpu")
        post_probs = pdet.apply_model_to_batch(X)
        assert post_probs.shape == (2, 1008)
        assert np.max((post_probs*100).astype(int)) == 0
 
    def test_apply_model_to_batch_no_center_one_example(self):
        X = np.zeros((1, 1008, 1))
        model_file = f"{models_path}/oneCompPDetectorMEW_model_022.pt"
        pdet = apply_detectors.PhaseDetector(model_file, 1, min_presigmoid_value=-70, device="cpu")
        post_probs = pdet.apply_model_to_batch(X)
        assert post_probs.shape == (1, 1008)
        assert np.max((post_probs*100).astype(int)) == 0
 
    def test_apply_model_to_batch_one_example(self):
        X = np.zeros((1, 1008, 1))
        model_file = f"{models_path}/oneCompPDetectorMEW_model_022.pt"
        pdet = apply_detectors.PhaseDetector(model_file, 1, min_presigmoid_value=-70, device="cpu")
        post_probs = pdet.apply_model_to_batch(X, center_window=250)
        assert post_probs.shape == (1, 500)

    def test_trim_post_probs_no_trim(self):
        post_probs = np.arange(2508)
        model_file = f"{models_path}/oneCompPDetectorMEW_model_022.pt"
        pdet = apply_detectors.PhaseDetector(model_file, 1, device="cpu")
        trimmed = pdet.trim_post_probs(post_probs, 254, 254, 254)
        assert trimmed.shape == (2508, )

    def test_trim_post_probs_end_trim(self):
        post_probs = np.arange(1256)
        model_file = f"{models_path}/oneCompPDetectorMEW_model_022.pt"
        pdet = apply_detectors.PhaseDetector(model_file, 1, device="cpu")
        trimmed = pdet.trim_post_probs(post_probs, 0, 510, 254)
        assert trimmed.shape == (1000, )
        assert trimmed[-1] == 999

    def test_save_probs(self):
        model_file = f"{models_path}/oneCompPDetectorMEW_model_022.pt"
        pdet = apply_detectors.PhaseDetector(model_file, 1, device="cpu")
        post_probs = np.arange(1, 1001)/1000
        outfile = f"{examples_dir}/postprobs.mseed"
        stats = obspy.core.trace.Stats()
        stats.npts = 1000
        pdet.save_post_probs(outfile, post_probs, stats)

        st = obspy.read(outfile)
        assert np.max(st[0].data[0:9]) == 0
        assert st[0].data[10] == 1
        assert st[0].data[-1] == 100
        assert st[0].stats.npts == 1000

    def test_flatten_post_probs(self):
        model_file = f"{models_path}/oneCompPDetectorMEW_model_022.pt"
        pdet = apply_detectors.PhaseDetector(model_file, 1, device="cpu")
        post_probs = np.arange(50).reshape((10, 5))
        flattened = pdet.flatten_model_output(post_probs)
        assert np.array_equal(flattened, np.arange(50))

    def test_apply_to_continous_no_center_window(self):
        data = np.zeros((11, 1008, 1))
        model_file = f"{models_path}/oneCompPDetectorMEW_model_022.pt"
        pdet = apply_detectors.PhaseDetector(model_file, 1, device="cpu")
        post_probs = pdet.apply_to_continuous(data, batchsize=2)
        assert post_probs.shape == (11, 1008)
        # Make sure there aren't any examples with all zeros (like they were skipped)
        assert len(np.unique(np.where(post_probs == np.zeros((1, 1008)))[0])) == 0

    def test_apply_to_continous_center_window(self):
        data = np.zeros((11, 1008, 1))
        model_file = f"{models_path}/oneCompPDetectorMEW_model_022.pt"
        pdet = apply_detectors.PhaseDetector(model_file, 1, device="cpu")
        post_probs = pdet.apply_to_continuous(data, batchsize=2, center_window=250)
        assert post_probs.shape == (11, 500)
        # Make sure there aren't any examples with all zeros (like they were skipped)
        assert len(np.unique(np.where(post_probs == np.zeros((1, 500)))[0])) == 0

    def test_make_outfile_name_1c(self):
        model_file = f"{models_path}/oneCompPDetectorMEW_model_022.pt"
        pdet = apply_detectors.PhaseDetector(model_file, 1, device="cpu")
        file1 = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        outfile = pdet.make_outfile_name(file1, examples_dir, "P")
        assert os.path.basename(outfile) == "probs.P__WY.YMR..HHZ__2002-01-01__2002-01-02.mseed"
        assert os.path.dirname(outfile) == examples_dir

    def test_make_outfile_name_3c(self):
        model_file = f"{models_path}/pDetectorMew_model_026.pt"
        pdet = apply_detectors.PhaseDetector(model_file, 3, device="cpu")
        file1 = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        outfile = pdet.make_outfile_name(file1, examples_dir, "P")
        assert os.path.basename(outfile) == "probs.P__WY.YMR..HH__2002-01-01__2002-01-02.mseed"
        assert os.path.dirname(outfile) == examples_dir        

if __name__ == '__main__':
    from seis_proc_dl.pytests.test_apply_detectors_unit import TestDataLoader
    dltester = TestDataLoader()
    dltester.test_load_data_3c_prepend_previous()
  
    # from seis_proc_dl.pytests.test_apply_detectors_unit import TestPhaseDetector
    # pdtester = TestPhaseDetector()
    # pdtester.test_save_probs()