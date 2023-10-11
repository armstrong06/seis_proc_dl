from seis_proc_dl.apply_to_continuous import apply_detectors
from obspy.core.utcdatetime import UTCDateTime as UTC
import numpy as np
import obspy
from obspy.core.util.attribdict import AttribDict
import os

examples_dir = '/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/seis_proc_dl/pytests/example_files'
models_path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"

class TestWorkflow():

    def test_prepend_previous_and_save(self):
        file1 = f'{examples_dir}/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        # Load first days data data succesfully
        dl = apply_detectors.DataLoader(store_N_seconds=10)
        dl.load_1c_data(file1, min_signal_percent=0)

        # load the second days data
        file2 = f'{examples_dir}/WY.YWB..EHZ__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed'
        dl.load_1c_data(file2, min_signal_percent=0)

        # Just want to check that the metadata is correct after saving =>
        # don't need to do any additional formatting

        model_file = f"{models_path}/oneCompPDetectorMEW_model_022.pt"
        pd = apply_detectors.PhaseDetector(model_file,
                                    1,
                                    min_presigmoid_value=-70,
                                    device="cpu")

        fake_data = np.arange(dl.metadata['npts'])
        outfile = pd.make_outfile_name("fake.EHZ__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed", examples_dir)
        pd.save_post_probs(outfile, fake_data, dl.metadata)
        st = obspy.read(outfile)
        assert st[0].stats.starttime == dl.metadata['starttime']
        assert st[0].stats.endtime == dl.metadata['endtime']
        assert st[0].stats.npts == dl.metadata['npts']
        assert st[0].stats.delta == dl.metadata['dt']
        assert st[0].stats.station == dl.metadata['station']
        assert st[0].stats.network == dl.metadata['network']
        assert st[0].stats.channel == dl.metadata['channel']

    def load_day_and_apply_1c(self):
        dl = apply_detectors.DataLoader()
        file1 = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        dl.load_1c_data(file1)
        window_length = 1008
        sliding_interval = 500
        data_unproc, start_pad_npts, end_pad_npts = dl.format_continuous_for_unet(window_length,
                                                                           sliding_interval,
                                                                           dl.process_1c_P
                                                                           )
   
        model_file = f"{models_path}/oneCompPDetectorMEW_model_022.pt"

        pd = apply_detectors.PhaseDetector(model_file,
                                           1,
                                           min_presigmoid_value=-70,
                                           device="cpu")
        
        outfile = pd.make_outfile_name(file1, examples_dir)
        unet_output = pd.apply_to_continuous(data, center_window=250)
        cont_post_probs = pd.flatten_model_output(unet_output)
        cont_post_probs = pd.trim_post_probs(cont_post_probs, 
                                             start_pad_npts, 
                                             end_pad_npts,
                                             254)
        pd.save_post_probs(outfile, cont_post_probs, dl.metadata)

        st = obspy.read(outfile)
        print(st[0].stats)
        assert st[0].data.shape == (8640000, )

    def load_day_and_apply_3c(self):
        pass
    
    def load_day_and_apply_1c_prepend_previous(self):
        dl = apply_detectors.DataLoader(store_N_seconds=10)
        file1 = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        file2 = f'{examples_dir}/WY.YMR..HHZ__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed'
        dl.load_1c_data(file1)
        dl.load_1c_data(file2)

        window_length = 1008
        sliding_interval = 500
        data, start_pad_npts, end_pad_npts = dl.format_continuous_for_unet(window_length,
                                                                           sliding_interval,
                                                                           dl.process_1c_P,
                                                                           normalize=True
                                                                           )
        model_file = f"{models_path}/oneCompPDetectorMEW_model_022.pt"

        pd = apply_detectors.PhaseDetector(model_file,
                                           1,
                                           min_presigmoid_value=-70,
                                           device="cpu",
                                           num_torch_threads=10)
        
        outfile = pd.make_outfile_name(file2, examples_dir)
        unet_output = pd.apply_to_continuous(data, center_window=250)
        cont_post_probs = pd.flatten_model_output(unet_output)
        cont_post_probs = pd.trim_post_probs(cont_post_probs, 
                                             start_pad_npts, 
                                             end_pad_npts,
                                             254)
        pd.save_post_probs(outfile, cont_post_probs, dl.metadata)

        st = obspy.read(outfile)
        print(st[0].stats)
        assert st[0].data.shape == (8641000, )

if __name__ == '__main__':
    from seis_proc_dl.pytests.test_apply_detectors_together import TestWorkflow
    wftester = TestWorkflow()
    wftester.load_day_and_apply_1c_prepend_previous()