from seis_proc_dl.apply_to_continuous import apply_pyuussml_detectors
from obspy.core.utcdatetime import UTCDateTime as UTC
import numpy as np
import obspy
from obspy.core.util.attribdict import AttribDict
import os
import json
from copy import deepcopy

examples_dir = '/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/seis_proc_dl/pytests/example_files'
models_path = "/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/mlmodels/detectors"
apply_detectors_outdir = f"{examples_dir}/applypyuussmldetector_results"

apply_detector_config = {"paths":{
                            "data_dir":examples_dir,
                            "output_dir":apply_detectors_outdir,
                            "one_comp_p_model":f"{models_path}/uNetOneComponentP/models/detectorsUNetOneComponentP.onnx",
                            "three_comp_p_model":f"{models_path}/uNetThreeComponentP/models/detectorsUNetThreeComponentP.onnx",
                            "three_comp_s_model":f"{models_path}/uNetThreeComponentS/models/detectorsUNetThreeComponentS.onnx",
                            }, 
                        "unet":{
                            "device":"cpu",
                        },
                        "dataloader":{
                            "store_N_seconds":10,
                            #"expected_file_duration_s":3600,
                            "min_signal_percent":0,
                        }}

class TestApplyDetectorPyuussml():
    """ These tests are pretty slow to run b/c applying the detector to 256 examples"""
         
    def test_init_1c(self):
        applier = apply_pyuussml_detectors.ApplyDetectorPyuussml(1, apply_detector_config)
        assert applier.data_dir == examples_dir
        assert applier.outdir == f"{examples_dir}/applypyuussmldetector_results"
        assert applier.device == "cpu"
        assert applier.p_model_file == f"{models_path}/uNetOneComponentP/models/detectorsUNetOneComponentP.onnx"
        assert applier.s_model_file == None
        assert applier.dataloader.store_N_seconds == 10
        assert applier.p_detector is not None
        assert applier.s_detector == None
        assert applier.p_proc_func.__qualname__ == "DataLoader.process_1c_P"
        assert applier.min_signal_percent == 0
        #assert applier.expected_file_duration_s == 3600

    def test_init_3c(self):
        applier = apply_pyuussml_detectors.ApplyDetectorPyuussml(3, apply_detector_config)
        assert applier.data_dir == examples_dir
        assert applier.outdir == f"{examples_dir}/applypyuussmldetector_results"
        assert applier.device == "cpu"
        assert applier.p_model_file == f"{models_path}/uNetThreeComponentP/models/detectorsUNetThreeComponentP.onnx"
        assert applier.s_model_file == f"{models_path}/uNetThreeComponentS/models/detectorsUNetThreeComponentS.onnx"
        assert applier.dataloader.store_N_seconds == 10
        assert applier.p_detector is not None
        assert applier.s_detector is not None
        assert applier.p_proc_func.__qualname__ == "DataLoader.process_3c_P"
        assert applier.min_signal_percent == 0
        #assert applier.expected_file_duration_s == 3600

    def test_apply_to_file_day_1c(self):
        applier = apply_pyuussml_detectors.ApplyDetectorPyuussml(1, apply_detector_config)
        applier.apply_to_one_file([f"{examples_dir}/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed"],
                                  debug_N_examples=-1)
        
        expected_p_probs_file = f"{apply_detectors_outdir}/probs.P__WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed"
        assert os.path.exists(expected_p_probs_file)
        probs_st = obspy.read(expected_p_probs_file)
        #assert probs_st[0].data.shape == (256*1008, )
        # There is no data at the beginning of this trace => no detections
        assert probs_st[0].data.max() <= 100 and probs_st[0].data.max() >= 0 
        assert probs_st[0].stats.station == "YWB"
        assert not os.path.exists(f"{examples_dir}/probs.S__WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed")
        expected_json_file = f"{apply_detectors_outdir}/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.json"
        assert os.path.exists(expected_json_file)
        with open(os.path.join(examples_dir, expected_json_file), "r") as fp:
            json_dict = json.load(fp)
        assert json_dict['channel'] == "EHZ"        

        os.remove(expected_p_probs_file)
        os.remove(expected_json_file)

    def test_apply_to_file_day_3c(self):
        applier = apply_pyuussml_detectors.ApplyDetectorPyuussml(3, apply_detector_config)
        files = [f"{examples_dir}/WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed",
                 f"{examples_dir}/WY.YMR..HHN__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed",
                 f"{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed"]
        applier.apply_to_one_file(files,
                                  debug_N_examples=-1)
        
        # P Probs - 3c name should have E or 1 channel
        expected_p_probs_file = f"{apply_detectors_outdir}/probs.P__WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed"
        assert os.path.exists(expected_p_probs_file)
        probs_st = obspy.read(expected_p_probs_file)
        #assert probs_st[0].data.shape == (256*1008, )
        assert probs_st[0].data.max() <= 100 and probs_st[0].data.max() > 1
        assert probs_st[0].stats.station == "YMR"
        # S Probs - 3c name should have E or 1 channel
        expected_s_probs_file = f"{apply_detectors_outdir}/probs.S__WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed"
        assert os.path.exists(expected_s_probs_file)
        probs_st = obspy.read(expected_s_probs_file)
        #assert probs_st[0].data.shape == (256*1008, )
        assert probs_st[0].data.max() <= 100 and probs_st[0].data.max() > 1
        assert probs_st[0].stats.station == "YMR"
        # Meta data file - 3c name should have E or 1 channel
        expected_json_file = f"{apply_detectors_outdir}/WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.json"
        assert os.path.exists(expected_json_file)
        with open(os.path.join(examples_dir, expected_json_file), "r") as fp:
            json_dict = json.load(fp)
        assert json_dict['channel'] == "HH?"

        os.remove(expected_p_probs_file)
        os.remove(expected_s_probs_file)
        os.remove(expected_json_file)

    def test_apply_to_multiple_days_1c(self):
        applier = apply_pyuussml_detectors.ApplyDetectorPyuussml(1, apply_detector_config)
        applier.apply_to_multiple_days("YWB", "EHZ", 2002, 1, 1, 2, debug_N_examples=-1)

        # Day 1
        expected_p_probs_file = f"{apply_detectors_outdir}/2002/01/01/probs.P__WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed"
        assert os.path.exists(expected_p_probs_file)
        probs_st = obspy.read(expected_p_probs_file)
        #assert probs_st[0].data.shape == (256*1008, )
        # There is no data at the beginning of this trace => no detections
        assert probs_st[0].data.max() <= 100 and probs_st[0].data.max() >= 0 
        assert probs_st[0].stats.station == "YWB"
        assert not os.path.exists(f"{apply_detectors_outdir}/2002/01/01/probs.S__WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed")
        expected_json_file = f"{apply_detectors_outdir}/2002/01/01/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.json"
        assert os.path.exists(expected_json_file)
        with open(os.path.join(examples_dir, expected_json_file), "r") as fp:
            json_dict = json.load(fp)
        assert json_dict['channel'] == "EHZ"

        # Check the starttime - should be no data from the previous day
        assert abs(probs_st[0].stats.starttime - (UTC('2002-01-01'))) < 0.01
        assert abs(UTC(json_dict['starttime']) - (UTC('2002-01-01'))) < 0.01
        assert probs_st[0].stats.starttime == UTC(json_dict['starttime'])

        os.remove(expected_p_probs_file)
        os.remove(expected_json_file)

        # Day 2
        expected_p_probs_file = f"{apply_detectors_outdir}/2002/01/02/probs.P__WY.YWB..EHZ__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed"
        assert os.path.exists(expected_p_probs_file)
        probs_st = obspy.read(expected_p_probs_file)
        # assert probs_st[0].data.shape == (256*1008, )
        # There is no data at the beginning of this trace => no detections
        assert probs_st[0].data.max() <= 100 and probs_st[0].data.max() >= 0 
        assert probs_st[0].stats.station == "YWB"
        assert not os.path.exists(f"{apply_detectors_outdir}/2002/01/02/probs.S__WY.YWB..EHZ__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed")
        expected_json_file = f"{apply_detectors_outdir}/2002/01/02/WY.YWB..EHZ__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.json"
        assert os.path.exists(expected_json_file)
        with open(os.path.join(examples_dir, expected_json_file), "r") as fp:
            json_dict = json.load(fp)
        assert json_dict['channel'] == "EHZ"

        # Check the starttime - should be 10 s of data from the previous day
        assert abs(probs_st[0].stats.starttime - (UTC('2002-01-02') - 10)) < 0.01
        assert abs(UTC(json_dict['starttime']) - (UTC('2002-01-02') - 10)) < 0.01
        assert probs_st[0].stats.starttime == UTC(json_dict['starttime'])

        os.remove(expected_p_probs_file)
        os.remove(expected_json_file)

    def test_apply_to_multiple_days_3c(self):
        applier = apply_pyuussml_detectors.ApplyDetectorPyuussml(3, apply_detector_config)
        applier.apply_to_multiple_days("YMR", "HH?", 2002, 1, 1, 2, debug_N_examples=-1)
        
        # Day 1
        # P Probs
        expected_p_probs_file = f"{apply_detectors_outdir}/2002/01/01/probs.P__WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed"
        assert os.path.exists(expected_p_probs_file)
        probs_st = obspy.read(expected_p_probs_file)
        #assert probs_st[0].data.shape == (256*1008, )
        assert probs_st[0].data.max() <= 100 and probs_st[0].data.max() > 1
        assert probs_st[0].stats.station == "YMR"
        assert abs(probs_st[0].stats.starttime - (UTC('2002-01-01'))) < 0.01

        # S probs
        expected_s_probs_file = f"{apply_detectors_outdir}/2002/01/01/probs.S__WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed"
        assert os.path.exists(expected_s_probs_file)
        probs_st = obspy.read(expected_s_probs_file)
        #assert probs_st[0].data.shape == (256*1008, )
        assert probs_st[0].data.max() <= 100 and probs_st[0].data.max() > 1
        assert probs_st[0].stats.station == "YMR"
        assert abs(probs_st[0].stats.starttime - (UTC('2002-01-01'))) < 0.01

        # Json File
        expected_json_file = f"{apply_detectors_outdir}/2002/01/01/WY.YMR..HHE__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.json"
        assert os.path.exists(expected_json_file)
        with open(os.path.join(examples_dir, expected_json_file), "r") as fp:
            json_dict = json.load(fp)
        assert json_dict['channel'] == "HH?"

        # Check the starttime - should be no data from the previous day
        assert abs(UTC(json_dict['starttime']) - (UTC('2002-01-01'))) < 0.01
        assert probs_st[0].stats.starttime == UTC(json_dict['starttime'])

        os.remove(expected_p_probs_file)
        os.remove(expected_s_probs_file)
        os.remove(expected_json_file)

        # Day 2
        expected_p_probs_file = f"{apply_detectors_outdir}/2002/01/02/probs.P__WY.YMR..HHE__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed"
        assert os.path.exists(expected_p_probs_file)
        probs_st = obspy.read(expected_p_probs_file)
        #assert probs_st[0].data.shape == (256*1008, )
        assert probs_st[0].data.max() <= 100 and probs_st[0].data.max() > 1
        assert probs_st[0].stats.station == "YMR"
        # Check the starttime - should be 10 s of data from the previous day
        assert abs(probs_st[0].stats.starttime - (UTC('2002-01-02') - 10)) < 0.01

        # S Probs
        expected_s_probs_file = f"{apply_detectors_outdir}/2002/01/02/probs.S__WY.YMR..HHE__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.mseed"
        assert os.path.exists(expected_s_probs_file)
        probs_st = obspy.read(expected_s_probs_file)
        #assert probs_st[0].data.shape == (256*1008, )
        assert probs_st[0].data.max() <= 100 and probs_st[0].data.max() > 1
        assert probs_st[0].stats.station == "YMR"       
        # Check the starttime - should be 10 s of data from the previous day
        assert abs(probs_st[0].stats.starttime - (UTC('2002-01-02') - 10)) < 0.01
        
        # Json File
        expected_json_file = f"{apply_detectors_outdir}/2002/01/02/WY.YMR..HHE__2002-01-02T00:00:00.000000Z__2002-01-03T00:00:00.000000Z.json"
        assert os.path.exists(expected_json_file)
        with open(os.path.join(examples_dir, expected_json_file), "r") as fp:
            json_dict = json.load(fp)
        assert json_dict['channel'] == "HH?"

        assert probs_st[0].stats.starttime == UTC(json_dict['starttime'])
        assert abs(UTC(json_dict['starttime']) - (UTC('2002-01-02') - 10)) < 0.01 

        os.remove(expected_p_probs_file)
        os.remove(expected_s_probs_file)
        os.remove(expected_json_file)

    def test_make_outfile_name_1c(self):
        pdet = apply_pyuussml_detectors.ApplyDetectorPyuussml(1, apply_detector_config)    
        file1 = f'{examples_dir}/WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        outfile = pdet.make_outfile_name("P", file1, examples_dir)
        assert os.path.basename(outfile) == 'probs.P__WY.YWB..EHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        assert os.path.dirname(outfile) == examples_dir

    def test_make_outfile_name_3c(self):
        pdet = apply_pyuussml_detectors.ApplyDetectorPyuussml(3, apply_detector_config)    
        file1 = f'{examples_dir}/WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        outfile = pdet.make_outfile_name("P",file1, examples_dir)
        assert os.path.basename(outfile) == 'probs.P__WY.YMR..HHZ__2002-01-01T00:00:00.000000Z__2002-01-02T00:00:00.000000Z.mseed'
        assert os.path.dirname(outfile) == examples_dir        

    def test_save_probs(self):
        pdet = apply_pyuussml_detectors.ApplyDetectorPyuussml(3, apply_detector_config)
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

        os.remove(outfile)

if __name__ == '__main__':
    dltester = TestApplyDetectorPyuussml()
    dltester.test_apply_to_file_day_1c()